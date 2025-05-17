import time
import mujoco.viewer
import mujoco
import numpy as np
import argparse
import jax
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
import etils.epath as epath
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def_pth = "Go2JoystickFlatTerrain-20250404-201439-maybe/checkpoints"

def load_policy(ENV_NAME="Go2JoystickFlatTerrain", CHECKPOINT_PATH=def_pth, SEED=0):
    print("Loading environment and policy...")
    env_cfg = registry.get_default_config(ENV_NAME)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    ppo_params.num_timesteps = 0
    env = registry.load(ENV_NAME, config=env_cfg)

    ckpt_path = epath.Path(CHECKPOINT_PATH).resolve()
    if ckpt_path.is_dir():
        latest_ckpts = list(ckpt_path.glob("*"))
        latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
        latest_ckpts.sort(key=lambda x: int(x.name))
        restore_checkpoint_path = latest_ckpts[-1]
        print(f"Restoring from: {restore_checkpoint_path}")

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    network_fn = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(network_fn, **ppo_params.network_factory)
    else:
        network_factory = network_fn

    num_eval_envs = ppo_params.get("num_eval_envs", 128)
    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    def policy_params_fn(current_step, make_policy, params):
        return

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        policy_params_fn=policy_params_fn,
        seed=SEED,
        restore_checkpoint_path=restore_checkpoint_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

    def progress(num_steps, metrics):
        pass


    env_cfg.pert_config.enable = False
    # env_cfg.pert_config.velocity_kick = [3.0, 6.0]
    # env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
    env_cfg.command_config.a = [1.5, 0.8, 2*np.pi]

    eval_env = registry.load(ENV_NAME, config=env_cfg)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env=eval_env)
    running_stats = params[0]
    rest = params[1:]

    # Replace mean/std inside the RunningStatisticsState
    running_stats = running_stats.replace(
        mean={'state': running_stats.mean['state']},
        std={'state': running_stats.std['state']}
    )
    # Reconstruct the params tuple
    params = (running_stats,) + rest

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    return jit_inference_fn, eval_env, env_cfg

def get_sensor_data(
    model: mujoco.MjModel, data: mujoco.mjx.Data, sensor_name: str
):
  """Gets sensor data given sensor name."""
  sensor_id = model.sensor(sensor_name).id
  sensor_adr = model.sensor_adr[sensor_id]
  sensor_dim = model.sensor_dim[sensor_id]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

def quat_to_rotmat(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def estimate_linvel(acc_body, quat, prev_vel, dt):
    # Convert quaternion to rotation matrix (body to world)
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x,y,z,w]
    acc_world = rot.apply(acc_body)

    # Subtract gravity
    acc_world -= np.array([0, 0, 9.81])

    # Integrate to estimate velocity
    vel = prev_vel + acc_world * dt
    return vel

class ContinuousVelocityWithCommandGenerator:
    def __init__(self, stabilize_steps=10, walk_steps=500, stop_steps=10, 
                 max_vel=1.5, ramp_ratio=0.3, noise_std=0.005):
        self.stabilize_steps = stabilize_steps
        self.walk_steps = walk_steps
        self.stop_steps = stop_steps
        self.max_vel = max_vel
        self.ramp_ratio = ramp_ratio
        self.noise_std = noise_std

        self.total_steps = stabilize_steps + walk_steps + stop_steps
        self.counter = 0

        self.ramp_up_steps = int(walk_steps * ramp_ratio)
        self.ramp_down_steps = int(walk_steps * ramp_ratio)
        self.steady_steps = walk_steps - self.ramp_up_steps - self.ramp_down_steps

    def ease_in_out(self, x):
        # Smoothstep (0 to 1): x in [0, 1]
        return 3*x**2 - 2*x**3

    def get(self):
        vel = np.zeros(3, dtype=np.float32)
        command = np.zeros(3, dtype=np.float32)

        if self.counter < self.stabilize_steps:
            pass
        elif self.counter < self.stabilize_steps + self.walk_steps:
            walk_idx = self.counter - self.stabilize_steps
            if walk_idx < self.ramp_up_steps:
                x = walk_idx / self.ramp_up_steps
                vx = self.ease_in_out(x) * self.max_vel
            elif walk_idx < self.ramp_up_steps + self.steady_steps:
                vx = self.max_vel
            else:
                x = (walk_idx - self.ramp_up_steps - self.steady_steps) / self.ramp_down_steps
                vx = (1 - self.ease_in_out(x)) * self.max_vel

            vx_noisy = vx + np.random.normal(0, self.noise_std)
            vel = np.array([vx_noisy, 0.0, 0.0], dtype=np.float32)
            command = np.array([vx, 0.0, 0.0], dtype=np.float32)
        else:
            pass

        self.counter += 1
        return vel, command

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    x_vel, y_vel, yaw_vel = 0.65, 0, 0
    command = np.array([x_vel, y_vel, yaw_vel], dtype=np.float32)

    jit_inference_fn, eval_env, env_cfg = load_policy(args.env_name, args.checkpoint, args.seed)

    simulation_duration = 20.0
    simulation_dt = getattr(env_cfg, 'sim_dt', 0.004)
    control_decimation = int(env_cfg.ctrl_dt / env_cfg.sim_dt) if hasattr(env_cfg, 'sim_dt') else 2

    kps = np.full(12, getattr(env_cfg, 'Kp', 35.0), dtype=np.float32)
    kds = np.full(12, getattr(env_cfg, 'Kd', 0.5), dtype=np.float32)
    # kps = np.full(12, 60, dtype=np.float32)
    # kds = np.full(12, 5, dtype=np.float32)

    default_angles = np.array([
        0.1,  0.9, -1.8,   # front right
    -0.1,  0.9, -1.8,   # front left
        0.1,  0.9, -1.8,   # rear right
    -0.1,  0.9, -1.8    # rear left
    ], dtype=np.float32)

    # ang_vel_scale = getattr(env_cfg, 'ang_vel_scale', 1.0)
    # dof_pos_scale = getattr(env_cfg, 'dof_pos_scale', 1.0)
    # dof_vel_scale = getattr(env_cfg, 'dof_vel_scale', 1.0)
    action_scale = getattr(env_cfg, 'action_scale', 0.5)
    cmd_scale = np.array(env_cfg.command_config.a, dtype=np.float32) if hasattr(env_cfg, 'command_config') else np.ones(3, dtype=np.float32)
    # num_actions = 12 # env_cfg.num_action
    num_actions = eval_env.action_size
    num_obs = eval_env.observation_size["state"]

    # command /= cmd_scale
    og_command = command.copy()

    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # velocity_kick_range = [0.0, 0.0]  # Disable velocity kick.
    # kick_duration_range = [0.05, 0.2]

    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    current_path = Path(__file__).resolve()
    xml_path = current_path.parent / 'mujoco_playground' / 'external_deps' / 'mujoco_menagerie' / 'unitree_go2' / 'scene_mjx2.xml'
    # xml_path = '/home/laser/projects/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/unitree_go2/scene_mjx2.xml'
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    dt = m.opt.timestep

    stabilize_duration_sec = 2.0  # seconds to stabilize before walking
    stabilize_steps = int(stabilize_duration_sec / (control_decimation * simulation_dt))

    walk_duration_sec = 3.0  # walk for 2 seconds
    walk_steps = int(walk_duration_sec / (control_decimation * simulation_dt))

    d.qpos[7:] = default_angles.copy()
    d.qvel[:] = 0
    action =  np.zeros(3, dtype=np.float32)
    # linvel_log = []
    vel_gen = ContinuousVelocityWithCommandGenerator()

    # d.qpos[2] = 0.278  # raise z height (try 0.35–0.4 for Go2)
    # d.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion (no rotation)

    # jit_reset = jax.jit(eval_env.reset)
    # jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(args.seed)

    with mujoco.viewer.launch_passive(m, d) as viewer:

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:

            step_start = time.time()

            if counter // control_decimation < stabilize_steps:
                command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Stand still
            elif counter // control_decimation < stabilize_steps + walk_steps:
                command = og_command  # Walk forward
            else:
                command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Stop

            if counter % control_decimation == 0:
                # -- Step 1: Extract from mujoco data --
                qpos = d.qpos
                qvel = d.qvel

                joint_pos = qpos[7:]  # shape (12,)
                joint_vel = qvel[6:]  # shape (12,)

                gyro = get_sensor_data(m, d, "gyro")

                # linvel = np.array([0,0,0])
                # linvel = get_sensor_data(m, d, "local_linvel")
                # elapsed_time = time.time() - start
                linvel, command = vel_gen.get()
                print(linvel, command)
                # command /= cmd_scale

                imu_id = m.site("imu").id
                gravity = d.site_xmat[imu_id].reshape(3, 3).T @ np.array([0, 0, -1])
                # gravity = rot.apply(np.array([0, 0, -1]))

                obs = np.concatenate([
                    linvel,                                    # 3
                    gyro,                                      # 3
                    gravity,                                   # 3
                    joint_pos - default_angles,                # 12
                    joint_vel,                                 # 12
                    last_action,                               # 12
                    command                                    # 3
                ], dtype=np.float32)                           # Total = 48
                # print
                obs_dict = {"state": obs}
                act_rng, rng = jax.random.split(rng)

                action, _ = jit_inference_fn(obs_dict, act_rng)
                action = np.array(action)
                print("Action:", action)

                last_action = action
                target_dof_pos = default_angles + action * action_scale

            d.ctrl[:] = target_dof_pos
            mujoco.mj_step(m, d)

            counter += 1

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # print('Done!')
    # logger.plot()
