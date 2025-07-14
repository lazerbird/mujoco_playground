# distill_student_policy.py

import os
import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import checkpoints
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from tqdm import tqdm
from brax.training.agents.ppo.networks import make_ppo_networks, make_inference_fn


# --- ENV & TEACHER SETUP ---
ENV_NAME = 'Go2JoystickFlatTerrain'
CHECKPOINT_PATH = '/home/charleney/projects/mujoco_playground/checkpoints/Go2JoystickFlatTerrain-20250404-201439-maybe/checkpoints/200540160'

# Load env
env_cfg = registry.get_default_config(ENV_NAME)
env = registry.load(ENV_NAME, config=env_cfg)
# print(registry.ALL_ENVS)
# Teacher obs key: 'privileged_state'
# ppo_params = registry.get_ppo_config(ENV_NAME)
ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
ppo_params.network_factory.policy_obs_key = 'privileged_state'
network_factory_teacher = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)

print("Loading teacher policy...")
temp_env = env
key = jax.random.PRNGKey(0)
nets_teacher = network_factory_teacher(temp_env.observation_size['privileged_state'], temp_env.action_size)
teacher_params = checkpoints.restore_checkpoint(CHECKPOINT_PATH, target=None)
teacher_policy = nets_teacher.policy_network
# print(type(teacher_params))
# print(teacher_params.keys(), teacher_params['0'].keys(),  teacher_params['1'].keys())
# params_teacher = (teacher_params['normalizer'], teacher_params['policy'])
# params_teacher = (None, teacher_policy)
params_teacher = (teacher_params['0'], teacher_params['1']['policy'])

# print("Available attributes:", dir(nets_teacher))

# --- STUDENT SETUP ---
# Student uses 'state' obs only
print("Loading student policy...")
ppo_params.network_factory.policy_obs_key = 'state'
network_factory_student = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
nets_student = network_factory_student(temp_env.observation_size['state'], temp_env.action_size)

obs_shape = temp_env.observation_size["state"]
if isinstance(obs_shape, tuple):
    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)  # e.g., (1, 45)
else:
    dummy_obs = jnp.zeros((1, obs_shape), dtype=jnp.float32)  # e.g., (1, 45)

student_params = nets_student.policy_network.init(key)
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(student_params)

# Make inference function
teacher_policy_fn = make_inference_fn(nets_teacher)(params_teacher, deterministic=True)
# teacher_policy_fn = make_inference_fn(nets_teacher)(params_teacher, deterministic=True)

# --- DATA COLLECTION ---
def collect_teacher_data(env, teacher_policy, teacher_params, num_steps=5):
    dataset = []
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    # for _ in range(num_steps):
    for _ in tqdm(range(num_steps), desc="Collecting teacher data"):
        # print(state.obs.keys())
        obs_priv = state.obs['state'] #  privileged_state
        obs_std = state.obs['state'][3:]
        action, _ = teacher_policy_fn(obs_priv, rng)
        # print(obs_priv.shape)
        # action = nets_teacher.policy_network.apply(
        #     {'params': teacher_policy},
        #     obs_priv[None, ...]  # Add batch dim if needed
        # )

        # action = nets_teacher.policy_apply_fn(teacher_params, obs_priv)
        dataset.append((obs_std, action))
        state = env.step(state, action)
        if state.done:
            rng, subkey = jax.random.split(rng)
            state = env.reset(subkey)
    obs_arr = jnp.stack([x[0] for x in dataset])
    act_arr = jnp.stack([x[1] for x in dataset])
    return obs_arr, act_arr

obs_data, act_data = collect_teacher_data(env, teacher_policy, teacher_params)

# --- DISTILLATION TRAINING ---
@jax.jit
def distill_step(params, opt_state, batch_obs, batch_act):
    def loss_fn(p):
        # pred = nets_student.policy_network.apply(p, batch_obs)
        # pred = nets_student.policy.apply(p, batch_obs)
        # print(p)
        pred = nets_student.policy_network.apply({'params': p}, batch_obs)
        # pred = nets_student.policy_network.apply(*p, batch_obs)
        return jnp.mean((pred - batch_act) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

batch_size = 256
num_epochs = 20

for epoch in range(num_epochs):
    permutation = np.random.permutation(len(obs_data))
    obs_data = obs_data[permutation]
    act_data = act_data[permutation]
    for i in range(0, len(obs_data), batch_size):
        batch_obs = obs_data[i:i+batch_size]
        batch_act = act_data[i:i+batch_size]
        student_params, opt_state, loss = distill_step(student_params, opt_state, batch_obs, batch_act)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Save distilled student
checkpoints.save_checkpoint('student_checkpoints', target=student_params, step=0, overwrite=True)
