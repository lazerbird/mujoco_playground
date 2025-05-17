import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

import numpy as np
import jax

from deploy_mujoco import *

class Custom:
    def __init__(self):
        self.Kp = 60.0
        self.Kd = 5.0
        self.time_consume = 0
        self.rate_count = 0
        self.sin_count = 0
        self.motiontime = 0
        self.dt = 0.002  # 0.001~0.01

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        # self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        #                      0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self._targetPos_2 = [0.1,  0.9, -1.8,
                            -0.1,  0.9, -1.8,
                            0.1,  0.9, -1.8,
                            -0.1,  0.9, -1.8]
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0

        self.firstRun = True
        self.done = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC() 

        self.jit_inference_fn, eval_env, _ = load_policy()
        self.rng = jax.random.PRNGKey(0)
        print("Loaded policy.")

        self.default_angles = np.array([
        0.1,  0.9, -1.8,   # front right
        -0.1,  0.9, -1.8,   # front left
        0.1,  0.9, -1.8,   # rear right
        -0.1,  0.9, -1.8    # rear left
        ], dtype=np.float32)

        self.action_scale = 0.5
        self.cmd_scale = np.array([1.5, 0.8, 2*np.pi])

        self.num_actions = eval_env.action_size # 12
        self.num_obs = eval_env.observation_size["state"] # 48

        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.counter = 0
        self.vel_gen = ContinuousVelocityWithCommandGenerator(stabilize_steps=2, walk_steps=500, stop_steps=2, 
                 max_vel=2, ramp_ratio=0.3, noise_std=0.005)
        self.policy_ongoing = False
        self.start_policy = 0
        self.last_policy_pose = None
        self.policy_duration = 25

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

    def LowCmdWrite(self):
        # return
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        # STAND TO DEFAULT POSE
        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            # print('phase1 over')
        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            # print('phase2 over')
        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self._targetPos_2[i] 
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            if self.percent_3 == 1:
                self.policy_ongoing = True
                self.start_policy = time.time()
                print("Policy started!")
                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.lowcmd_publisher.Write(self.low_cmd)
                return
            
        # START POLICY ACTIONS
        if self.policy_ongoing and (time.time() - self.start_policy < self.policy_duration):
            # Run your RL policy code here (as you already do)
            obs = self.get_obs()
            obs_dict = {"state": obs}
            act_rng, self.rng = jax.random.split(self.rng)
            action, _ = self.jit_inference_fn(obs_dict, act_rng)
            action = np.array(action)
            # print("Action", action)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.default_angles[i] + action[i] * self.action_scale
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

            self.last_action = action

        if self.policy_ongoing and (time.time() - self.start_policy > self.policy_duration):
            self.policy_ongoing = False

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            return

        # LAY DOWN AFTER POLICY
        if not self.policy_ongoing and (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
            if self.last_policy_pose is None:
                self.last_policy_pose = [self.low_state.motor_state[i].q for i in range(12)]

            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self.last_policy_pose[i] + self.percent_4 * self._targetPos_3[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


    def get_obs(self):
        imu = self.low_state.imu_state
        linvel, self.cmd = self.vel_gen.get()
        # self.cmd /= self.cmd_scale
        print(linvel, self.cmd)
        gyro = np.array(imu.gyroscope, dtype=np.float32)
        quat = self.low_state.imu_state.quaternion
        # gravity = np.array([0, 0, -1], dtype=np.float32)  # approximate, or use orientation matrix if needed
        gravity = get_gravity_orientation(quat)

        # def get_gravity_from_quaternion(quat):
        #     rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
        #     return rot.apply([0, 0, -1])  # Gravity vector in base frame

        joint_pos = np.array([m.q for m in self.low_state.motor_state[:self.num_actions]], dtype=np.float32)
        joint_vel = np.array([m.dq for m in self.low_state.motor_state[:self.num_actions]], dtype=np.float32)

        obs = np.concatenate([
            linvel,                  # 3
            gyro,                    # 3
            gravity,                 # 3
            joint_pos - self.default_angles,  # 12
            joint_vel,              # 12
            self.last_action,       # 12
            self.cmd            # 3
        ])
        # print("observation", obs)
        return obs


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        if custom.percent_4 == 1.0: 
           time.sleep(1)
           print("Done!")
           sys.exit(-1)     
        time.sleep(1)