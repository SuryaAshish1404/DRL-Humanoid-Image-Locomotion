import pybullet as p
import pybullet_data
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HumanoidWalkEnv(gym.Env):
    def __init__(self, urdf_path=None, use_gui=True):
        super().__init__()

        # --- Connect to PyBullet ---
        self.use_gui = use_gui
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        print("✅ Connected to PyBullet with gravity enabled.")

        # --- Load plane ---
        p.loadURDF("plane.urdf")

        # --- Load humanoid URDF ---
        if urdf_path is None:
            urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"❌ humanoid.urdf not found at {urdf_path}")
        self.humanoid_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.2], useFixedBase=False)
        print(f"✅ Humanoid loaded above the ground. ID: {self.humanoid_id}")

        # --- Define joints ---
        self.joint_order = [
            "base_to_torso", "neck",
            "right_shoulder", "right_elbow",
            "left_shoulder", "left_elbow",
            "right_hip", "right_knee",
            "left_hip", "left_knee"
        ]
        self.num_joints = len(self.joint_order)

        # --- Observation and action spaces ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints*2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        print(f"Number of joints: {self.num_joints}")
        print("Observation and action spaces defined.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset simulation ---
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
        self.humanoid_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.2], useFixedBase=False)

        # --- Hardcoded initial pose ---
        self.θ_init = [
            0.0,   # base_to_torso
            0.0,   # neck
            1,     # right_shoulder
            -1,    # right_elbow
            0.0,   # left_shoulder (fixed)
            0.0,   # left_elbow (fixed)
            0.7,   # right_hip
            -0.2,  # right_knee
            0.4,   # left_hip
            -0.1   # left_knee
        ]

        # --- Reset all joints according to order ---
        joint_indices = [p.getJointInfo(self.humanoid_id, i)[0] for i in range(p.getNumJoints(self.humanoid_id))]
        for idx, angle in zip(joint_indices[:self.num_joints], self.θ_init):
            p.resetJointState(self.humanoid_id, idx, targetValue=angle)

        print(f"Initial hardcoded pose applied: {self.θ_init}")

        return self._get_obs(), {}

    def _get_obs(self):
        joint_positions = []
        joint_velocities = []
        joint_indices = [p.getJointInfo(self.humanoid_id, i)[0] for i in range(p.getNumJoints(self.humanoid_id))]
        for idx in joint_indices[:self.num_joints]:
            js = p.getJointState(self.humanoid_id, idx)
            joint_positions.append(js[0])
            joint_velocities.append(js[1])
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def step(self, action):
        """
        Apply action to humanoid, step simulation, calculate reward, and check termination.
        Fixed joints: base_to_torso, neck, left_shoulder, left_elbow (action=0)
        """
        # Clip action
        action = np.clip(action, -1, 1)

        # Apply action to joints (fixed ones = 0)
        applied_action = self.θ_init.copy()
        for i, name in enumerate(self.joint_order):
            if name in ["base_to_torso", "neck", "left_shoulder", "left_elbow"]:
                continue  # action = 0 for fixed
            applied_action[i] += action[i]

        # Apply to PyBullet
        joint_indices = [p.getJointInfo(self.humanoid_id, i)[0] for i in range(p.getNumJoints(self.humanoid_id))]
        for idx, angle in zip(joint_indices[:self.num_joints], applied_action):
            p.setJointMotorControl2(self.humanoid_id, idx, p.POSITION_CONTROL, targetPosition=angle, force=100)

        # Step simulation
        p.stepSimulation()

        # --- Compute reward (simple: forward x movement) ---
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        reward = base_pos[0]  # reward = x forward

        # --- Check termination (fell) ---
        done = base_pos[2] < 0.1  # z < 0.1 → humanoid fell

        # --- Return observation ---
        obs = self._get_obs()
        return obs, reward, done, {}

# --- Test environment ---
if __name__ == "__main__":
    env = HumanoidWalkEnv(use_gui=True)
    obs, _ = env.reset()
    print(f"Observation after reset: {obs}")

    # Test step with random actions
    action = np.random.uniform(-0.1, 0.1, size=env.num_joints)
    obs, reward, done, _ = env.step(action)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}, Done: {done}")

    input("Press Enter to exit...")
    p.disconnect()
