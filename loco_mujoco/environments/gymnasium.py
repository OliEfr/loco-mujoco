import numpy as np

from loco_mujoco import LocoEnv

from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box


class GymnasiumWrapper(Env):
    """
    This class implements a simple wrapper to use all LocoMuJoCo environments
    with the Gymnasium interface.

    """

    def __init__(self, env_name, render_mode=None, latent_action_space_dim=False, use_expert_data=False, **kwargs):
        self.spec = EnvSpec(env_name)
        self.metadata = {"render_modes": ["human", "rgb_array"]}

        self.latent_action_space_dim = latent_action_space_dim

        if "UnitreeH1" in env_name:
            assert env_name == "UnitreeH1.walk.perfect" or env_name == "UnitreeH1.walk", f"Only UnitreeH1.walk is supported for now. Used {env_name} instead."

        self.use_expert_data = use_expert_data
        if self.use_expert_data:
            expert_data = np.load("expert_demonstrations/Unitree_A1_H1/UnitreeH1.walk.perfect.npy", allow_pickle=True).item()
            self.expert_obs_cycle = expert_data["recorded_obs_cycle"].copy()
            self.phase = 0

        key_render_mode = "render_modes"
        assert "headless" not in kwargs.keys(), f"headless parameter is not allowed in Gymnasium environment. " \
                                                f"Please use the render_mode parameter. Supported modes are: " \
                                                f"{self.metadata[key_render_mode]}"
        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Unsupported render mode: {render_mode}. " \
                                                                 f"Supported modes are: " \
                                                                 f"{self.metadata[key_render_mode]}."

        self.render_mode = render_mode

        # specify the headless based on render mode to initialize the LocoMuJoCo environment
        if render_mode == "human":
            kwargs["headless"] = False
        else:
            kwargs["headless"] = True

        self._env = LocoEnv.make(env_name, use_expert_data = self.use_expert_data, **kwargs)
        self.metadata["render_fps"] = 1.0 / self._env.dt
        if self.use_expert_data:
            self._env.phase = self.phase

        self.observation_space = self._convert_space(self._env.info.observation_space)
        self._set_action_space()
        # self.action_space = self._convert_space(self._env.info.action_space)

        
    def imitating_joint_pos_reward(self):
        joint_pos = self._env._data.qpos[2:].copy()
        joint_pos_ref = self.expert_obs_cycle[self.phase][
            :15
        ]
        return np.exp(-0.5 * np.linalg.norm(joint_pos_ref - joint_pos))

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        .. note:: We set the truncation always to False, as the time limit can be handled by the `TimeLimit`
         wrapper in Gymnasium.

        Args:
            action (np.ndarray):
                The action to be executed in the environment.

        Returns:
            Tuple of observation, reward, terminated, truncated and info.

        """

        obs, reward, absorbing, info = self._env.step(action)
        if self.use_expert_data:
            self.phase += 1
            self.phase = self.phase % len(self.expert_obs_cycle)
            self._env.phase = self.phase

            style_reward = self.imitating_joint_pos_reward()
            task_reward = reward
            reward = 0.33 * style_reward + 0.67 * task_reward
            info["style_reward"] = style_reward
            info["joint_pos_reward"] = style_reward
        else:
            task_reward = reward
        info["task_reward"] = task_reward

        return obs, reward, absorbing, False, info

    def reset(self, *, seed=None, options=None):
        """
        Resets the state of the environment, returning an initial observation and info.

        """
        if self.use_expert_data:
            self.phase=0
            self._env.phase = self.phase


        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self._env.reset(), {}

    def render(self):
        """
        Renders the environment.

        """
        if self.render_mode == "human":
            self._env.render()
        elif self.render_mode == "rgb_array":
            img = self._env.render(True)
            return np.swapaxes(img, 0, 1)

    def close(self):
        """
        Closes the environment.

        """
        self._env.stop()

    def create_dataset(self, **kwargs):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj). For perfect and preference datasets, the actions are also provided.

        """
        return self._env.create_dataset(**kwargs)

    def play_trajectory(self, **kwargs):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """
        return self._env.play_trajectory(**kwargs)

    def play_trajectory_from_velocity(self, **kwargs):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones calculated from the joint velocities
        in the trajectories at every step. Therefore, the joint positions
        are set from the trajectory in the first step. Afterwards, numerical
        integration is used to calculate the next joint positions using
        the joint velocities in the trajectory.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the replay will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """
        return self._env.play_trajectory_from_velocity(**kwargs)

    @property
    def unwrapped(self):
        """
        Returns the inner environment.
        """
        return self._env

    @staticmethod
    def _convert_space(space):
        """ Converts the observation and action space from mushroom-rl to gymnasium. """
        low = np.min(space.low)
        high = np.max(space.high)
        shape = space.shape
        return Box(low, high, shape, np.float64)
    
    def _set_action_space(self):
        if self.latent_action_space_dim:
            bounds = np.full((self.latent_action_space_dim, 2), [-1.0, 1.0]).astype(
                np.float32
            )
            low, high = bounds.T
            self.action_space = Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = self._convert_space(self._env.info.action_space)
        return self.action_space
