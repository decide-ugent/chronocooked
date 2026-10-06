# script to run multi-oven case for more timesteps - can also serve as more runs/logs - saved in bisection_multioven_cl_models2

import sys

import numpy as np

sys.path.append("/project_ghent/chronocooked")

import os
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# feature extractors
from singleagent.sb3_utils.feature_extractors.custom_cnn_simplified import CustomCNN

# load env
from singleagent.rl_environments.bisection_multi_oven import ChronoCookedBisectionMultiOven

#import custom ctrnn policy
from singleagent.sb3_utils.custom_policies.ctrnn_policy import CTRNNPolicy


class CurriculumCallback(BaseCallback):
    def __init__(self, thresholds_list, env_multioven_list, checkpoint_callback, window_size=100, verbose=1):
        super().__init__(verbose)
        self.thresholds = thresholds_list
        self.env_multioven_list = env_multioven_list
        self.window_size = window_size
        self.rewards = []
        self.current_level = 0
        self.checkpoint_callback = checkpoint_callback
        self.prev_old_updates = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_info_buffer = self.model.ep_info_buffer
        n_updates = self.logger.name_to_value.get('train/n_updates')

        if len(ep_info_buffer) >= self.window_size:
            avg_reward = np.mean([ep_info["r"] for ep_info in ep_info_buffer])
        else:
            return

        # if n_updates is None:
        #     return

        # switch to next env
        if (self.current_level < len(self.thresholds)) and (
                avg_reward >= self.thresholds[self.current_level]):

            self.model.save(
                f"{self.checkpoint_callback.save_path}/"
                f"rl_model_env{self.current_level}_{self.num_timesteps}_steps"
            )


            self.rewards = []
            # self.model.policy.optimizer.state.clear() # commented to preserve adam optimizer state
            self.model.ep_info_buffer.clear()
            self.model.set_env(self.env_multioven_list[self.current_level])
            obs = self.model.env.reset()
            self.model._last_obs = obs

            self.current_level += 1
            self.checkpoint_callback.name_prefix = f"rl_model_env{self.current_level}"
            # self.checkpoint_callback.save_freq = 20000
            self.prev_old_updates = n_updates

            print(f"Switched to level {self.current_level}")


ent_coeff = 0
timestep = 200
g=0.99
# target kl is none
short_duration = int(os.environ["SHORT_DURATION"])
long_duration = int(os.environ["LONG_DURATION"])
max_ovens = 3

thresholds = [0.95, 0.90]

models_folder = "/project_ghent/chronocooked/singleagent/train/bisection_multioven_cl_models2"

for ctrnn_size in [125,256]:
    for log in [1,2,3,4]:
        for asynchronous in [True, False]:
            env0 = ChronoCookedBisectionMultiOven( max_timesteps=timestep, short_duration=short_duration, long_duration=long_duration,
                                                  asynchronous =asynchronous, n_ovens=1)

            env_multioven_list = []
            for n in range(2, max_ovens+1):
                env_i = ChronoCookedBisectionMultiOven( max_timesteps=timestep, short_duration=short_duration, long_duration=long_duration,
                                                  asynchronous =asynchronous, n_ovens=n)
                env_multioven_list.append(env_i)

            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=64, n_channels=6),
                shared_lstm=True,  # share LSTM between actor & critic
                enable_critic_lstm = False,
                lstm_hidden_size=ctrnn_size,
                n_lstm_layers=1,
                net_arch = dict(pi=[64], vf=[64])
            )
            multioven_suffix = "async" if asynchronous else "sync"
            model_save_path = f"{models_folder}/bisection_{max_ovens}oven_{multioven_suffix}_TD1{short_duration}_TD2{long_duration}_cnnout64_ts{timestep}_entcoef{ent_coeff}_nepoch20_ctrnn{ctrnn_size}_mlp64_gamma{g}_logs{log}/"

            model = RecurrentPPO(CTRNNPolicy, env0, policy_kwargs=policy_kwargs, verbose=1, n_steps=1000, n_epochs=20, ent_coef=ent_coeff,
                                 gamma = g, tensorboard_log=model_save_path)

            checkpoint_callback = CheckpointCallback(
                    save_freq=50000,
                    save_path=model_save_path,
                    name_prefix="rl_model_env0",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                )

            callback = CallbackList([
                checkpoint_callback,
                CurriculumCallback(thresholds_list=thresholds, env_multioven_list=env_multioven_list,
                                   checkpoint_callback=checkpoint_callback)
            ])
            # Train model
            total_timesteps = 1760000
            model.learn(total_timesteps=total_timesteps, callback=callback,  tb_log_name="first_run", reset_num_timesteps=False)