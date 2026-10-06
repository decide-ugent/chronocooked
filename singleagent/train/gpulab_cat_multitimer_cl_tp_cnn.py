import sys

import numpy as np
from stable_baselines3 import PPO


sys.path.append("/project_ghent/chronocooked")

import os

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# feature extractors
from singleagent.sb3_utils.feature_extractors.custom_cnn_simplified import CustomCNN

# load env
from singleagent.rl_environments.multitiming_production import MultiTimeProduction


class CurriculumCallback(BaseCallback):
    def __init__(self, thresholds_list, env_cat_multitimer_list, checkpoint_callback, window_size=100, verbose=1):
        super().__init__(verbose)
        self.thresholds = thresholds_list
        self.env_cat_multitimer_list = env_cat_multitimer_list
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
            self.model.set_env(self.env_cat_multitimer_list[self.current_level])
            obs = self.model.env.reset()
            self.model._last_obs = obs

            self.current_level += 1
            self.checkpoint_callback.name_prefix = f"rl_model_env{self.current_level}"
            # self.checkpoint_callback.save_freq = 20000
            self.prev_old_updates = n_updates

            print(f"Switched to level {self.current_level}")


#categorical multitimer case - agent can produce between 0 to TD1 for TD1, TD1 to TD2 for TD2 and so on.
# checks if agents can learn categories td1(red)<td2(green)<td3(blue)

# overcooked_buffer = int(os.environ["OVERCOOKED_BUFFER"])


# set thresholds for curriculum learning

thresholds = [0.95, 0.95, 0.90, 0.90, 0.90]

# oven_duration_list = [2,4,6,8,10] # must be equi-distant to incorporate hack for categorical multitimer case
oven_duration_list = [3,6,9,12,15] # change undercooked lower bound
# lstm_size =  int(os.environ["LSTM_SIZE"])


for log in [1,2,3,4]:
    ent_coeff = 0
    timestep = 200
    g=0.99

    # FI case - undercooked not possible so agent cannot take soup before td
    # num_timers =1 so oven_duration is random.choice(oven_duration_list[:1])
    # reduced reward for overcooked case is 0.1 unlike fi which is 0. To teach agent time <= td. The transition to time=td
    env0 = MultiTimeProduction(oven_duration_list=oven_duration_list, max_timesteps=timestep, num_timers=1,
                             reduced_reward=0.1, overcooked_buffer=0, undercooked_buffer=0, undercooked_possible=False)

    # cat timer with 1 td - undercooked possible and undercooked buffer = TD1
    # set reducd reward to 0 - inducing timing through reward function
    env1 = MultiTimeProduction(oven_duration_list=oven_duration_list, max_timesteps=timestep, num_timers=1,
                             reduced_reward=0, overcooked_buffer=0, undercooked_buffer=oven_duration_list[0], undercooked_possible=True)

    # cl envs
    env_cat_multitimer_list = [env1]
    for n in range(2, len(oven_duration_list)+1):
        undercooked_buffer = 3 # hack to incorporate categorical time production for the specific oven_duration_list
        env_i = MultiTimeProduction(oven_duration_list=oven_duration_list, max_timesteps=timestep, num_timers=n,
                                    reduced_reward=0, overcooked_buffer=0, undercooked_buffer=undercooked_buffer, undercooked_possible=True)
        env_cat_multitimer_list.append(env_i)


    duration_list_id = "-".join(map(str, oven_duration_list))

    models_folder = "/project_ghent/chronocooked/singleagent/train/multitimer_tp_cl_cat_modelsv2"



    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64, n_channels=6),
        net_arch = dict(pi=[64], vf=[64])
    )

    model_save_path = f"{models_folder}/v5_cl_cat_multitimer{duration_list_id}_tp_cnnout64_ts{timestep}_entcoef{ent_coeff}_nepoch20_mlp64_gamma{g}_logs{log}/"

    model = PPO("MlpPolicy", env0, policy_kwargs=policy_kwargs, verbose=1, n_steps=1000, n_epochs=20, ent_coef=ent_coeff,
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
        CurriculumCallback(thresholds_list=thresholds, env_cat_multitimer_list=env_cat_multitimer_list,
                           checkpoint_callback=checkpoint_callback)
    ])
    # Train model
    total_timesteps = 1760000
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="first_run",
                reset_num_timesteps=False)