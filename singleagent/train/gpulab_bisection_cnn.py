import sys
sys.path.append("/project_ghent/chronocooked")

import os
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CheckpointCallback

# feature extractors
from singleagent.sb3_utils.feature_extractors.custom_cnn_simplified import CustomCNN

# load env
from singleagent.rl_environments.bisection_task import ChronoCookedBisectionTask

ent_coeff = 0
timestep = 200
g=0.99
#kl is None
short_duration = int(os.environ["SHORT_DURATION"])
long_duration = int(os.environ["LONG_DURATION"])


models_folder = "/project_ghent/chronocooked/singleagent/train/bisection_modelsv1"

for log in [1]:
    env = ChronoCookedBisectionTask( max_timesteps=timestep, short_duration=short_duration, long_duration=long_duration )


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64, n_channels=6),
        net_arch = dict(pi=[64], vf=[64])
    )

    model_save_path = f"{models_folder}/bisectionv1_TD1{env.short_duration}_TD2{env.long_duration}_cnnout64_ts{timestep}_entcoef{ent_coeff}_nepoch20_mlp64_gamma{g}_logs{log}/"

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1000, n_epochs=20, ent_coef=ent_coeff,
                         gamma = g, tensorboard_log=model_save_path)

    checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=model_save_path,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    # Train model
    total_timesteps = 960000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback,  tb_log_name="first_run", reset_num_timesteps=True)