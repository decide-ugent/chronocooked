import sys

sys.path.append("/project_ghent/chronocooked")

import os
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# feature extractors
from singleagent.sb3_utils.feature_extractors.custom_cnn_simplified import CustomCNN

# load env
from singleagent.rl_environments.time_production_basic import TimeProductionBasic

lstm_size = int(os.environ["LSTM_SIZE"])

for log in [1, 2, 3, 4]:
    for target_duration in [4,6,9,12,18,24]:
        ent_coeff = 0
        timestep = 200
        g=0.99

        env = TimeProductionBasic(oven_duration=target_duration, max_timesteps=timestep)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=64, n_channels=6),
            shared_lstm=True,  # share LSTM between actor & critic
            enable_critic_lstm=False,
            lstm_hidden_size=lstm_size,
            n_lstm_layers=1,
            net_arch=dict(pi=[64], vf=[64])
        )
        models_folder = "/project_ghent/chronocooked/singleagent/train/FI_modelsv1"

        model_save_path = f"{models_folder}/v1_FI{target_duration}_cnnout64_ts{timestep}_entcoef{ent_coeff}_nepoch20_sharedlstm{lstm_size}_mlp64_gamma{g}_logs{log}/"

        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=1000,
                             n_epochs=20, ent_coef=ent_coeff,
                             gamma = g, tensorboard_log=model_save_path)



        checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=model_save_path,
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

        total_timesteps = 900000
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="first_run",
                    reset_num_timesteps=True)

