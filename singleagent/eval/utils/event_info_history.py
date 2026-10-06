import numpy as np

def get_event_info_history(model, env, first_obs, timesteps=101):
    events_info_history = {}
    # print(f)
    # obs, _ = env.reset()
    obs = np.array(first_obs, copy=True)
    visual_obs = np.sum(obs, axis=2)
    # print("oven timers", env.oven1_timer, env.oven2_timer)
    rewards = 0
    states = None
    episode = 0
    reward = 0
    for i in range(timesteps):
        events_info_history[i] = {}
        # obs = obs + 300
        action, states = model.predict(obs, state=states, deterministic=True)
        # print(visual_obs)
        # print("action",action)
        events_info_history[i]["observation"] = visual_obs.copy()
        events_info_history[i]["oven_timer"] = env.oven1_timer
        events_info_history[i]["episode"] = episode  # or num_soups_delivered
        try:
            events_info_history[i]["clblock_value"] = env.clblock_value
        except:
            events_info_history[i]["clblock_value"] = 0

        # tag events
        if (env.grid_map[env.agent_position[0], env.agent_position[1], 2] == 1) and (
                env.oven1_timer == -1):  # carrying onion and oven off
            # events_action["put onion in oven"] = events_action.get("put onion in oven",[]) + [action]
            events_info_history[i]["event"] = "put onion in oven"

        elif (env.oven1_timer > -1):  # oven on
            # events_action["oven on"] = events_action.get("oven on",[]) + [action]
            events_info_history[i]["event"] = "oven on"
        elif (env.grid_map[env.agent_position[0], env.agent_position[1], 3] == 1) and (
                env.oven1_timer == -1):  # carrying soup and oven off
            # events_action["deliver soup"] = events_action.get("deliver soup",[]) + [action]
            events_info_history[i]["event"] = "deliver soup"
        elif all((env.grid_map[env.agent_position[0], env.agent_position[1], 2:4] == 0)) and (
                env.oven1_timer == -1):  # not carrying onion/soup and oven off
            # events_action["pick onion"] = events_action.get("pick onion",[]) + [action]
            events_info_history[i]["event"] = "pick onion"
        else:
            print("invalid game play", visual_obs)

        # step
        obs, reward, done, _, _ = env.step(action)
        visual_obs = np.sum(obs, axis=2)
        # print("reward",reward)
        events_info_history[i]["action"] = action
        events_info_history[i]["reward"] = reward
        events_info_history[i]["num_soups_delivered"] = env.num_soups_delivered
        try:
            events_info_history[i]["clblock_rewards"] = env.clblock_rewards
        except:
            events_info_history[i]["clblock_rewards"] = 0
        episode = env.num_soups_delivered
        rewards += reward
        if done:
            # obs, _ = env.reset()

            # print("rewards",rewards)
            break
    return events_info_history