# --------------------------------------------------------------------------------------------------
# Temporal bisection task - A Gym environment simulating a simplified single-agent version of OverCooked in a 4x3 grid world.
# It features an onion dispenser, an oven, and two delivery counters. The agent has to deliver soup to a specific delivery counter depending on how long the soup has been inside the oven.
# Once ready, the oven shows a ready signal for agent to take the soup. The agent has to determine if the oven duration was long (delivery counter 1) or short (delivery counter 2).
# The agent is trained on two target durations corresponding to long and short. During test phase, the oven durations are varied to be closer to either short or long target duration.
# ------------------------------------------------------------------------------------------------------

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class ChronoCookedBisectionTask(gym.Env):


    # object code - 1- Agent, 3- Onion, 4- delivery counter 1,   5 - oven, 7 - delivery counter 2

    ### channel 1 - map # There is no orientation in the basic version so make sure the agent has only one object to interact at a given time

    #    [4., 3,  8 ],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 5 , 0 ]

    ### channel 2 - Agent position
    #    [0., 0, 0],
    #    [0., 1, 0],
    #    [0., 0, 0],
    #    [0., 0, 0]

    ### channel 3 - carrying onion
    #    [0., 0, 0],
    #    [0., 1, 0.],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 4 - carrying soup
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 5 - oven states
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [0., 0., 0],
    #    [0., 1, 0.]

# oven states: 0 - off (no onion or soup), 1 -on , 2 - ready

    def __init__(self, short_duration=6, long_duration=12, n_actions=6, max_timesteps=100, grid_size=(4, 3), n_channels=6):
        self.onion_pos = (0, 1)
        self.delivery1_pos = (0, 0) # short
        self.delivery2_pos = (0, 2) # long
        self.oven1_pos = (3, 1)

        self.max_timesteps = max_timesteps
        self.short_duration = short_duration
        self.long_duration = long_duration
        # self.grid_map[4,1,0] = 5
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.n_actions = n_actions

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=20, shape=(self.grid_map.shape), dtype=np.uint8)


    def get_obs(self, ):
        return self.grid_map

    def reset_oven_timers(self):
        self.oven1_timer = -1
        self.oven1_state = 0

    def reset(self, seed=None, oven_duration = None, options=None):
        self.timestep = 0
        self.reset_oven_timers()
        self.delivery_counter_used = -1 # short(1) or long(2) depending on which delivery counter was used for soup delivery
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        self.grid_map[self.delivery1_pos[0], self.delivery1_pos[1], 0] = 4
        self.grid_map[self.delivery2_pos[0], self.delivery2_pos[1], 0] = 8
        self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 0] = 5

        self.free_positions = [
            (i, j) for i in range(self.grid_map.shape[0])
            for j in range(self.grid_map.shape[1])
            if self.grid_map[i, j, 0] == 0
        ]

        # Randomly select a free position for the agent
        if seed:
            rng = random.Random(seed)
            self.agent_position = rng.choice(self.free_positions)
        else:
            self.agent_position = random.choice(self.free_positions)
        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 1

        # set oven duration: for training phase - long or short duration (50:50), for test - pre-determined by env param
        if oven_duration is None:
            self.oven_duration = random.choice([self.short_duration, self.long_duration])
            print("oven duration selected", self.oven_duration)
        else:
            self.oven_duration = oven_duration
        return self.get_obs(), {}

    # Move agent and calculate reward.
    # reward calculation - 0 for movement actions
    def move_agent(self, action) -> int:
        reward = 0
        carrying_onion = carrying_soup = False
        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 0

        # check if agent is carrying onion
        if self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1:
            carrying_onion = True
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0
        if 1 in self.grid_map[:, :, 2]:
            raise ValueError("onion carrying channel contains incorrect information")

        # check if agent is carrying soup
        if self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1:
            carrying_soup = True
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
        if 1 in self.grid_map[:, :, 3]:
            raise ValueError("soup carrying channel contains incorrect information")

        if action == 1:  # down
            new_pos = (self.agent_position[0] + 1, self.agent_position[1])
            if (new_pos in self.free_positions) and (new_pos[0] < self.grid_map.shape[0]):
                self.agent_position = new_pos

        elif action == 2:  # up
            new_pos = (self.agent_position[0] - 1, self.agent_position[1])
            if (new_pos in self.free_positions) and (new_pos[0] >= 0):
                self.agent_position = new_pos

        elif action == 3:  # right
            new_pos = (self.agent_position[0], self.agent_position[1] + 1)
            if (new_pos in self.free_positions) and (new_pos[1] < self.grid_map.shape[1]):
                self.agent_position = new_pos

        elif action == 4:  # left
            new_pos = (self.agent_position[0], self.agent_position[1] - 1)
            if (new_pos in self.free_positions) and (new_pos[1] >= 0):
                self.agent_position = new_pos

        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 1
        if carrying_onion:
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 1
        elif carrying_soup:
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 1

        return reward

    def get_adjacent_cells(self, pos):
        """
        Get adjacent non-diagonal cell values from grid map for a given position.
        """
        rows, cols = self.grid_map[:, :, 0].shape
        r, c = pos
        adjacent = {}

        # Up
        if r > 0:
            adjacent[(r - 1, c)] = self.grid_map[r - 1, c, 0]
        else:
            adjacent[(r - 1, c)] = None

        # Down
        if r < rows - 1:
            adjacent[(r + 1, c)] = self.grid_map[r + 1, c, 0]
        else:
            adjacent[(r + 1, c)] = None

        # Left
        if c > 0:
            adjacent[(r, c - 1)] = self.grid_map[r, c - 1, 0]
        else:
            adjacent[(r, c - 1)] = None

        # Right
        if c < cols - 1:
            adjacent[(r, c + 1)] = self.grid_map[r, c + 1, 0]
        else:
            adjacent[(r, c + 1)] = None

        return adjacent

    # Agent interacts with an object -
    # pick onion from dispenser, drop onion in oven, pick soup, drop soup at delivery counter
    # Reward calculation - 1 if soup delivered in correct delivery counter, 0 otherwise
    def interact(self) -> int:
        reward = 0
        adjacent_cells = self.get_adjacent_cells(self.agent_position)
        # check if there are more than one objects to interact at a time
        interact_obj_count = sum(1 for val in adjacent_cells.values() if val not in [None, 0.0])
        if interact_obj_count > 1:
            raise ValueError("more than 1 object to interact with", self.grid_map[:, :, 0])

        # interacts with onion dispenser and not carrying soup or onion already
        if (self.onion_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)):  # carry onion
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 1

        # oven on: carrying onion and interacts with oven that is not on
        elif (self.oven1_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1) and (self.oven1_timer == -1):
            self.oven1_timer = 0
            self.oven1_state = 1
            self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 4] = 1
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0

        # oven off: not carrying onion or soup and interacts with oven when oven is ready
        elif (self.oven1_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)) and (
                self.oven1_state ==2): # oven ready
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 1
            self.oven1_timer = -1
            self.oven1_state = 0
            self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 4] = 0

        # interacts with delivery counter 1 and carrying soup
        # reward = 1, if oven timer = short else 0
        elif (self.delivery1_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1):
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
            self.delivery_counter_used = 1

            if self.oven_duration == self.short_duration:
                reward = 1

        # interacts with delivery counter 2 and carrying soup
        # reward = 1, if oven timer = long else 0
        elif (self.delivery2_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1):
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
            self.delivery_counter_used = 2

            if self.oven_duration == self.long_duration:
                reward = 1

        return reward

    # actions: 0:down, 1-up, 2-right, 3-left, 4- interact
    # reward calculation: 1 for soup delivery and 0 otherwise
    def step(self, action):
        self.timestep += 1
        if action != 5:  # navigation action
            reward = self.move_agent(action)
        else:
            # interact with object
            reward = self.interact()
        if self.oven1_timer != -1:  # increment oven timer if oven already on
            self.oven1_timer += 1

        if self.oven1_timer == self.oven_duration:
            self.oven1_state = 2
            self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 4] = 2 # oven ready in obs for agent

        if (self.timestep > self.max_timesteps) or (self.delivery_counter_used > 0):  # terminate episode after max timesteps or soup delivery in any delivery counter
            terminated = True
        else:
            terminated = False

        return self.get_obs(), reward, terminated, False, {}
