# --------------------------------------------------------------------------------------------------
# Temporal bisection task but with >2 anchors - A Gym environment simulating a simplified single-agent version of OverCooked in a 4x3 grid world.
# It features 3 or 4 delivery counters instead of 2. 1 for each anchor
# It also features an onion dispenser, an oven. The agent has to deliver soup to a specific delivery counter depending on how long the soup has been inside the oven.
# Once ready, the oven shows a ready signal for agent to take the soup. The agent has to determine if the oven duration was T1 (delivery counter 1) or T2 (delivery counter 2) or T3 (delivery counter 3).
# The agent is trained on 3 or 4 anchor durations corresponding to each delivery counter. During test phase, the oven durations are varied to also be inbetween the anchor durations
# Adding more anchors is possible by increasing the width of the grid
# ------------------------------------------------------------------------------------------------------

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class ChronoCookedBisectionMultiAnchors(gym.Env):


    # object code - 1- Agent, 3- Onion, 7(TD1),8(TD2),9(TD3),10(TD4)- delivery counters,   4 - oven

    ### channel 1 - map # There is no orientation in the basic version so make sure the agent has only one object to interact at a given time

    #    [7., 3,  8 ],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [9, 4 , 10]

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

    def __init__(self, anchors = (4,8,12), n_actions=6, max_timesteps=100, grid_size=(4, 3), n_channels=6):
        self.onion_pos = (0, 1)

        xs = [0, 3] # Todo: change for different grid sizes
        ys = [0, 2]
        self.delivery_positions = {} # TD1 - delivery counter 1,TD2 - delivery counter 2, TD3 - delivery counter 3
        self.anchors = anchors # TD1, TD2, TD3 correspond to anchors
        for i in range(len(self.anchors)):
            x = xs[i // 2]
            y = ys[i % 2]
            self.delivery_positions[i] = (x, y)

        self.oven1_pos = (3, 1)

        self.max_timesteps = max_timesteps
        self.anchors = anchors

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
        self.delivery_counter_used = -1 # TD1,TD2,TD3,TD4 depending on which delivery counter was used for soup delivery
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        for i in range(len(self.anchors)):
            self.grid_map[self.delivery_positions[i][0], self.delivery_positions[i][1], 0] = 7+i

        self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 0] = 4

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
            self.oven_duration = random.choice(self.anchors)
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

        # Agent carrying soup and interacts with a delivery counter
        # reward: check if selected oven durations matched delivery counter used TD (using self.anchor) - index of delivery counter corresponds to index in anchor
        elif (self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1):
            for i in range(len(self.anchors)):
                if (self.delivery_positions[i] in adjacent_cells.keys()):
                    self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
                    self.delivery_counter_used = i+1 # (i+1 - for termination condition): 1 - TD1, 2-TD2, 3-TD3 , TD1,TD2, TD3 correspond to anchors

                    if self.oven_duration == self.anchors[i]:
                        reward = 1
                    break


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
