# time production task with multiple intervals. The oven state indicates the interval to be timed (same as different coloured lights indicate different timing events)
# With undercooked_possible = False : Agent can take soup only after TD - overcooked has penalty (reduced_reward)
# With undercooked_possible = False : Agent can take soup anytime - overcooked and undercooked have penalty (reduced_reward) depending on the respective buffers

# Check 1, time production capabilities. Can agents produce time with some accuracy?
# Check 2, capacity to handle multiple intervals - Use curriculum learning - start with 1 duration, then add 2 durations then 3 (order 9,4,18,12,6,24)

# corrections
## V2 - observation space changed from high=20 to high=30 in order to accomodate different oven timers

## V3 - under and overcooked criteria (reward=0.1) for exact timing.
##    - compatible with cl for first no undercooked or overcooked criteria and add exct timing in later stages of training

##V4 - correction - check if oven is on befor taking soup in undercooked=True condition
##v5 - add oven duration in reset instead of init
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class MultiTimeProduction(gym.Env):


    # object code - 1- Agent, 3- Onion, 4- delivery counter,   5 - oven

    ### channel 1 - map # There is no orientation in the basic version so make sure the agent has only one object to interact at a given time

    #    [0., 3, 4],
    #    [0., 0., 0.],
    #    [5, 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0]

    ### channel 2 - Agent position
    #    [0., 0, 0],
    #    [0., 1, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 3 - carrying onion
    #    [0., 0, 0],
    #    [0., 1, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 4 - carrying soup
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 5 - oven states - depends on td
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [5., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

# oven_duration_states = {9: 3, 4: 1, 20: 5, 15: 4, 24: 6}

    def __init__(self, oven_duration_list= None, n_actions=6, max_timesteps=100, grid_size=(5, 3), n_channels=6, num_timers=1,
                 overcooked_buffer=0, undercooked_buffer=0, reduced_reward=0.1, undercooked_possible=False):

        if oven_duration_list is None:
            self.oven_duration_list = [9, 4, 20, 15, 24]
        else:
            self.oven_duration_list = oven_duration_list
        self.num_timers = num_timers
        self.onion_pos = (0, 1)
        self.delivery_pos = (0, 2)
        self.oven_pos = (2, 0)
        self.overcooked_buffer = overcooked_buffer # reduced reward for overcooked after TD+overcooked_buffer timesteps
        self.undercooked_buffer = undercooked_buffer
        self.max_timesteps = max_timesteps
        # self.grid_map[4,1,0] = 5
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.n_actions = n_actions
        self.oven_off_time = -1
        self.reduced_reward = reduced_reward
        self.undercooked_possible = undercooked_possible

        self.action_space = Discrete(self.n_actions)
        # changed from high=20 to accomodate different oven timers
        self.observation_space = Box(low=0, high=30, shape=(self.grid_map.shape), dtype=np.uint8)

    def get_obs(self, ):
        return self.grid_map

    def reset_oven_timers(self):
        self.oven_timer = -1
        self.oven_off_time = -1

    def reset(self, seed=None, oven_duration = None, options=None):
        self.timestep = 0
        self.reset_oven_timers()
        self.num_soups_delivered = 0

        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        self.grid_map[self.delivery_pos[0], self.delivery_pos[1], 0] = 4
        self.grid_map[self.oven_pos[0], self.oven_pos[1], 0] = 5

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

        #select oven duration for multitimer case
        if oven_duration is None:
            self.oven_duration = random.choice(self.oven_duration_list[:self.num_timers])
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
    # Reward calculation - 1 if soup delivered , 0 otherwise
    def interact(self) -> int:
        reward = 0
        adjacent_cells = self.get_adjacent_cells(self.agent_position)
        # check if there are more than one objects to interact at a time
        interact_obj_count = sum(1 for val in adjacent_cells.values() if val not in [None, 0.0])
        if interact_obj_count > 1:
            raise ValueError("more than 1 object to interact with", self.grid_map[:, :, 0])

        # interacts with onion dispenser and not carrying soup or onion already
        # Agent always carries 1 onion irrespective of oven duration
        if (self.onion_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)):  # carry onion
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 1

        # oven on: carrying onion and interacts with oven that is not on
        # Oven state depends on oven duration
        elif (self.oven_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1) and (self.oven_timer == -1):
            self.oven_timer = 0
            self.grid_map[self.oven_pos[0], self.oven_pos[1], 4] = self.oven_duration
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0

        # oven off: not carrying onion or soup and interacts with oven when oven time more than or equal to target duration
        # v3: agent can take soup anytime even before td. undercooked penalty 0.1
        # v4 but oven should have soup
        elif (self.oven_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)) \
                 and (self.oven_timer>=0 ) and ( self.oven_timer >= self.oven_duration or self.undercooked_possible): #if oven can open before td then release soup even before td else wait till td

            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 1
            self.oven_off_time = self.oven_timer
            self.oven_timer = -1
            self.grid_map[self.oven_pos[0], self.oven_pos[1], 4] = 0

        # interacts with delivery counter and carrying soup
        # Overcooked: reduce reward for overcooked
        # v3: rediced reward of runder and over cooked
        elif (self.delivery_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1):
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
            self.num_soups_delivered += 1
            if self.undercooked_possible: # case when oven can open before td
                if (self.oven_off_time <= self.oven_duration + self.overcooked_buffer) \
                        and (self.oven_off_time >= self.oven_duration - self.undercooked_buffer): #undercooked buffer
                    reward = 1
                else:
                    reward = self.reduced_reward
            else:  #oven cannot open before td
                if self.oven_off_time > self.oven_duration + self.overcooked_buffer:
                    reward = self.reduced_reward
                else:
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
        if self.oven_timer != -1:  # increment oven timer if oven already on
            self.oven_timer += 1

        if (self.timestep > self.max_timesteps) or (self.num_soups_delivered>0):  # terminate episode after max timesteps or soup delivery
            terminated = True
        else:
            terminated = False

        return self.get_obs(), reward, terminated, False, {}


