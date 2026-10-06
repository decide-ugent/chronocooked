# --------------------------------------------------------------------------------------------------
# Multi -oven Temporal bisection task - A 4x3 grid world with 2 delivery counters (short and long) and upto 3 ovens
# There is a primary oven (oven 0) that is triggered by the agent's action of putting onion in it (just like the bisection task)
# The agent has to deliver soup to a specific delivery counter depending on how long the soup has been inside a specific oven.
# Each oven in a given episode can be associated with a long or short duration
# Once ready, the ovens show a ready signal for agent to take the soup. The agent has to determine if the oven duration was long (delivery counter 1) or short (delivery counter 2).
# The agent is trained on two target durations corresponding to long and short. During test phase, the oven durations are varied to be closer to either short or long target duration.
# For asynchronous = True, asynchronous case, only primary oven is triggered by agent. Others are triggered by the environment at a random time  between 0 to 6+TDprimary/2
# For asynchronous = False, all ovens triggered at the same time as the primary oven
# The environment can be extended by increasing the grid to accommodate more ovens. It is recommended to keep the delivery counters centered and add grids on two sides
# ------------------------------------------------------------------------------------------------------

# corrections
## v2 - deactivate oven after 1 soup delivery - implements as after first oven on, oven 0 deactivated

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class ChronoCookedBisectionMultiOven(gym.Env):


    # object code - 1- Agent, 3- Onion, 4- delivery counter 1,   6,9,12 - oven, 5 - delivery counter 2

    ### channel 1 - map # There is no orientation in the basic version so make sure the agent has only one object to interact at a given time

    #    [4., 3,  5 ],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [6,  9,  12 ]

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
    #    [0., 6, 0.],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 5 - oven states - on, off, ready
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [0., 0., 0],
    #    [1, 2, 0.]

# oven states: 0 - off (no onion or soup), 1 -on , 2 - ready

    def __init__(self, short_duration=6, long_duration=12, n_actions=6, max_timesteps=100, grid_size=(4, 3),
                 n_channels=6, n_ovens=2, asynchronous=False):
        self.onion_pos = (0, 1)
        self.delivery1_pos = (0, 0) # short
        self.delivery2_pos = (0, 2) # long
        self.n_ovens = n_ovens
        self.oven_positions  = {}
        self.oven_timers = {}
        self.oven_states = {}
        self.oven_durations = {}
        self.delivery_counter_used = {}
        self.oven_on_times = {}
        for i in range(self.n_ovens):
            self.oven_positions[i] = (3,i)
            self.oven_timers[i] = -1
            self.oven_states[i] = 0
            self.delivery_counter_used[i] = -1 # short(1) or long(2) depending on which delivery counter was used for soup delivery
            self.oven_on_times[i] = -1

        self.max_timesteps = max_timesteps
        self.short_duration = short_duration
        self.long_duration = long_duration
        # self.grid_map[4,1,0] = 5
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.n_actions = n_actions
        self.oven0_deactivated = False

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=20, shape=(self.grid_map.shape), dtype=np.uint8)
        self.asynchronous = asynchronous


    def get_obs(self, ):
        return self.grid_map

    def reset_oven_timers(self):
        for i in range(self.n_ovens):
            self.oven_timers[i] = -1
            self.oven_states[i] = 0
            self.delivery_counter_used[i] = -1

    def reset(self, seed=None, oven_durations = [], options=None):
        self.timestep = 0
        self.reset_oven_timers()
        self.oven0_deactivated = False
        # self.delivery_counter_used = -1
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        self.grid_map[self.delivery1_pos[0], self.delivery1_pos[1], 0] = 4
        self.grid_map[self.delivery2_pos[0], self.delivery2_pos[1], 0] = 5
        for i in range(self.n_ovens):
            self.grid_map[self.oven_positions[i][0], self.oven_positions[i][1], 0] = 6 + 3*i

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

        for i in range(self.n_ovens):
            if len(oven_durations) == 0:
                self.oven_durations[i] = random.choice([self.short_duration, self.long_duration])
                print(f"oven duration selected for oven {i}", self.oven_durations[i])
            else:
                self.oven_durations[i] = oven_durations[i]

        # for asynch case, get oven on times for ovens. maximum time for oven 0 on in optimal policy is 6.
        # lower = max(0, 5 - (self.oven_durations[0] // 2))
        if self.asynchronous:
            upper = 6 + (self.oven_durations[0] // 2)
            lower = 0
            oven_on_sample_times = random.sample(range(lower, upper + 1), self.n_ovens - 1) # sample without replacememnnt so ovens do not have same start times
            for i, t in enumerate(oven_on_sample_times, start=1):
                self.oven_on_times[i] = t

            # create revsed dict for faster retrieval
        self.on_time_to_oven = {v: k for k, v in self.oven_on_times.items()}

        return self.get_obs(), {}

    # Move agent and calculate reward.
    # reward calculation - 0 for movement actions
    def move_agent(self, action) -> int:
        reward = 0
        carrying_onion = carrying_soup = False
        oven_identifier = 0
        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 0

        # check if agent is carrying onion
        if self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1:
            carrying_onion = True
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0
        if 1 in self.grid_map[:, :, 2]:
            raise ValueError("onion carrying channel contains incorrect information")

        # check if agent is carrying soup
        if self.grid_map[self.agent_position[0], self.agent_position[1], 3] >= 1: # changed to accomodate multi-oven case
            carrying_soup = True
            oven_identifier = self.grid_map[self.agent_position[0], self.agent_position[1], 3]
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
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = oven_identifier

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

        # oven on: carrying onion and interacts with oven 0 that is not on and not deactivates (due to soup delivery)
        # deactivate oven 0 after 1 soup delivery or after 1st oven on
        elif ((self.oven_positions[0] in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1) and
              (self.oven_timers[0] == -1) and (self.oven0_deactivated== False)):
            if self.asynchronous:
                n_oven_on = 1
            else:
                n_oven_on = self.n_ovens

            for i in range(n_oven_on):
                self.oven_timers[i] = 0
                self.oven_states[i] = 1
                self.grid_map[self.oven_positions[i][0], self.oven_positions[i][1], 4] = 1

            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0
            self.oven0_deactivated = True

        # oven off: not carrying onion or soup and interacts with oven when oven is ready
        elif all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0):
            for i in range(self.n_ovens):
                if (self.oven_positions[i] in adjacent_cells.keys()) and (self.oven_states[i] == 2): # oven ready
                    self.grid_map[self.agent_position[0], self.agent_position[1], 3] = self.grid_map[self.oven_positions[i][0], self.oven_positions[i][1], 0]
                    self.oven_timers[i] = -1
                    self.oven_states[i] = 0
                    self.grid_map[self.oven_positions[i][0], self.oven_positions[i][1], 4] = 0
                    break


        # interacts with delivery counter 1 and carrying soup
        # reward = 1, if oven timer = short else 0
        elif (self.delivery1_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] >= 1):

            oven_index = int((self.grid_map[self.agent_position[0], self.agent_position[1], 3] - 6)/3)
            self.delivery_counter_used[oven_index] = 1
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0

            if self.oven_durations[oven_index] == self.short_duration:
                reward = float(1/self.n_ovens)

        # interacts with delivery counter 2 and carrying soup
        # reward = 1, if oven timer = long else 0
        elif (self.delivery2_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] >= 1):

            oven_index = int((self.grid_map[self.agent_position[0], self.agent_position[1], 3] - 6) / 3)
            self.delivery_counter_used[oven_index] = 2
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0

            if self.oven_durations[oven_index] == self.long_duration:
                reward = float(1/self.n_ovens)

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

        # manage oven on for async case
        if self.asynchronous:
            oveni_on = self.on_time_to_oven.get(self.timestep)
            if oveni_on is not None:
                self.oven_timers[oveni_on] = 0
                self.oven_states[oveni_on] = 1
                self.grid_map[self.oven_positions[oveni_on][0], self.oven_positions[oveni_on][1], 4] = 1

        #increment oven timer if oven already on
        for i in range(self.n_ovens):
            if self.oven_timers[i] != -1:
                self.oven_timers[i] += 1

            # oven ready in obs for agent
            if self.oven_timers[i] == self.oven_durations[i]:
                self.oven_states[i] = 2
                self.grid_map[self.oven_positions[i][0], self.oven_positions[i][1], 4] = 2

        if (self.timestep > self.max_timesteps) or all(v > 0 for v in self.delivery_counter_used.values()):  # terminate episode after max timesteps or soup delivered from all ovens
            terminated = True
        else:
            terminated = False

        return self.get_obs(), reward, terminated, False, {}
