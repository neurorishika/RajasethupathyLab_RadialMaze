import gymnasium as gym
import numpy as np
from gymnasium import spaces
import datetime


class RadialMaze(gym.Env):
    """Custom Radial Maze Environment to simulate a mice's Radial Arm Maze Task"""

    metadata = {'render_modes': ['file','none']}

    def __init__(
            self,
            render_mode='none',
            N_arms = 8,
            arm_width = 0.1,
            arm_discretization = 10,
            animal_params = {
                'max_velocity': 0.05,
                'persistence': 0.5,
                'cost_of_exertion': 1e-3,
                'benefit_of_food': 1,
                'benefit_of_exploration': 1e-2,
                'exploratory_decay_time': 1,
                'cost_of_hitting_wall': 1,
            },
            experiment_params = {
                'reward_list': []
            },
            file_path = None,
            max_episode_steps=10000,
    ):
        """
        PARAMETERS
        N_arms : int
            Number of arms in the maze.
        arm_width : float
            Width of each arm.
        arm_discretization : int
            Number of bins to discretize each arm into.
        animal_params : dict
            Parameters of the animal.
        experiment_params : dict
            Parameters of the experiment.
        file_path : str
            Path to save the data to.
        max_episode_steps : int
            Maximum number of steps in an episode.
        """
        super().__init__()
        
        # Parameters
        self.N_arms = N_arms
        self.arm_width = arm_width
        self.arm_discretization = arm_discretization

        # Animal parameters
        self.max_velocity = animal_params['max_velocity'] if 'max_velocity' in animal_params else 0.05
        self.persistence = animal_params['persistence'] if 'persistence' in animal_params else 0.9
        self.cost_of_exertion = animal_params['cost_of_exertion'] if 'cost_of_exertion' in animal_params else 1e-3
        self.benefit_of_food = animal_params['benefit_of_food'] if 'benefit_of_food' in animal_params else 1
        self.benefit_of_exploration = animal_params['benefit_of_exploration'] if 'benefit_of_exploration' in animal_params else 1e-2
        self.exploratory_decay_time = animal_params['exploratory_decay_time'] if 'exploratory_decay_time' in animal_params else 1
        self.cost_of_hitting_wall = animal_params['cost_of_hitting_wall'] if 'cost_of_hitting_wall' in animal_params else 1

        # Experiment parameters
        self.backup_reward_list = experiment_params['reward_list'] if 'reward_list' in experiment_params else []


        self.reward_bound = 1/self.arm_discretization

        # Maximum number of steps in an episode
        self.max_episode_steps = max_episode_steps

        # File path
        self.file_path = file_path if file_path is not None else '../data/'+datetime.datetime.now().strftime("%Y%m%d-%H%M")+str(np.random.randint(1000))+'.csv'

        # Action space = 2D INTENDED VELOCITY VECTOR
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Observation space = 2D POSITION VECTOR + HEADING + SPEED + DISTANCE TO EDGE + DISTANCE TO NEAREST FOOD + HEADING TO NEAREST FOOD
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize state
        self.reset()
    
    def in_maze(self, x, y):
        """
        Check if the agent is in the maze.

        PARAMETERS
        x : float
            x-coordinate of the agent.
        y : float
            y-coordinate of the agent.

        OUTPUTS
        in_maze : bool
            True if the agent is in the maze, False otherwise.
        arm : int
            Index of the arm the agent is in.
        distance_to_edge : float
            Distance from the agent to the nearest edge.
        distance_in_arm : float
            Distance into the arm the agent is.
        """
        # get projection distances from the agent to each arm
        distances = np.zeros(self.N_arms)
        for i in range(self.N_arms):
            arm_angle = i * 2 * np.pi / self.N_arms
            point_angle = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi) # convert to [0, 2pi]
            if np.abs(point_angle - arm_angle) > np.pi/2 and np.abs(point_angle - arm_angle) < 3*np.pi/2: # if the point is behind the arm
                projection_distance = np.inf
            else: # if the point is in front of the arm
                projection_distance = np.abs(np.sin(point_angle - arm_angle))*np.sqrt(x**2 + y**2) # projection distance
            distances[i] = projection_distance
        # get the arm the agent is likely in
        arm = np.argmin(distances)
        # check if the agent is in the maze
        in_maze = (distances[arm] < self.arm_width/2 and (x**2 + y**2) < 1)
        # get distance to edge
        # check if in central area
        circumcircle_radius = self.arm_width/(2*np.sin(np.pi/self.N_arms))
        if np.sqrt(x**2 + y**2) < circumcircle_radius:
            distance_to_edge = np.max([circumcircle_radius - np.sqrt(x**2 + y**2), self.arm_width - distances[arm]])
        elif in_maze:
            distance_to_edge = self.arm_width - distances[arm]
        else:
            distance_to_edge = np.inf
        # distance_to_edge = np.min([self.arm_width - distances[arm], 1 - np.sqrt(x**2 - y**2)]) if np.sqrt(x**2 + y**2) > self.arm_width else self.arm_width - np.sqrt(x**2 + y**2)
        # get distance into arm
        distance_in_arm = np.abs(np.cos(np.arctan2(y, x) - (arm * 2 * np.pi / self.N_arms)))*np.sqrt(x**2 + y**2)
        return in_maze, arm, distance_to_edge, distance_in_arm
    
    def nearest_food(self,x, y, return_index=False, return_direction=False):
        """
        Get the distance to the nearest food.

        PARAMETERS
        x : float
            x-coordinate of the agent.
        y : float
            y-coordinate of the agent.
        return_index : bool
            If True, return the index of the food in the reward list.
        return_direction : bool
            If True, return the direction of the food.

        OUTPUTS
        distance_to_food : float
            Distance to the nearest food.
        index (optional) : int
            Index of the food in the reward list.
        direction (optional) : int
            Direction of the food.
        """
        # get arm
        _, arm, _, distance_in_arm = self.in_maze(x, y)
        # if N_arms is even, we look for food in that arm and the opposite arm
        if self.N_arms % 2 == 0:
            distance_to_food = [x[1]-distance_in_arm for x in self.reward_list if x[0] == arm]
            indices = [i for i, x in enumerate(self.reward_list) if x[0] == arm]
            same_arm = [1 for x in self.reward_list if x[0] == arm]
            distance_to_food += [x[1]+distance_in_arm for x in self.reward_list if x[0] == (arm + self.N_arms//2) % self.N_arms]
            indices += [i for i, x in enumerate(self.reward_list) if x[0] == (arm + self.N_arms//2) % self.N_arms]
            same_arm += [0 for x in self.reward_list if x[0] == (arm + self.N_arms//2) % self.N_arms]
        else:
            distance_to_food = [x[1]-distance_in_arm for x in self.reward_list if x[0] == arm]
            indices = [i for i, x in enumerate(self.reward_list) if x[0] == arm]
            same_arm = [1 for x in self.reward_list if x[0] == arm]

        if len(distance_to_food) > 0:
            dist = np.min(np.abs(distance_to_food))
            index = indices[np.argmin(np.abs(distance_to_food))]
            direction = arm*2*np.pi/self.N_arms if same_arm[np.argmin(np.abs(distance_to_food))] == 1 else (arm + self.N_arms//2)*2*np.pi/self.N_arms
            direction = (direction + np.pi) % (2*np.pi) - np.pi
            if distance_to_food[np.argmin(np.abs(distance_to_food))] < 0:
                direction = -direction
        else:
            dist = 2
            index = None
            direction = (arm*2*np.pi/self.N_arms + np.pi) % (2*np.pi) - np.pi
            direction = -direction
            
        if return_index and return_direction:
            return dist, index, direction
        elif return_index:
            return dist, index
        elif return_direction:
            return dist, direction
        else:
            return dist
    
    def _to_obs(self,distance_to_edge):
        # get position (between -1 and 1)
        x, y = self.position
        # get heading (between -1 and 1)
        heading = np.arctan2(self.velocity[1], self.velocity[0]) / np.pi
        # get speed (between 0 and 1)
        speed = np.linalg.norm(self.velocity) / self.max_velocity
        # get food properties
        distance_to_nearest_food, heading_to_nearest_food = self.nearest_food(x, y, return_direction=True)
        # get distance to nearest food (between 0 and 1)
        distance_to_nearest_food = distance_to_nearest_food / 2
        # direction of nearest_food (between -1 and 1)
        heading_to_nearest_food = heading_to_nearest_food / np.pi
        # return observation as a numpy array
        return np.array([x, y, heading, speed, distance_to_edge, distance_to_nearest_food, heading_to_nearest_food], dtype=np.float32)
    
    def _to_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)

        self.current_step = 0

        # start at a random position within arm_width of the center
        r = np.random.uniform(0, 1) * (self.arm_width/(2*np.sin(np.pi/self.N_arms)))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.position = np.array([x, y])

        # start with a no velocity
        self.velocity = np.array([0,0])

        # initialize exploration state
        self.exploration_state = np.zeros((self.N_arms, self.arm_discretization))

        # initialize observation and reward history
        self.observation_history = []
        self.reward_history = []

        # check if self.reward_list is a function (if so, call it)
        if callable(self.backup_reward_list):
            self.reward_list = self.backup_reward_list()
        # else, set reward list
        else:
            self.reward_list = self.backup_reward_list.copy()

        # get distance to edge
        _, _, distance_to_edge, _ = self.in_maze(x, y)

        # return observation and info
        obs = self._to_obs(distance_to_edge)
        info = self._to_info()
        return obs, info


    def step(self, action, debug=False):
        """
        Update the agents position and velocity based on action = heading and speed.
        """

        # increment step counter
        self.current_step += 1

        # initialize reward
        reward = 0

        intended_velocity = np.array([np.cos(action[0]*np.pi), np.sin(action[0]*np.pi)]) * (action[1]+1)/2 * self.max_velocity

        # update velocity (clip to max velocity)
        self.velocity = self.persistence * self.velocity + (1 - self.persistence) * intended_velocity
        if np.linalg.norm(self.velocity) > self.max_velocity:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_velocity
        
        # update reward based on velocity (wanting to move is expensive)
        reward -= self.cost_of_exertion * np.linalg.norm(action)
        if debug and self.cost_of_exertion * np.linalg.norm(action) > 0:
            print('punished for exertion')

        # update position and check if the agent is in the maze if dont update position
        x, y = self.position
        x += self.velocity[0]
        y += self.velocity[1]
        in_maze, _, _, _ = self.in_maze(x, y)
        if in_maze:
            self.position = np.array([x, y])
        else:
            self.velocity = np.zeros(2)
        # punish for leaving the maze
        if not in_maze:
            reward -= self.cost_of_hitting_wall
            if debug:
                print('punished for hitting wall')


        # get food reward
        nearest_food, index = self.nearest_food(x, y, return_index=True)
        if nearest_food < self.reward_bound:
            self.reward_list.pop(index)
            reward += self.benefit_of_food
            if debug:
                print('got food')

        # terminate if all food is eaten
        termination = len(self.reward_list) == 0 or self.current_step >= self.max_episode_steps
        reward += 0 if not termination else self.benefit_of_food * 2 if len(self.reward_list) == 0 else 0
        if debug and termination:
            print('terminated')

        # get distance to edge, arm, and distance into arm
        _, arm, distance_to_edge, distance_in_arm = self.in_maze(*self.position)

        obs = self._to_obs(distance_to_edge)
        info = self._to_info()

        # update exploration state
        self.exploration_state[arm, int(distance_in_arm * self.arm_discretization)] += 1

        # get exploration reward
        current_exploration_state = self.exploration_state[arm, int(distance_in_arm * self.arm_discretization)]
        exploration_rewards = self.benefit_of_exploration * np.exp(-current_exploration_state/self.exploratory_decay_time)

        reward += exploration_rewards
        if debug and exploration_rewards > 0:
            print('got exploration reward')

        # update observation history
        obs_ = obs.copy()
        obs_[2] = obs_[2] * np.pi
        obs_[3] = obs_[3] * self.max_velocity
        obs_[5] = obs_[5] * 2
        self.observation_history.append(obs_)
        self.reward_history.append(reward)

        return obs, reward, termination, False, info
    
    def render(self, mode='none'):
        if mode == 'file':
            with open(self.file_path, 'a') as f:
                # get observation vector
                _, _, distance_to_edge, _ = self.in_maze(*self.position)
                data = self._to_obs(distance_to_edge)
                data[2] = data[2] * np.pi
                data[3] = data[3] * self.max_velocity
                data[5] = data[5] * 2
                for i in range(len(data)):
                    f.write(f'{data[i]},')
                f.write('\n')
        elif mode == 'none':
            pass
        else:
            raise NotImplementedError(f'Unknown render mode {mode}')

