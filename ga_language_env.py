import gym
from gym import spaces
import numpy as np

class GaLanguageEnv(gym.Env):
    """
    Custom Gym environment for navigating a learning grid
    to preserve and teach the Ga language.
    """
    def __init__(self):
        super(GaLanguageEnv, self).__init__()
        self.grid_size = (5, 5)

        # Define action space (4 directions: up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space (5x5 grid: 25 states)
        self.observation_space = spaces.Discrete(25)

        # Initial position of the agent (top-left corner)
        self.agent_position = 0

        # Define grid layout with rewards
        # -1: empty cell, 10: resource, 50: goal, -5: distraction
        self.grid = np.array([
            [-1, -1, -1, -1, 10],
            [-1, -5, -1, -1, -1],
            [-1, 10, -1, -5, -1],
            [-1, -1, 10, -1, -1],
            [-1, -1, -1, -1, 50]
        ])

        # Keep track of visited cells
        self.path = np.zeros_like(self.grid, dtype=int)

        # Goal position (bottom-right corner)
        self.goal_position = 24

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.agent_position = 0
        self.path.fill(0)  # Clear the path tracking
        return self.agent_position

    def step(self, action):
        """
        Take an action and return the next state, reward,
        done flag, and additional info.
        """
        # Define movement directions
        row, col = divmod(self.agent_position, self.grid_size[1])

        if action == 0 and row > 0:  # Move up
            row -= 1
        elif action == 1 and row < self.grid_size[0] - 1:  # Move down
            row += 1
        elif action == 2 and col > 0:  # Move left
            col -= 1
        elif action == 3 and col < self.grid_size[1] - 1:  # Move right
            col += 1

        # Update agent position
        new_position = row * self.grid_size[1] + col

        # Update the path to mark the current cell as visited
        self.path[row, col] = 1

        # Get the reward for the current cell
        reward = self.grid[row, col]

        # Check if the agent has reached the goal
        done = new_position == self.goal_position

        self.agent_position = new_position

        return self.agent_position, reward, done, {}

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        grid_display = self.grid.copy()

        # Mark the agent's position
        row, col = divmod(self.agent_position, self.grid_size[1])
        grid_display[row, col] = 99  # Mark agent position

        # Display the path
        print("Grid with Path:")
        for i in range(self.grid_size[0]):
            row_display = [
                "A" if (i, j) == (row, col) else
                "*" if self.path[i, j] == 1 else
                str(grid_display[i, j]) for j in range(self.grid_size[1])
            ]
            print(" ".join(row_display))

        # Legend and reward display
        print("\nLegend:")
        print("99 (A): Agent Position")
        print("*: Path Taken")
        print("-1: Empty Cell")
        print("10: Resource")
        print("-5: Distraction")
        print("50: Goal (Ojekoo)\n")

