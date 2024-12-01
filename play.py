import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ga_language_env import GaLanguageEnv  # Custom environment

# Load the trained model
model = PPO.load("ga_language_model")

# Initialize the environment
env = GaLanguageEnv()
obs = env.reset()

# Define the grid size
grid_size = env.grid.shape  # Matches the custom environment's grid

# Create the plot for visualization
fig, ax = plt.subplots(figsize=(10, 10))  # Larger size for better visibility
ax.set_xlim(0, grid_size[1])
ax.set_ylim(0, grid_size[0])
ax.set_xticks(range(grid_size[1]))
ax.set_yticks(range(grid_size[0]))
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# Custom labels for the grid (representing greetings)
grid_labels = [
    ["Good Morning", "Ojekoo", "", "", ""],
    ["Good Afternoon", "Oshewɛɛ", "", "", ""],
    ["Good Evening", "Minaa Okooo", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "Goal", ""],
]

# Plot custom labels on the grid
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        label = grid_labels[i][j]
        if label:
            ax.text(j + 0.5, grid_size[0] - i - 0.5, label,
                    ha='center', va='center', fontsize=10, color='black')

# Initialize variables for tracking the agent's path and rewards
rewards_collected = 0  # To accumulate rewards
path_points = []       # To track the agent's path

done = False
step_count = 0

# Plot the goal position
goal_position = (3, 4)  # The goal cell
ax.scatter(goal_position[1] + 0.5, grid_size[0] - goal_position[0] - 0.5,
           color='red', label="Goal", s=200)

# Keep track of the current agent position
agent_position = env.agent_position

# Create text for steps and rewards (placed outside the grid, above)
step_text = ax.text(0.1, grid_size[0] + 0.05, f"Steps: {step_count}", fontsize=12)
reward_text = ax.text(0.1, grid_size[0] + 0.02, f"Reward: {rewards_collected}", fontsize=12)

print("\n--- Starting Model Test ---\n")

# Run the test
while not done:
    step_count += 1
    
    # Predict the action using the model
    action, _states = model.predict(obs)
    
    # Apply the action to the environment
    obs, reward, done, info = env.step(action)
    rewards_collected += reward
    
    # Get the new agent position
    new_agent_position = env.agent_position
    
    # Convert positions to grid coordinates
    old_x, old_y = agent_position % grid_size[1], grid_size[0] - (agent_position // grid_size[1]) - 1
    new_x, new_y = new_agent_position % grid_size[1], grid_size[0] - (new_agent_position // grid_size[1]) - 1
    
    # Update agent's position and path
    agent_position = new_agent_position
    path_points.append((new_x + 0.5, new_y + 0.5))  # Track the path

    # Redraw the grid elements without clearing the entire plot
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(range(grid_size[1]))
    ax.set_yticks(range(grid_size[0]))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Re-plot the goal position
    ax.scatter(goal_position[1] + 0.5, grid_size[0] - goal_position[0] - 0.5,
               color='red', label="Goal", s=200)

    # Plot the custom labels again
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            label = grid_labels[i][j]
            if label:
                ax.text(j + 0.5, grid_size[0] - i - 0.5, label,
                        ha='center', va='center', fontsize=10, color='black')

    # Plot the agent's path
    if len(path_points) > 1:
        path_x, path_y = zip(*path_points)
        ax.plot(path_x, path_y, color='blue', linewidth=2, linestyle='-')

    # Plot the agent's current position
    ax.scatter(new_x + 0.5, new_y + 0.5, color='green', label="Agent", s=200)
    
    # Update step and reward text (this avoids adding new text each time)
    step_text.set_text(f"Steps: {step_count}")
    reward_text.set_text(f"Reward: {rewards_collected}")
    
    # Update the plot without re-plotting the legend
    fig.canvas.draw_idle()  # This will update the plot without clearing
    plt.pause(0.5)  # Pause slightly longer for better visualization

# Display the final summary
print("\n--- Test Completed ---\n")
print(f"Total Steps: {step_count}")
print(f"Total Reward: {rewards_collected}")

# Keep the plot open for review
plt.show()

