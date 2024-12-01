import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from ga_language_env import GaLanguageEnv  # Adjust if necessary

# Load the trained model
model = PPO.load("ga_language_model")

# Initialize the environment
env = GaLanguageEnv()
obs = env.reset()

done = False
total_reward = 0
step_count = 0

# Lists to store rewards and steps for plotting
rewards = []
steps = []

print("\n--- Starting Model Test ---\n")

# Run the test
while not done:
    step_count += 1
    
    # Predict the action using the model
    action, _states = model.predict(obs)
    
    # Apply the action to the environment
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Store reward and step for plotting
    rewards.append(total_reward)
    steps.append(step_count)
    
    # Display step details
    print(f"Step {step_count}:")
    print(f"  Action Taken: {action}")
    print(f"  Reward Gained: {reward}")
    print(f"  Total Reward: {total_reward}")
    print(f"  Observation: {obs}")
    print(f"  Done: {done}\n")
    
    env.render()  # Visualize the environment if render is implemented

# Display the test summary
print("\n--- Test Completed ---\n")
print(f"Total Steps: {step_count}")
print(f"Total Reward: {total_reward}")

# Plotting the reward over time
plt.plot(steps, rewards)
plt.xlabel('Steps')
plt.ylabel('Total Reward')
plt.title('Agent Performance During Test')
plt.show()

