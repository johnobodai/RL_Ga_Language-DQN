from ga_language_env import GaLanguageEnv

# Create an instance of the custom environment
env = GaLanguageEnv()

# Reset the environment
state = env.reset()
print(f"Initial state: {state}")

# Simulate actions
for action in [3, 3, 1, 1, 1]:  # Example actions: right, right, down, down, down
    next_state, reward, done, info = env.step(action)
    print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
    env.render()
    
    if done:
        print("Goal reached!")
        break

