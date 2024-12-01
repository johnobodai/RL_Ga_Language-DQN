import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from ga_language_env import GaLanguageEnv

# Create a vectorized environment for stable-baselines3
env = make_vec_env(lambda: GaLanguageEnv(), n_envs=1)

# Define the DQN model using Stable-Baselines3
model = DQN(
    "MlpPolicy",  # MLP-based policy
    env,           # The custom environment
    verbose=1,     # Show training progress
    tensorboard_log="./dqn_ga_language_tensorboard/",  # For monitoring
    learning_starts=1000,  # Number of steps before starting training
    buffer_size=50000,  # Size of experience replay buffer
    exploration_fraction=0.1,  # Fraction of total steps for exploration
    exploration_final_eps=0.05,  # Final value for exploration probability
    policy_kwargs={"net_arch": [64, 64]},  # Neural network architecture
)

# Train the model
print("Training the agent...")
model.learn(total_timesteps=10000)  # Adjust timesteps for longer training

# Save the trained model
model.save("ga_language_model_dqn")  # Save model weights in H5 format
print("Model saved as 'ga_language_model_dqn'.")


