from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ga_language_env import GaLanguageEnv

# Create a vectorized environment for stable-baselines3
env = make_vec_env(lambda: GaLanguageEnv(), n_envs=1)

# Define the PPO model
model = PPO(
    "MlpPolicy",  # Multi-layer perceptron policy
    env,          # Custom environment
    verbose=1,    # Show training progress
    tensorboard_log="./ppo_ga_language_tensorboard/"  # For monitoring
)

# Train the model
print("Training the agent...")
model.learn(total_timesteps=10000)  # Adjust timesteps for longer training

# Save the trained model
model.save("ga_language_model")
print("Model saved as 'ga_language_model'.")

