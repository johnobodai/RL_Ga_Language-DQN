# train.py
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from ga_language_env import GaLanguageEnv  # Import custom environment

# Define the neural network for Q-value approximation
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))  # Q-values for each action
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Set up environment
env = GaLanguageEnv()

# Define state and action size based on environment
state_size = env.observation_space.n  # 25 grid cells (5x5)
action_size = env.action_space.n  # 4 possible actions (up, down, left, right)

# Build the neural network model
model = build_model(state_size, action_size)

# Set up the memory
memory = SequentialMemory(limit=50000, window_length=1)

# Define the exploration policy (epsilon-greedy)
policy = EpsGreedyQPolicy()

# Create the DQN agent
dqn_agent = DQNAgent(model=model, 
                     nb_actions=action_size, 
                     memory=memory, 
                     policy=policy, 
                     nb_steps_warmup=10,  # warmup steps before training starts
                     target_model_update=1e-2,  # target model update frequency
                     train_interval=4,  # train every 4 steps
                     delta_clip=1.0)

# Compile the agent
dqn_agent.compile(Adam(learning_rate=0.001), metrics=['mae'])

# Train the agent
print("Training the agent...")
dqn_agent.fit(env, nb_steps=10000, visualize=False, verbose=2)

# Save the trained model weights to an H5 file
dqn_agent.save_weights("dqn_ga_language_weights.h5", overwrite=True)
print("Model weights saved as 'dqn_ga_language_weights.h5'.")

# To load the weights back, you can use:
# dqn_agent.load_weights("dqn_ga_language_weights.h5")

# Test the trained agent (after training)
# dqn_agent.test(env, nb_episodes=5, visualize=True)

