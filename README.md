
---

# Ga Language Environment and PPO Model for Grid Navigation

This project demonstrates the use of **Proximal Policy Optimization (PPO)** with a custom environment for navigating a grid-based map while learning greetings in the **Ga language**. The environment simulates an agent moving on a grid and interacting with locations that correspond to common greetings in Ga. The goal of the agent is to reach the target location (goal) while learning and maximizing its reward.

## Features

- **Custom Ga Language Environment**: A grid-based environment where each cell contains a greeting in the Ga language.
- **Reinforcement Learning with PPO**: Using the PPO algorithm for training an agent to learn optimal actions in the environment.
- **Real-time Visualization**: Interactive plot showing the agent's movement on the grid, rewards, and steps.

## Installation

### Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.6+
- pip
- `matplotlib` for visualization
- `stable-baselines3` for reinforcement learning
- `gym` for creating custom environments

### Setup

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/ga-language-ppo.git
```

2. Navigate into the project directory:

```bash
cd ga-language-ppo
```

3. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```


## Usage

1. **Training the Model**:
   To train the PPO model on the Ga Language environment, use the following script:

   ```bash
   python train.py
   ```

   This will start training and save the model in the current directory as `ga_language_model.zip`.

2. **Testing the Model**:
   After training, you can test the model by running:

   ```bash
   python play.py
   ```

   This will load the trained model and run a test where the agent tries to navigate the grid and maximize its reward while learning greetings in Ga.

## Structure

```
ga-language-ppo/
│
├── train.py             # Script to train the PPO model on the custom environment
├── play.py              # Script to play and visualize the trained model
├── ga_language_env.py   # Custom environment for the Ga Language task
├── requirements.txt     # List of Python package dependencies
└── README.md            # This README file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Stable-Baselines3**: For providing the PPO implementation.
- **Gym**: For creating the environment framework.
- **Matplotlib**: For visualization of the agent’s actions and the environment.

---

