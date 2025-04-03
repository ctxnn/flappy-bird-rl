import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from replaymemory import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os
import torch.nn.functional as F
import wandb

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

# Device selection with proper MPS support
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# Deep Q-Learning Agent
class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        # Initialize lists for tracking
        self.rewards_per_episode = []
        self.epsilon_history = []

        # Initialize wandb
        self.wandb_run = None

    def init_wandb(self):
        """Initialize Weights & Biases tracking"""
        self.wandb_run = wandb.init(
            project="flappy-bird-rl",
            config={
                "env_id": self.env_id,
                "learning_rate": self.learning_rate_a,
                "discount_factor": self.discount_factor_g,
                "network_sync_rate": self.network_sync_rate,
                "replay_memory_size": self.replay_memory_size,
                "mini_batch_size": self.mini_batch_size,
                "epsilon_init": self.epsilon_init,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "fc1_nodes": self.fc1_nodes,
            },
            name=f"{self.hyperparameter_set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def run(self, is_training=True, render=False):
        if is_training:
            # Initialize wandb at the start of training
            self.init_wandb()
            
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            state_dict = torch.load(self.MODEL_FILE, map_location=device)
            policy_dqn.load_state_dict(state_dict)
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():
            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float32, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode
            episode_loss = None     # Track loss for the episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            self.rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph()
                    last_graph_update_time = current_time

                # Train if enough samples
                if is_training and len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    episode_loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                    
                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    # Track epsilon history
                    self.epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            # Log metrics to wandb after each episode
            if is_training and self.wandb_run is not None:
                metrics = {
                    "episode": episode,
                    "reward": episode_reward,
                    "epsilon": epsilon,
                    "avg_reward": np.mean(self.rewards_per_episode[-100:]) if self.rewards_per_episode else 0,
                }
                # Only add loss if we had training steps this episode
                if episode_loss is not None:
                    metrics["loss"] = episode_loss
                    
                self.wandb_run.log(metrics)

        if is_training and self.wandb_run is not None:
            # Close wandb run when training ends
            self.wandb_run.finish()

    def save_graph(self):
        # Save plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(self.rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(self.rewards_per_episode[max(0, x-99):(x+1)])
        ax1.plot(mean_rewards)
        ax1.set_ylabel('Mean Rewards')

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax2.plot(self.epsilon_history)
        ax2.set_ylabel('Epsilon')

        plt.tight_layout()

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

        # Log additional training metrics to wandb
        if self.wandb_run is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), float('inf'))
            self.wandb_run.log({
                "gradient_norm": grad_norm,
                "target_q_mean": target_q.mean().item(),
                "target_q_max": target_q.max().item(),
                "q_value_diff": (current_q - target_q).abs().mean().item(),  # Q-value prediction error
                "termination_rate": terminations.float().mean().item(),  # How often episodes end
                "reward_mean": rewards.mean().item(),  # Average reward in batch
                "target_q_max": target_q.max().item(),
            })

        return loss.item()

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set name from hyperparameters.yml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--view', help='View mode - shows untrained agent behavior', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        print("Starting training mode...")
        dql.run(is_training=True)
    elif args.view:
        # Run with random actions (epsilon=1) and rendering
        print("Viewing untrained agent behavior...")
        env = gym.make(dql.env_id, render_mode='human', **dql.env_make_params)
        for episode in range(3):  # Show 3 episodes
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = env.action_space.sample()  # Random actions
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {episode + 1} reward: {total_reward}")
        env.close()
        
        # Ask if user wants to see trained model
        response = input("\nWould you like to see the trained model? (y/n): ")
        if response.lower() == 'y':
            print("\nLoading trained model for evaluation...")
            dql.run(is_training=False, render=True)
    else:
        print("Loading trained model for evaluation...")
        dql.run(is_training=False, render=True)

# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)