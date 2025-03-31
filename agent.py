import flappy_bird_gymnasium
import gymnasium
import torch 
from dqn import DQN
from replaymemory import ReplayMemory
import itertools
import yaml
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Agent:
    
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as f:
            all_hyperparameters_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_start = hyperparameters['epsilon_start']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_end = hyperparameters['epsilon_end']
        
        
        
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode='human' if render else None)

        num_states = env.observation_space.shape[0] 
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []
        
        # Initialize the DQN agent
        policty_dqn = DQN(num_states, num_actions).to(device)
        
        memory = ReplayMemory(self.replay_memory_size) 
        state, _ = env.reset()
        
        
        for i in itertools.count():
                    terminated = False
                    state, _ = env.reset()
                    episode_reward = 0.0
                    
                    while not terminated:
                    # (feed the observation to your agent here)
                    
                        if is_training and random.random() < self.epsilon_end:
                            # Exploration
                            action = env.action_space.sample()
                        else:
                            # Exploitation
                            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                            action = policty_dqn(state_tensor).argmax().item()
                            
                        action = env.action_space.sample()

                        # Processing:
                        new_state, reward, terminated, _, info = env.step(action)
                        
                        reward = torch.tensor(reward, dtype=torch.float32).to(device) 
                        new_state_tensor = torch.tensor(new_state, dtype=torch.float32).to(device)
                        
                        episode_reward += reward.item()
                        # Store the experience in memory
                        
                        if is_training:
                            memory.append((state, action, reward, new_state, terminated))
                            
                        state = new_state
        
        rewards_per_episode.append(episode_reward)
        epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        epsilon_history.append(epsilon)

        
# Example usage
if __name__ == "__main__":
    agent = Agent('cartpolev1')
    agent.run(render=True)
    


# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)