import flappy_bird_gymnasium
import gymnasium
import torch 
from dqn import DQN
from replaymemory import ReplayMemory
import itertools
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Agent:
    
    def __init__(self, hyperparameter_set):
        with open(hyperparameter_set, 'r') as f:
            self.hyperparameters = yaml.safe_load(f)
            hyperparameters = self.hyperparameters['hyperparameters']
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['batch_size']
        self.epsilon = hyperparameters['epsilon']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        
        
        
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode='human' if render else None)

        num_states = env.observation_space.shape[0] 
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        
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
                        action = env.action_space.sample()

                        # Processing:
                        new_state, reward, terminated, _, info = env.step(action)
                        
                        episode_reward += reward
                        # Store the experience in memory
                        
                        if is_training:
                            memory.appned((state, action, reward, new_state, terminated))
                            
                        state = new_state
                
        rewards_per_episode.append(episode_reward)

        
# Example usage
if __name__ == "__main__":
    agent = Agent()
    agent.run(render=True)
    


# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)