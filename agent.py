import flappy_bird_gymnasium
import gymnasium
import torch 
from dqn import DQN
from replaymemory import ReplayMemory
import itertools
import yaml
import random 
import torch.nn.functional as F 
import torch.nn as nn 


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
        self.discount_factor = hyperparameters['discount_factor']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        
        self.optimizer = None 
        
        
        
        
        
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode='human' if render else None)

        num_states = env.observation_space.shape[0] 
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []
        
        # Initialize the DQN agent
        policy_dqn = DQN(num_states, num_actions).to(device)
        
        memory = ReplayMemory(self.replay_memory_size) 
        state, _ = env.reset()
        
        if is_training: 
            
            target_dqn =  DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict()) 
            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)
            
        
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
                            with torch.no_grad():
                                # Get the action from the policy network
                                action = policy_dqn(state_tensor).argmax().item()

                        # Processing:
                        new_state, reward, terminated, _, info = env.step(action)
                        
                        reward = torch.tensor(reward, dtype=torch.float32).to(device) 
                        new_state_tensor = torch.tensor(new_state, dtype=torch.float32).to(device)
                        
                        episode_reward += reward.item()
                        # Store the experience in memory
                        
                        if is_training:
                            memory.append((state, action, reward, new_state, terminated))
                            
                            step_count += 1

                                
                            
                        state = new_state
        
        rewards_per_episode.append(episode_reward)
        epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        epsilon_history.append(epsilon)
        
                                    
        if len(memory) > self.mini_batch_size: 
            mini_batch = memory.sample(self.mini_batch_size) 
            self.optimize(mini_batch, policy_dqn, target_dqn) 
            
            if step_count > self.network_sync_rate: 
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0  
                
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
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
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
                    

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze() 
        loss = F.mse_loss(current_q, torch.tensor(target_q)) 
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()        

        
# Example usage
if __name__ == "__main__":
    agent = Agent('cartpolev1')
    agent.run(render=True)
    


# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)