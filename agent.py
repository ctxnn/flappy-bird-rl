import flappy_bird_gymnasium
import gymnasium
import torch 
from dqn import DQN

class Agent:
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode='human' if render else None)

        num_states = env.observation_space.shape[0] 
        num_actions = env.action_space.n
        
        # Initialize the DQN agent
        policty_dqn = DQN(num_states, num_actions) 
        
        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            
            # Checking if the player is still alive
            if terminated:
                break


        env.close()
        
    

# Example usage
if __name__ == "__main__":
    agent = Agent()
    agent.run(render=True)
    


# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)