import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class DQN(nn.Module):
    def __init__(self, num_states, num_actions, fc1_nodes=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_states, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

if __name__ == "__main__":
    dqn = DQN(12, 2)
    state = torch.randn(1,12) 
    print(state)
    output = dqn(state)
    print(output)
    print(dqn) 
