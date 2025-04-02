# Deep Q-Learning Implementation for Flappy Bird and CartPole

This repository implements Deep Q-Learning (DQL) to solve reinforcement learning environments like Flappy Bird and CartPole. Below is a detailed mathematical explanation of the algorithm and implementation.

## 1. Markov Decision Process (MDP) Formulation

The game is modeled as an MDP with:
- State space S (game state variables)
- Action space A (possible actions)
- Reward function r_t = R(s_t, a_t)
- Transition dynamics s_{t+1} ~ P(s_t, a_t)

## 2. Q-Learning Fundamentals

The Q-function Q^π(s,a) represents the expected return when taking action a in state s and following policy π thereafter:

$$
Q^\pi(s,a) = E[\sum_{k=0}^\infty \gamma^k r_{t+k} | s_t=s, a_t=a]
$$

where γ ∈ [0,1] is the discount factor.

## 3. Bellman Optimality Equation

The optimal Q-function satisfies:

$$
Q^*(s,a) = E[r + \gamma \max_{a'} Q^*(s',a') | s,a]
$$

## 4. Deep Q-Network (DQN) Approximation

We approximate Q*(s,a) using a neural network Q(s,a;θ) with parameters θ. The network architecture is:

$$
\text{Input} \rightarrow \text{FC}_1(\text{ReLU}) \rightarrow \text{FC}_2(\text{ReLU}) \rightarrow \text{Output}
$$

$$
\text{FC}_1: R^{|S|} \rightarrow R^{n_{\text{hidden}}}
$$

$$
\text{FC}_2: R^{n_{\text{hidden}}} \rightarrow R^{|A|}
$$

## 5. Loss Function

The network is trained to minimize the temporal difference error:

$$
L(\theta) = E_{(s,a,r,s') \sim D}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]
$$

where:
- D is the replay buffer
- θ^- are target network parameters
- γ is the discount factor

## 6. Experience Replay

The replay buffer D stores transitions (s_t, a_t, r_t, s_{t+1}) and samples mini-batches for training:

$$
(s_i, a_i, r_i, s_{i+1}) \sim \text{Uniform}(D)
$$

## 7. ε-Greedy Exploration

The action selection policy balances exploration and exploitation:

$$
\pi(s) = \begin{cases}
\text{Uniform}(A) & \text{with probability } \epsilon \\
\arg\max_a Q(s,a;\theta) & \text{otherwise}
\end{cases}
$$

where ε decays over time: ε ← max(ε_{min}, ε · ε_{decay})

## Usage

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

1. **View untrained behavior**:
```bash
python agent.py cartpole1 --view
# or
python agent.py flappybird1 --view
```

2. **Train the agent**:
```bash
python agent.py cartpole1 --train
# or
python agent.py flappybird1 --train
```

3. **Use trained model**:
```bash
python agent.py cartpole1
# or
python agent.py flappybird1
```

## Implementation Details

### Hyperparameters
Hyperparameters are configured in `hyperparameters.yml`:
- `replay_memory_size`: Size of experience replay buffer
- `mini_batch_size`: Training batch size
- `epsilon_init`: Initial exploration rate
- `epsilon_decay`: Exploration decay rate
- `epsilon_min`: Minimum exploration rate
- `network_sync_rate`: Target network update frequency
- `learning_rate_a`: Learning rate for Adam optimizer
- `discount_factor_g`: Reward discount factor
- `fc1_nodes`: Hidden layer size

### Network Architecture
- Input layer: State dimension
- Hidden layer: Configurable size (fc1_nodes)
- Output layer: Number of actions
- Activation: ReLU

### Training Features
- Experience replay for sample efficiency
- Target network for stability
- ε-greedy exploration
- Automatic model saving
- Training visualization
- MPS (Metal Performance Shaders) support for Apple Silicon

## Results

The agent learns to:
- CartPole: Balance the pole by moving the cart left/right
- FlappyBird: Navigate through pipes by learning optimal flap timing

Training progress is visualized in real-time showing:
- Mean rewards over time
- Exploration rate decay
