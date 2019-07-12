## Algorithm 1: deep Q-learning with experience replay

- Initialize replay memory $D$ to capacity $N$
- Initialize action-value function $Q$ with random weights $\theta$
- Initialize target action-value function $\hat{Q}$ with weights $\theta^- = \theta$
- **For** episode = 1, $M$ **do**
    - Initalize sequence $s_1 = \{x_1\}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
    - **For** $t = 1,T$ **do**
        - With probability $\epsilon$ select a random action $a_t$
        - otherwise select $a_t = argmax_a Q(\phi(s_t), a; \theta)$
        - Excute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
        - Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        - Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$
        - Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$
        - Set $y_j = r_j$ if episode terminates at step $j+1$
        - otherwise set $y_j = r_j + \gamma \max_{a^{'}} \hat{Q}(\phi_{j+1}, a^{'};\theta^{-})$
        - Perform a gradient descent step on $(y_i - Q(\phi_j, a_j; \theta))^2$ with respect to the network paramethers $\theta$
        - Every $C$ steps reset $\hat{Q} = Q$
    - **End For**
- **End For**

## List of hyperparameters and their values

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| minibatch size | 32 | Number of training cases over which each stochastic gradient descent (SGD) update is computed |
| replay memory size | 1_000_000 | SGD updates are sampled from this number of most recent frames. |
| agent history length | 4 | The number of most recent frames experienced by agent that are given as input to the Q network. |
| target network update frequency | 10_000 | The frequency (measured in the number of parameter updates) with which the target network is updated (this corresponds to the parameter *C* from [Algorithm 1]()). |
| discount_factor | 0.99 | Discount factor gamma used in the Q-learning update. |
| action repeat | 4 | Repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4th input frame. |
| update frequency | 4 | The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates. |
| learning rate | 0.00025 | The learning rate used by RMSProp. |
| gradient momentum | 0.95 | Gradient momentum used by RMSProp. |
| squared gradient momentum | 0.95 | Squared gradient (denominator) momentum used by RMSProp. |
| min squared gradient | 0.01 | Constant added to the squared gradient in the denominator of the RMSProp update. |
| inital exploration | 1 | Initial value of $\epsilon$ in $\epsilon$-greedy exploration. |
| final exploration | 0.1 | Final value of $\epsilon$ in $\epsilon$-greedy exploration. |
| final exploration frame | 1_000_000 | The number of frames over which the initial value of $\epsilon$ is linearly annealed to its final value. |
| replay start size | 50_000 | A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory. |
| no-op max | 30 | Maximum number of "do nothing" actions to be performed by the agent at the start of an episode. |