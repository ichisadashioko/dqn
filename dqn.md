# DQN architecture

- Agent
    - Network (for updating q values)
    - Target network (for evaluating q values)
- Environment
- Memory
    - `State[]`
    - `NextState[]`
    - `Action[]`
    - `Reward[]`
    - `Terminal[]`

## Algorithm

```py
Memory.initialize()
Agent.initialize()
Environment.initialize()

State = Environment.resetEnv()

for i in range(numSteps):
    Epsilon = Agent.calcEpsilon()
    if Random < Epsilon:
        Action = Agent.Network.chooseAction(State)
    else:
        Action = Agent.randomAction()
    
    NextState, Reward, Terminal = Environment.step(Action)

    Memory.store(State, NextState, Action, Reward, Terminal)

    #if after the `Memory` is filled with enough data then train the `Agent` every steps.
    Batch = Memory.sample()
    FutureQ = Agent.TargetNet.predict(Batch.NextState)
    LabelQ = Bellman(Batch.NextState, Batch.Action, Batch.Reward, Batch.Terminal, FutureQ)
    Agent.fit(Agent.Network, Batch.State, LabelQ)
    #endif

    #if every `updateFrequency`
    Agent.TargetNet.Weights = Agent.Network.Weights
    #endif

    if Terminal:
        State = Environment.resetEnv()
    else:
        State = NextState
```