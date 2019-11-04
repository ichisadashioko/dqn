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

## Saving and loading progress

- Save model weights
- Save `Memory`
    - Save `Memory`'s data
    - Save `Environment`'s state
- Save number of trained steps
- Save logging information
- JSON format

    ```json
    {
        "agent": {
            "network": "networkFilePath",
            "targetNet": "targetNetFilePath",
            "trainedSteps": 0,
            "updateFrequency": 1000,
            "epsilonStart": 1.0,
            "epsilonEnd": 0.1,
            "epsilonEndStep": 1000000,
            "learningRate": 0.00025,
            "discount": 0.99,
        },
        "memory": {
            "data": {
                "state": "stateDataFilePath",
                "nextState": "nextStateDataFilePath",
                "reward": "rewardDataFilePath",
                "action": "actionDataFilePath",
                "terminal": "terminalDataFilePath",
            },
        },
        "env": {
            "state": "envStateDataFilePath",
        },
    }
    ```