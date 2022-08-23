# Improving Rainbow: Improving Improvements in Deep Reinforcement Learning

__by Alexander Ludwig & Sören Viegener__

This is an implementation of the Rainbow reinforcement learning agent presented by Hessel et al.
The implementation uses parallel asynchronous environments and has some extensions to the original Rainbow agent:
- Different neural network architectures, namely the original DQN architecture, the Impala CNN, and D2RL
- Different exploration strategies, namely epsilon-greedy, noisy nets, softmax exploration, and random network distillation

## Running the Agent

To train the agent with the original Rainbow settings:
```
python main.py env-name="ALE/Breakout-v5" log-wandb=False
```



_This was created as part of the Project Deep Reinforcement Learning at Ulm University_
