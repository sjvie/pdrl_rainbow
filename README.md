# Improving Rainbow: Improving Improvements in Deep Reinforcement Learning

This is an implementation of the Rainbow reinforcement learning agent presented by Hessel et al.
The implementation uses parallel asynchronous environments and has some extensions to the original Rainbow agent:
- Different neural network architectures, namely the original DQN architecture, the Impala CNN, and D2RL
- Different exploration strategies, namely epsilon-greedy, noisy nets, softmax exploration, and random network distillation

To train the agent with the original Rainbow settings:
```
python main.py env-name="ALE/Breakout-v5" log-wandb=False
```
