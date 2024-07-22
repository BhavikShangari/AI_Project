# Stock Trading using Reinforcement Learning with DDPG Algorithm

## Introduction
This project explores using Deep Deterministic Policy Gradient (DDPG), a powerful Deep Reinforcement Learning (DRL) technique, to generate profitable trading signals for stocks. Leveraging the FinRL library, renowned for its comprehensive Stock Trading environment, we aim to develop an intelligent trading agent that maximizes portfolio returns while managing risk effectively.

## FinRL
Developed by AI4Finance Foundation, FinRL is a framework providing environments for stock trading, portfolio optimization, and cryptocurrency trading. It offers various Deep RL agents like PPO, A3C, DDPG, and SAC, along with tutorials for easy reproduction and comparison with existing schemes. FinRL configures virtual environments with market data, trains trading agents with neural networks, and analyzes extensive backtesting via trading performance.

## Deep Deterministic Policy Gradient (DDPG)
DDPG is a model-free Deep Reinforcement Learning algorithm for continuous domain actions. It combines concepts like Replay Memory Buffer, Actor Critic Network, and Target Networks, utilizing 4 Neural Networks: Q network, deterministic policy network, target Q network, and target policy network.


## Code Structure
```bash
├── train.py                         For training DDPG model
├── ddpg_torch.py                    Contains implementation of DDPG in PyTorch
├── test.py                          For testing the trained model
├── util.py                          For plotting utility
├── portfolio.png                    Portfolio value during inference
├── tesla_train_price.png            Chart of Tesla stock prices for the training period
├── tesla_trade_price.png            Chart of Tesla stock prices for the inference period
└── tmp
    └── ddpg
        ├── actor_model_weights.txt            Actor model weights
        ├── target_actor_model_weights.txt     Target actor model weights
        ├── critic_model_weights.txt           Critic model weights
        └── target_critic_model_weights.txt    Target critic model weights
```



## Steps to Reproduce
1. Install dependencies by running:
`pip install -r requirements.txt`

2. For training the Agent
`python3 train.py`

3. For testing the Agent capability
`python3 test.py`