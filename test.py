from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
import torch as T
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
# import warnings
# warnings.filterwarnings('warning')
# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following two lines.


train = pd.read_csv('trade_data_tesla_5min.csv')
train = train.set_index(train.columns[0])
train.index.names = ['']
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 500,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs) 
#env = StockTradingEnv(df = train, **env_kwargs) 
env, _ = e_train_gym.get_sb_env()
# env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[state_space], tau=0.005, env=env,
              batch_size=32,  layer1_size=400, layer2_size=300, layer3_size=200, layer4_size=100, layer5_size=50, n_actions=stock_dimension)

agent.load_models()
np.random.seed(0)

score_history = []

obs = env.reset()
done = False
rewards = []
portfolio_value = []
while not done:
    portfolio_value.append(obs[0][0] + obs[0][1] * obs[0][2])
    agent.target_actor.eval()
    act = agent.target_actor.forward(T.tensor([obs],dtype=T.float).to(device)).detach().cpu().numpy()
    new_state, reward, done, info = env.step(act[0])
    rewards.append(reward[0]*1e4)
    
    obs = new_state
    #env.render()

print(np.array(rewards).sum() * 1e4)
#if i % 25 == 0:
#    agent.save_models()

plotLearning(portfolio_value, 'portfolio.png',xlabel = 'time', ylabel = 'portfolio value',window=500)
# plotLearning(rewards, 'rewards_test.png', xlabel = 'time', ylabel = 'rewards', window=100)

