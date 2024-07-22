# The code defines an Agent class for implementing a Deep Deterministic Policy Gradient (DDPG)
# algorithm for reinforcement learning.
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# The `OUActionNoise` class implements Ornstein-Uhlenbeck action noise for reinforcement learning
# algorithms.
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

# The `ReplayBuffer` class implements a memory buffer for storing and sampling transitions for
# reinforcement learning algorithms.
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        """
        This function stores a transition tuple consisting of the current state, action taken, reward
        received, next state, and whether the episode is done in a memory buffer for reinforcement
        learning.
        
        :param state: State refers to the current state of the environment or system in a reinforcement
        learning context. It represents the information or observations that the agent receives from the
        environment at a given time step. This information is typically used by the agent to make
        decisions on what action to take next
        :param action: The `action` parameter in the `store_transition` function represents the action
        taken by an agent in a reinforcement learning environment at a particular state. It is stored in
        the memory buffer along with other information such as the current state, next state, reward
        received, and whether the episode is done or not
        :param reward: The `reward` parameter in the `store_transition` function represents the reward
        received after taking a specific action in a given state. It is a scalar value that indicates
        the immediate feedback or reinforcement received by the agent for its action in the environment.
        Rewards are used to guide the learning process in reinforcement learning
        :param state_: In the provided code snippet, `state_` is a parameter representing the next state
        in a transition. It is used to store the next state information in a memory buffer for
        reinforcement learning algorithms
        :param done: The `done` parameter in the `store_transition` function typically represents
        whether the current episode has ended or not. It is a boolean value where `True` indicates that
        the episode has ended (i.e., the agent has reached a terminal state) and `False` indicates that
        the episode is still
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        The function `sample_buffer` randomly selects a batch of experiences from the memory buffer and
        returns the corresponding states, actions, rewards, new states, and terminal flags.
        
        :param batch_size: Batch size is the number of samples that will be randomly chosen from the
        memory buffer for training the neural network. It determines how many transitions (states,
        actions, rewards, new states, and terminal flags) will be included in each training batch
        :return: The `sample_buffer` function returns a batch of states, actions, rewards, new states,
        and terminal flags from the replay memory buffer.
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# The `CriticNetwork` class defines a neural network model for a critic in a Deep Deterministic Policy
# Gradient (DDPG) algorithm, with multiple fully connected layers and layer normalization.
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,fc3_dims, fc4_dims, fc5_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg/'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(self.fc4_dims)
        
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        f5 = 1./np.sqrt(self.fc5.weight.data.size()[0])
        T.nn.init.uniform_(self.fc5.weight.data, -f5, f5)
        T.nn.init.uniform_(self.fc5.bias.data, -f5, f5)
        self.bn5 = nn.LayerNorm(self.fc5_dims)
        
        
        self.action_value = nn.Linear(self.n_actions, self.fc5_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc5_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc4(state_value)
        state_value = self.bn4(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc5(state_value)
        state_value = self.bn5(state_value)
        state_value = F.relu(state_value)

        action_value = F.relu(self.action_value(action))
        
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file+f'critic_{self.fc1_dims}_{self.fc2_dims}_{self.fc3_dims}_{self.fc4_dims}_{self.fc5_dims}')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file+f'critic_{self.fc1_dims}_{self.fc2_dims}_{self.fc3_dims}_{self.fc4_dims}_{self.fc5_dims}'))

# The ActorNetwork class is a subclass of nn.Module in Python.
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg/'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.n_actions = n_actions
        
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(self.fc4_dims)
        
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        f5 = 1./np.sqrt(self.fc5.weight.data.size()[0])
        T.nn.init.uniform_(self.fc5.weight.data, -f5, f5)
        T.nn.init.uniform_(self.fc5.bias.data, -f5, f5)
        self.bn5 = nn.LayerNorm(self.fc5_dims)
        
        #f3 = 0.004
        f6 = 0.003
        self.mu = nn.Linear(self.fc5_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f6, f6)
        T.nn.init.uniform_(self.mu.bias.data, -f6, f6)
        #self.mu.weight.data.uniform_(-f3, f3)
        #self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        """
        This function defines a forward pass through a neural network with multiple fully connected
        layers, batch normalization, and activation functions.
        
        :param state: It looks like you are implementing a neural network forward pass function in
        PyTorch. The `state` parameter is likely the input data that is passed through the network
        layers to generate an output. In this case, the `state` is being processed through fully
        connected layers (`fc1`, `fc
        :return: The forward method is returning the output of the neural network model after passing
        the input state through multiple fully connected layers (fc1, fc2, fc3, fc4, fc5) followed by
        batch normalization layers (bn1, bn2, bn3, bn4, bn5) and ReLU activation functions. Finally, the
        output is passed through a tanh activation function applied to the
        """
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        """
        This function saves a checkpoint for the current state of the program.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file+f'actor_{self.fc1_dims}_{self.fc2_dims}_{self.fc3_dims}_{self.fc4_dims}_{self.fc5_dims}')

    def load_checkpoint(self):
        """
        The function `load_checkpoint` loads a checkpoint file using the `load_state_dict` method in
        Python.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file+f'actor_{self.fc1_dims}_{self.fc2_dims}_{self.fc3_dims}_{self.fc4_dims}_{self.fc5_dims}'))

# This is a Python class named Agent.
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=3, max_size=1000000, layer1_size=400,
                 layer2_size=300,layer3_size = 200,layer4_size = 100,layer5_size = 50, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, layer4_size, layer5_size, n_actions=n_actions,
                                  name='Actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, layer5_size, n_actions=n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, layer4_size, layer5_size, n_actions=n_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, layer5_size, n_actions=n_actions,
                                           name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        This function chooses an action based on the given observation using an actor neural network
        with added noise.
        
        :param observation: Observation typically refers to the input data or state that the agent
        receives from the environment in a reinforcement learning setting. It could be a set of values
        representing the current state of the environment, such as sensor readings, pixel values from a
        camera, or any other relevant information
        :return: The function `choose_action` returns the action generated by the actor neural network
        with added noise. The action is converted to a numpy array before being returned.
        """
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        """
        The `remember` function stores a transition tuple consisting of the current state, action taken,
        reward received, new state reached, and whether the episode is done in a memory buffer.
        
        :param state: State refers to the current state of the environment or system in a reinforcement
        learning context. It represents the information or observations that the agent uses to make
        decisions
        :param action: The `action` parameter typically refers to the action taken by an agent in a
        reinforcement learning environment. It represents the decision made by the agent at a particular
        state in order to interact with the environment. Examples of actions could include moving left
        or right, jumping, attacking, or any other possible action within
        :param reward: The `reward` parameter in the `remember` function represents the immediate reward
        received after taking a specific action in a particular state during the reinforcement learning
        process. It is a numerical value that indicates the feedback or outcome of the action taken by
        the agent in the environment
        :param new_state: The `new_state` parameter typically refers to the new state that the agent
        transitions to after taking a specific action in the environment. It represents the state of the
        environment after the agent has performed an action
        :param done: The `done` parameter typically represents whether the current episode or task has
        been completed. It is a boolean value that is set to `True` when the episode is finished or the
        agent reaches a terminal state, and `False` otherwise. This information is important for
        reinforcement learning algorithms to understand when to
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
        The function implements a learning algorithm for an actor-critic reinforcement learning agent.
        :return: The `learn` method is not explicitly returning any value. It is a method that performs
        a series of operations related to training a reinforcement learning agent, such as updating the
        critic and actor networks based on the sampled experiences from the replay buffer.
        """
        if self.memory.mem_cntr < 3*self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        The function `update_network_parameters` updates the target actor and critic network parameters
        using a soft update approach with a specified tau value.
        
        :param tau: Tau is a hyperparameter used in the soft update of target network parameters in
        reinforcement learning algorithms like Deep Deterministic Policy Gradient (DDPG). It controls
        the interpolation between the current network parameters and the target network parameters
        """
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        
    def save_models(self):
        """
        The `save_models` function saves the checkpoints of actor, target actor, critic, and target
        critic models.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """
        The `load_models` function loads checkpoints for actor and critic models along with their target
        counterparts.
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        """
        This function compares the parameters of actor and critic models with their original versions.
        """
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
    
    def predict(self, observation, deterministic = True):
        """
        This Python function predicts an action based on an observation using a target actor model.
        
        :param observation: Observation typically refers to the input data or state that is provided to
        a machine learning model for prediction or decision-making. In this context, the `observation`
        parameter is likely the input data that will be used by the `predict` method to generate an
        action
        :param deterministic: The `deterministic` parameter in the `predict` method is a boolean flag
        that indicates whether the prediction should be deterministic or stochastic, defaults to True
        (optional)
        :return: The predict function returns the action predicted by the target actor neural network
        for the given observation. In this case, the action is printed and returned as output along with
        None.
        """
        self.target_actor.eval()
        action = self.target_actor.forward(T.tensor([observation], device = self.device))
        print(action)
        return action, None
