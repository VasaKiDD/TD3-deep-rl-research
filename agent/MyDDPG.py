import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Actions(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actions, self).__init__()
        self.actions = nn.Parameter(torch.zeros(1, action_dim))
        self.max_action = max_action

    def forward(self):
        return self.max_action * F.tanh(self.actions)

    def reset(self):
        self.actions.data = self.actions.data * 0.0


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):

        self.max_action = max_action
        self.action_dim = action_dim
        self.actions = Actions(self.action_dim, self.max_action)
        # self.actions_optimizer = torch.optim.Adam(self.actions.parameters())
        self.actions_optimizer = torch.optim.SGD(self.actions.parameters(), lr=0.0, momentum=0.0)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def reset(self):
        self.actions.reset()
        return self.actions().cpu().data.numpy().flatten()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state_value = -self.critic(state, self.actions())
        self.actions_optimizer.zero_grad()
        state_value.backward()
        self.actions_optimizer.step()
        self.critic.zero_grad()

        # print(self.actions().cpu().data.numpy().flatten())

        return self.actions().cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, v, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            new_action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, new_action)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))
