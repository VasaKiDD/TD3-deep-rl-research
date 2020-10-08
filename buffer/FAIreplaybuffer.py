import numpy as np
import torch
from numba import jit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


@jit
def relativize(x):
    std = x.std()
    if float(std) == 0:
        return np.ones(x.shape)
    standard = (x - x.mean()) / std
    standard[standard > 0] = np.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = np.exp(standard[standard <= 0])
    return standard


@jit
def calculate_vrs(td_errors, epsilon, states, actions):
    compas_ix = np.random.permutation(np.size(td_errors))
    rewards = relativize(np.abs(td_errors) + epsilon)
    state_dist = np.array(states) - np.array(states)[compas_ix]
    state_dist = relativize(np.linalg.norm(state_dist, axis=1))
    action_dist = np.array(actions) - np.array(actions)[compas_ix]
    action_dist = relativize(np.linalg.norm(action_dist, axis=1))
    dist = relativize(state_dist + action_dist)
    vrs = dist * rewards
    vrs = vrs / np.sum(vrs)
    return vrs


class FractalReplayBuffer(object):
    def __init__(self, max_size, epsilon=1e-3):
        self.storage = []
        self.new_storage = []
        self.td_errors = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.vrs = None
        self.vrs2 = None
        self.max_size = max_size
        self.new_data_cpt = 0
        self.epsilon = epsilon

        # Expects tuples of (state, next_state, action, reward, done)

    def add(self, data):
        self.new_storage.append(data)

    def sample(self, batch_size=100):
        self.calculate_vrs()
        ind = np.random.choice(
            len(self.storage),
            size=batch_size if len(self.storage) > batch_size else len(self.storage),
            replace=False,
            p=self.vrs,
        )
        vr, x, y, u, r, d = [], [], [], [], [], []

        for i in ind:  # x, y, u, r, d state, next_state, action, reward, done
            R, D = self.storage[i]
            X = self.states[i]
            Y = self.next_states[i]
            U = self.actions[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            vr.append(self.vrs2[i])
            d.append(np.array(D, copy=False))

        return (
            ind,
            np.array(vr),
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1),
        )

    def update_samples(self, ind, td_errors, targets):
        for i in range(len(ind)):
            self.td_errors[ind[i]] = float(td_errors[i])
            self.rewards[ind[i]] = float(targets[i])

    def calculate_vrs(self):
        compas_ix = np.random.permutation(len(self.td_errors))
        rewards = relativize(np.abs(np.array(self.td_errors)) + self.epsilon)
        state_dist = np.array(self.states) - np.array(self.states)[compas_ix]
        state_dist = relativize(np.linalg.norm(state_dist, axis=1))
        action_dist = np.array(self.actions) - np.array(self.actions)[compas_ix]
        action_dist = relativize(np.linalg.norm(action_dist, axis=1))
        dist = relativize(state_dist + action_dist)
        self.vrs = dist * rewards
        self.vrs2 = dist * relativize(np.array(self.rewards))
        self.vrs = self.vrs / np.sum(self.vrs)

    def init_new_memory(self, critic, actor, discount):
        for (
            x,
            y,
            u,
            r,
            d,
        ) in self.new_storage:  # x, y, u, r, d state, next_state, action, reward, done
            state = np.array([x], copy=False)
            next_state = np.array([y], copy=False)
            action = np.array([u], copy=False)
            reward = np.array([r], copy=False)
            done = np.array([d], copy=False)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)
            with torch.no_grad():
                target_Q = critic(next_state, actor(next_state))
                if len(target_Q) == 2:
                    target_Q = torch.min(target_Q[0], target_Q[1])
                target_Q = reward + (done * discount * target_Q)
                current_Q = critic(state, action)
                if len(current_Q) == 2:
                    errors = target_Q * 2 - (current_Q[0] + current_Q[1])
                else:
                    errors = target_Q - current_Q
            self.storage.append((r, d))
            self.states.append(x)
            self.next_states.append(y)
            self.actions.append(u)
            self.td_errors.append(float(errors))
            self.rewards.append(target_Q)
        self.new_storage = []
        if len(self.storage) > self.max_size:
            self.calculate_vrs()
            indexes = np.random.choice(
                len(self.storage), size=int(self.max_size), replace=False, p=self.vrs
            )
            self.storage = np.array(self.storage)[indexes].tolist()
            self.td_errors = np.array(self.td_errors)[indexes].tolist()
            self.states = np.array(self.states)[indexes].tolist()
            self.next_states = np.array(self.next_states)[indexes].tolist()
            self.actions = np.array(self.actions)[indexes].tolist()
            self.rewards = np.array(self.rewards)[indexes].tolist()


class PrioritizeBuffer(object):
    def __init__(self, max_size, epsilon=1e-3):
        self.storage = []
        self.new_storage = []
        self.td_errors = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.vrs = None
        self.max_size = max_size
        self.new_data_cpt = 0
        self.epsilon = epsilon

        # Expects tuples of (state, next_state, action, reward, done)

    def add(self, data):
        self.new_storage.append(data)

    def sample(self, batch_size=100):
        self.vrs = calculate_vrs(
            np.array(self.td_errors),
            np.array(self.epsilon),
            np.array(self.states),
            np.array(self.actions),
        )
        ind = np.random.choice(
            len(self.storage),
            size=batch_size if len(self.storage) > batch_size else len(self.storage),
            replace=False,
            p=self.vrs,
        )
        # ind = np.argsort(self.vrs)[-batch_size:]
        x, y, u, r, d, w = [], [], [], [], [], []

        for i in ind:  # x, y, u, r, d state, next_state, action, reward, done
            R, D = self.storage[i]
            X = self.states[i]
            Y = self.next_states[i]
            U = self.actions[i]
            W = 1.0 / (self.vrs[i] * len(self.storage))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            w.append(np.array(W, copy=False))

        return (
            ind,
            np.array(w),
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1),
        )

    def update_samples(self, ind, td_errors):
        for i in range(len(ind)):
            self.td_errors[ind[i]] = float(td_errors[i])

    def init_new_memory(self, critic, actor, discount):
        for (
            x,
            y,
            u,
            r,
            d,
        ) in self.new_storage:  # x, y, u, r, d state, next_state, action, reward, done
            state = np.array([x], copy=False)
            next_state = np.array([y], copy=False)
            action = np.array([u], copy=False)
            reward = np.array([r], copy=False)
            done = np.array([d], copy=False)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)
            with torch.no_grad():
                target_Q = critic(next_state, actor(next_state))
                if len(target_Q) == 2:
                    target_Q = torch.min(target_Q[0], target_Q[1])
                target_Q = reward + (done * discount * target_Q)
                current_Q = critic(state, action)
                if len(current_Q) == 2:
                    errors = target_Q * 2 - (current_Q[0] + current_Q[1])
                else:
                    errors = target_Q - current_Q
            self.storage.append((r, d))
            self.states.append(x)
            self.next_states.append(y)
            self.actions.append(u)
            self.td_errors.append(float(errors.cpu()))
        self.new_storage = []
        if len(self.storage) > self.max_size:
            self.vrs = calculate_vrs(
                np.array(self.td_errors),
                np.array(self.epsilon),
                np.array(self.states),
                np.array(self.actions),
            )
            # indexes = np.argsort(self.vrs)[-self.max_size:]
            indexes = np.random.choice(
                len(self.storage), size=int(self.max_size), replace=False, p=self.vrs
            )
            self.storage = np.array(self.storage)[indexes].tolist()
            self.td_errors = np.array(self.td_errors)[indexes].tolist()
            self.states = np.array(self.states)[indexes].tolist()
            self.next_states = np.array(self.next_states)[indexes].tolist()
            self.actions = np.array(self.actions)[indexes].tolist()
