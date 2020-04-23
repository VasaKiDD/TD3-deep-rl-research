import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"


class CuriosityBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.new_storage = []
        self.curiosity_targets = []

        # Expects tuples of (state, next_state, action, reward, done)

    def add(self, data):
        self.new_storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.choice(
            len(self.storage),
            size=batch_size if len(self.storage) > batch_size else len(self.storage),
            replace=False,
            p=self.vrs,
        )
        x, y, u, r, d, c = [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            cur = self.curiosity_targets[ind]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            c.append(cur)

        return (
            ind,
            np.array(c),
            np.array(x),
            np.array(y),
            np.array(u),
            np.array(r).reshape(-1, 1),
            np.array(d).reshape(-1, 1),
        )

    def update_samples(self, ind, curiosities):
        for i in range(len(ind)):
            c = float(curiosities[i])
            self.curosity_targets[ind[i]] = np.random.binomial(1, c)

    def init_new_memory(self, curiosity_net):
        for (
            x,
            y,
            u,
            r,
            d,
        ) in self.new_storage:  # x, y, u, r, d state, next_state, action, reward, done
            next_state = np.array([y], copy=False)
            next_state = torch.FloatTensor(next_state).to(device)
            with torch.no_grad():
                c = float(curiosity_net(next_state))
            self.curiosity_targets.append(np.random.binomial(1, c))
            self.storage.append((x, y, u, r, d))
        self.new_storage = []
