import numpy as np, torch, torch.nn as nn, torch.optim as optim, random
from collections import deque, namedtuple
Transition = namedtuple("Transition", "obs actions reward next_obs done")

class DuelingQ(nn.Module):
    def __init__(self, obs_dim, act_bins):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, act_bins)
    def forward(self, x):
        f = self.feature(x)
        V = self.V(f)
        A = self.A(f)
        return V + (A - A.mean(dim=-1, keepdim=True))

class MultiAgentDuelingDDQN:
    def __init__(self, obs_dim, act_bins, lr=1e-3, gamma=0.99, tau=0.005, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma, self.tau, self.act_bins = gamma, tau, act_bins
        self.nets = {k: DuelingQ(obs_dim, act_bins).to(self.device) for k in ["A_ctrl","B_ctrl","C_ctrl","EMS"]}
        self.tgt_nets = {k: DuelingQ(obs_dim, act_bins).to(self.device) for k in self.nets}
        for k in self.nets: self.tgt_nets[k].load_state_dict(self.nets[k].state_dict())
        self.optims = {k: optim.Adam(self.nets[k].parameters(), lr=lr) for k in self.nets}
        self.buffer = deque(maxlen=50000)

    def act(self, obs_dict, eps=0.1):
        acts = {}
        for k, net in self.nets.items():
            obs = torch.tensor(obs_dict[k], dtype=torch.float32, device=self.device).unsqueeze(0)
            if np.random.rand() < eps:
                acts[k] = np.random.randint(self.act_bins)
            else:
                with torch.no_grad():
                    q = net(obs)
                    acts[k] = int(q.argmax(dim=-1).item())
        return acts

    def push(self, obs, actions, reward, next_obs, done):
        self.buffer.append(Transition(obs, actions, reward, next_obs, done))

    def soft_update(self, src, dst):
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

    def learn(self, batch_size=64):
        if len(self.buffer) < batch_size: return {}
        batch = random.sample(self.buffer, batch_size)
        losses = {}
        for agent in self.nets:
            obs = torch.tensor([b.obs[agent] for b in batch], dtype=torch.float32, device=self.device)
            next_obs = torch.tensor([b.next_obs[agent] for b in batch], dtype=torch.float32, device=self.device)
            rewards = torch.tensor([b.reward[agent] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            dones = torch.tensor([b.done[agent] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            q = self.nets[agent](obs)
            a_idx = torch.tensor([b.actions[agent] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(-1)
            qa = q.gather(1, a_idx)
            with torch.no_grad():
                next_a = self.nets[agent](next_obs).argmax(dim=-1, keepdim=True)
                next_qa = self.tgt_nets[agent](next_obs).gather(1, next_a)
                y = rewards + self.gamma * (1 - dones) * next_qa
            loss = ((qa - y)**2).mean()
            self.optims[agent].zero_grad(); loss.backward(); self.optims[agent].step()
            self.soft_update(self.nets[agent], self.tgt_nets[agent])
            losses[agent] = float(loss.item())
        return losses
