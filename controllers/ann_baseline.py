# controllers/ann_baseline.py
import torch
import torch.nn as nn

class ANNBaseline(nn.Module):
    """
    Single-agent ANN baseline used for comparison with MARL.
    It predicts discrete action logits for each agent independently.
    """
    def __init__(self, obs_dim, action_bins):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_bins = action_bins

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Separate heads for each agent
        self.heads = nn.ModuleDict({
            "A_ctrl": nn.Linear(64, action_bins),
            "B_ctrl": nn.Linear(64, action_bins),
            "C_ctrl": nn.Linear(64, action_bins),
            "EMS": nn.Linear(64, action_bins)
        })

    def forward(self, obs_tensor):
        """
        obs_tensor: (batch, obs_dim)
        returns dict of logits for each agent
        """
        z = self.encoder(obs_tensor)
        return {agent: head(z) for agent, head in self.heads.items()}
