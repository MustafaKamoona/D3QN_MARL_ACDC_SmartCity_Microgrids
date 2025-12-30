import torch, torch.nn as nn

class ANNBaseline(nn.Module):
    """
    Traditional ANN baseline: state -> discrete actions for A,B,C,EMS.
    Train supervised on heuristic/OPF-derived labels.
    """
    def __init__(self, obs_dim, action_bins):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.head_A   = nn.Linear(128, action_bins)
        self.head_B   = nn.Linear(128, action_bins)
        self.head_C   = nn.Linear(128, action_bins)
        self.head_EMS = nn.Linear(128, action_bins)

    def forward(self, obs):
        x = self.shared(obs)
        return {
            "A_ctrl": self.head_A(x),
            "B_ctrl": self.head_B(x),
            "C_ctrl": self.head_C(x),
            "EMS": self.head_EMS(x)
        }
    @torch.no_grad()
    def act(self, obs):
        logits = self.forward(obs)
        return {k: int(v.argmax(dim=-1).item()) for k, v in logits.items()}
