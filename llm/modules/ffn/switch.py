import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.d_model = d_model

        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, use_aux_loss=False):
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = orig_shape
            x_flat = x.view(B * T, D)
        else:
            x_flat = x

        gate_scores = self.w_gate(x)
        gate_probs = torch.softmax(gate_scores, dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        # Get the top-1 expert indices
        top1_expert_indices = torch.argmax(gate_probs, dim=-1)

        # Mask to enforce capacity constraints
        mask = F.one_hot(top1_expert_indices, num_classes=self.num_experts).to(gate_probs.dtype)
        masked_gate_scores = gate_scores * mask

        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity
        aux_loss = None
        if use_aux_loss:
            # Load-balancing auxiliary loss (Switch-style): encourage uniform importance and load
            # importance: sum of probabilities per expert; load: number of tokens assigned per expert
            importance = gate_probs.sum(dim=0)  # [E]
            load = mask.sum(dim=0)  # [E]
            # Mean-squared difference between normalized vectors
            imp_norm = importance / (importance.sum() + self.epsilon)
            load_norm = load / (load.sum() + self.epsilon)
            aux_loss = ((imp_norm - load_norm) ** 2).mean()

        return gate_scores, aux_loss


class SwitchMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_experts: int = 4,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff

        self.router = Router(d_model, num_experts, capacity_factor, epsilon)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, d_ff), act_fn(), nn.Linear(d_ff, d_model))
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor, use_aux_loss=False):
        gate_scores, aux_loss = self.router(x, use_aux_loss)

        # Create a tensor to hold the expert outputs
        expert_outputs = torch.zeros_like(x)

        # For each expert, compute its output and weight it by the gate scores
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs += gate_scores[:, i].unsqueeze(1) * expert_output

        return expert_outputs, aux_loss
