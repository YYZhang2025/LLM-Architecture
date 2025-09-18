import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor, use_aux_loss=False):
        gate_scores = self.w_gate(x)
        gate_probs = torch.softmax(gate_scores, dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        # Get the top-1 expert indices
        top1_expert_indices = torch.argmax(gate_probs, dim=-1)

        # Mask to enforce capacity constraints
        mask = torch.zeros_like(gate_scores).scatter_(1, top1_expert_indices.unsqueeze(1), 1)
        masked_gate_scores = gate_scores * mask

        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


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
