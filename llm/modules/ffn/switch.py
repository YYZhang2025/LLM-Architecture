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

        # Gating probabilities
        logits = self.w_gate(x_flat)  # [N, E]
        probs = torch.softmax(logits, dim=-1)  # [N, E]

        # Top-1 expert assignment
        top1 = probs.argmax(dim=-1)  # [N]
        gate_scores = F.one_hot(top1, num_classes=self.num_experts).to(probs.dtype)  # [N, E]

        # Optional load-balancing aux loss
        aux_loss = None
        if use_aux_loss:
            importance = probs.sum(dim=0)  # [E]
            load = gate_scores.sum(dim=0)  # [E]
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

        # ------ Old implementation with nn.ModuleList of experts ------
        # self.experts = nn.ModuleList(
        #     [
        #         nn.Sequential(nn.Linear(d_model, d_ff), act_fn(), nn.Linear(d_ff, d_model))
        #         for _ in range(num_experts)
        #     ]
        # )
        # ----------------------------------------------------------

        self.act = act_fn()
        # Fused expert parameters: one big linear combination + split layer
        self.act = act_fn()
        self.W1 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))  # [E, D, H]
        self.W2 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))  # [E, H, D]

        # Xavier/kaiming-style init
        for e in range(num_experts):
            nn.init.xavier_uniform_(self.W1[e])
            nn.init.xavier_uniform_(self.W2[e])

    def forward(self, x: torch.Tensor, use_aux_loss: bool = False):
        # Support [B, T, D] or [N, D] by flattening tokens
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = orig_shape
            x_flat = x.reshape(B * T, D)
        elif x.dim() == 2:
            x_flat = x
            D = x_flat.size(-1)

        # gate_scores: [N, E] (one weight per token per expert)
        gate_scores, aux_loss = self.router(x_flat, use_aux_loss)

        # Fused experts: compute per-expert FFN for all tokens, then weight by gate scores
        # x_flat: [N, D], gate_scores: [N, E]
        # First layer: [N, D] x [E, D, H] -> [N, E, H]
        h = torch.einsum("nd,edh->neh", x_flat, self.W1)
        h = self.act(h)
        # Second layer: [N, E, H] x [E, H, D] -> [N, E, D]
        y_all = torch.einsum("neh,ehd->ned", h, self.W2)

        # Weight by gate scores and sum over experts -> [N, D]
        expert_outputs_flat = (gate_scores.unsqueeze(-1) * y_all).sum(dim=1)

        # ------ Old implementation with nn.ModuleList of experts ------
        # for i, expert in enumerate(self.experts):
        #     expert_output = expert(x)
        #     expert_outputs += gate_scores[:, i].unsqueeze(1) * expert_output
        # ----------------------------------------------------------

        # Restore original shape
        if x.dim() == 3:
            expert_outputs = expert_outputs_flat.view(B, T, D)
        else:
            expert_outputs = expert_outputs_flat

        assert expert_outputs.shape == orig_shape, (
            f"Expected output shape {orig_shape}, got {expert_outputs.shape}"
        )
        return expert_outputs, aux_loss
