"""
nnm_module.py — Nuclear Norm Matching add-on for DistiLLMTrainer.

Reuses model.projectors (Linear d_s -> d_t per layer) created in trainer.
Pre-computes teacher centroids once before training.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Newton-Schulz polar factor + nuclear norm
# ═══════════════════════════════════════════════════════════════

_NS_COEFFS = (15 / 8, -10 / 8, 3 / 8)


def newton_schulz_polar(M: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    assert M.ndim == 2
    dtype = M.dtype
    transposed = False
    if M.shape[0] < M.shape[1]:
        M = M.T
        transposed = True
    X = M / (M.norm() + 1e-7)
    a, b, c = _NS_COEFFS
    for _ in range(n_iters):
        A = X.T @ X
        X = a * X + b * (X @ A) + c * (X @ (A @ A))
    if transposed:
        X = X.T
    return X.to(dtype)


class _NuclearNormNS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, n_iters):
        with torch.no_grad():
            P = newton_schulz_polar(M.detach(), n_iters)
        ctx.save_for_backward(P)
        return (P * M).sum()

    @staticmethod
    def backward(ctx, grad_output):
        (P,) = ctx.saved_tensors
        return grad_output * P, None


def nuclear_norm_ns(M, n_iters=5):
    return _NuclearNormNS.apply(M, n_iters)


# ═══════════════════════════════════════════════════════════════
#  Running centroids (used only during pre-pass)
# ═══════════════════════════════════════════════════════════════

class RunningCentroids:
    def __init__(self, K, d, eta, T_dead, device):
        self.K, self.d, self.eta, self.T_dead = K, d, eta, T_dead
        self.device = device
        self.C = torch.randn(K, d, device=device, dtype=torch.float32) * 0.01
        self.dead = torch.zeros(K, device=device, dtype=torch.int32)
        self._step = 0

    @torch.no_grad()
    def update(self, H):
        H = H.to(self.device).float()
        if H.shape[0] == 0:
            return
        self._step += 1
        eta = self.eta / (1 + 0.001 * self._step)
        dists = torch.cdist(H, self.C)
        assign = dists.argmin(dim=1)
        for k in range(self.K):
            mask = (assign == k)
            if mask.any():
                self.C[k] = (1 - eta) * self.C[k] + eta * H[mask].mean(0)
                self.dead[k] = 0
            else:
                self.dead[k] += 1
                if self.dead[k] >= self.T_dead:
                    self.C[k] = H[random.randint(0, len(H) - 1)].clone()
                    self.dead[k] = 0


# ═══════════════════════════════════════════════════════════════
#  Utils
# ═══════════════════════════════════════════════════════════════

def make_R(d: int, d_prime: int, device, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, d_prime, generator=g) / math.sqrt(d_prime)
    return R.to(device).float()


def layer_weight(l: int, L: int, sigma: float = 0.15) -> float:
    return math.exp(-((l / L - 0.5) ** 2) / (2 * sigma ** 2))
    # return 1.0 if 0.4 <= l / L <= 0.85 else 0.5



def select_mid_layers(n_layers: int, n_mid: int = 4) -> list:
    """40-85% range, dedup."""
    import numpy as np
    lo = max(1, int(0.4 * n_layers))
    hi = min(n_layers, int(0.85 * n_layers))
    if lo >= hi:
        lo = max(0, hi - n_mid)
    return sorted(set(int(i) for i in np.linspace(lo, hi, n_mid, dtype=int).tolist()))


# ═══════════════════════════════════════════════════════════════
#  NNM loss per layer
# ═══════════════════════════════════════════════════════════════

def nnm_loss_one_layer(
    H_s_proj: torch.Tensor,   # student hidden projected to teacher dim, [N, d_t]  (with grad)
    H_t:      torch.Tensor,   # teacher hidden [N, d_t]                           (no grad)
    C_t:      torch.Tensor,   # teacher centroids [K, d_t]                        (precomputed, no grad)
    R:        torch.Tensor,   # random projection [d_t, d_prime]
    lw:       float,
    ns_iters: int = 5,
) -> torch.Tensor:
    H_s_proj = H_s_proj.float()
    H_t      = H_t.float().detach()
    C_t      = C_t.float().detach()
    R        = R.float()

    M_s = torch.cat([C_t, H_s_proj], dim=0) @ R
    M_t = torch.cat([C_t, H_t],      dim=0) @ R

    m, n = M_s.shape
    scale = math.sqrt(m * n)

    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale
    nn_t = (nuclear_norm_ns(M_t, ns_iters) / scale).detach()
    return lw * (nn_s - nn_t) ** 2


# ═══════════════════════════════════════════════════════════════
#  Main NNM loss across selected layers
# ═══════════════════════════════════════════════════════════════

def compute_nnm_loss(
    projectors,                 # nn.ModuleList — model.projectors (Linear d_s -> d_t)
    s_hidden_states,            # tuple of [B, T, d_s] (output_hidden_states=True)
    t_hidden_states,            # tuple of [B, T, d_t]
    labels,                     # [B, T] with -100 for prompt tokens
    chosen_size,                # first half of batch is chosen, rest is rejected
    student_layer_mapping,
    teacher_layer_mapping,
    t_centroids,                # dict[s_lid -> tensor [K, d_t]]
    R,                          # [d_t, d_prime]
    layer_weights,              # dict[s_lid -> float]
    target,                     # "concatenated" | "chosen" | "both"
    ns_iters=5,
    chosen_weight=1.0,
    rejected_weight=0.5,
):
    device = labels.device
    total_loss = torch.tensor(0.0, device=device)
    n_layers = len(student_layer_mapping)
    if n_layers == 0:
        return total_loss

    label_mask = (labels != -100)

    for s_lid, t_lid, projector in zip(student_layer_mapping, teacher_layer_mapping, projectors):
        s_h = s_hidden_states[s_lid]
        t_h = t_hidden_states[t_lid]
        C_t = t_centroids[s_lid]
        lw  = layer_weights.get(s_lid, 1.0)

        d_s = s_h.shape[-1]
        d_t = t_h.shape[-1]
        s_flat = s_h.reshape(-1, d_s)
        t_flat = t_h.reshape(-1, d_t)

        if target == "concatenated":
            mask = label_mask.reshape(-1)
            if not mask.any():
                continue
            s_proj = projector(s_flat[mask])
            t_act  = t_flat[mask]
            total_loss = total_loss + nnm_loss_one_layer(s_proj, t_act, C_t, R, lw, ns_iters)

        elif target == "chosen":
            mask = label_mask.clone()
            mask[chosen_size:] = False
            mask = mask.reshape(-1)
            if not mask.any():
                continue
            s_proj = projector(s_flat[mask])
            t_act  = t_flat[mask]
            total_loss = total_loss + nnm_loss_one_layer(s_proj, t_act, C_t, R, lw, ns_iters)

        elif target == "both":
            mc = label_mask.clone(); mc[chosen_size:] = False
            mr = label_mask.clone(); mr[:chosen_size] = False
            mc = mc.reshape(-1); mr = mr.reshape(-1)

            if mc.any():
                s_proj_c = projector(s_flat[mc])
                t_act_c  = t_flat[mc]
                total_loss = total_loss + chosen_weight * nnm_loss_one_layer(
                    s_proj_c, t_act_c, C_t, R, lw, ns_iters,
                )
            if mr.any():
                s_proj_r = projector(s_flat[mr])
                t_act_r  = t_flat[mr]
                total_loss = total_loss + rejected_weight * nnm_loss_one_layer(
                    s_proj_r, t_act_r, C_t, R, lw, ns_iters,
                )
        else:
            raise ValueError(f"Unknown target: {target}")

    return total_loss / max(n_layers, 1)


# ═══════════════════════════════════════════════════════════════
#  Pre-compute teacher centroids (called once before training)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def build_teacher_centroids(
    teacher,
    dataloader,
    student_layer_mapping,
    teacher_layer_mapping,
    K=128,
    eta=0.05,
    T_dead=50,
    max_batches=500,
    device=None,
):
    """
    Pre-compute teacher centroids in TEACHER hidden space (d_t).
    Returns dict: s_lid -> frozen [K, d_t] tensor.
    """
    from tqdm import tqdm

    if device is None:
        device = next(teacher.parameters()).device

    teacher.eval()

    # Probe d_t
    sample_batch = next(iter(dataloader))
    sample_ids = sample_batch.get("chosen_input_ids", sample_batch.get("input_ids"))
    if sample_ids is None:
        raise ValueError("Cannot find chosen_input_ids/input_ids in dataloader batch")
    sample_ids  = sample_ids[:1].to(device)
    sample_mask = torch.ones_like(sample_ids)
    out = teacher(sample_ids, attention_mask=sample_mask, output_hidden_states=True, return_dict=True)
    d_t = out.hidden_states[teacher_layer_mapping[0]].shape[-1]

    centroids = {
        s_lid: RunningCentroids(K, d_t, eta, T_dead, device)
        for s_lid in student_layer_mapping
    }

    for i, batch in enumerate(tqdm(dataloader, desc="NNM teacher centroid pre-pass", total=max_batches)):
        if i >= max_batches:
            break
        ids  = batch.get("chosen_input_ids", batch.get("input_ids"))
        mask = batch.get("chosen_attention_mask", batch.get("attention_mask"))
        if ids is None or mask is None:
            continue
        ids = ids.to(device)
        mask = mask.to(device)

        out = teacher(ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        flat_mask = mask.reshape(-1).bool()

        for s_lid, t_lid in zip(student_layer_mapping, teacher_layer_mapping):
            h = out.hidden_states[t_lid].reshape(-1, d_t)[flat_mask]
            centroids[s_lid].update(h)

    return {s_lid: rc.C.detach().clone() for s_lid, rc in centroids.items()}