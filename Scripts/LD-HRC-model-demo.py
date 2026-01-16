from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


InitMode = Literal["xavier", "orthogonal"]
ActName = Literal["relu", "elu", "gelu", "tanh", "silu"]


def _get_activation(name: ActName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear_(m: nn.Module, mode: InitMode = "xavier") -> None:
    if isinstance(m, nn.Linear):
        if mode == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif mode == "orthogonal":
            nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unsupported init mode: {mode}")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def logit(x: torch.Tensor) -> torch.Tensor:
    """
    logit(x) = log(x) - log(1-x), for x in (0,1)
    """
    return torch.log(x) - torch.log1p(-x)


@dataclass(frozen=True)
class ActorOutput:

    logits: torch.Tensor  # [B, M, K]
    mu: torch.Tensor      # [B, M]
    std: torch.Tensor     # [B, M] > 0


class TelecomHybridActor(nn.Module):


    def __init__(
        self,
        input_dim: int,                 # D_h
        num_subchannels: int,           # M
        num_femtoCell: int,             # K
        max_power: float,               # P_max
        trunk_layers: Sequence[int] = (256, 256),
        activation: ActName = "relu",
        dropout: float = 0.0,
        min_std: float = 1e-3,
        eps: float = 1e-6,
        init_mode: InitMode = "xavier",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_subchannels <= 0:
            raise ValueError(f"num_subchannels (M) must be positive, got {num_subchannels}")
        if num_femtoCell <= 1:
            raise ValueError(f"num_femtoCell (K) must be >= 2, got {num_femtoCell}")
        if max_power <= 0:
            raise ValueError(f"max_power (P_max) must be > 0, got {max_power}")
        if any(h <= 0 for h in trunk_layers):
            raise ValueError(f"trunk_layers must be positive ints, got {trunk_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        if min_std <= 0:
            raise ValueError(f"min_std must be > 0, got {min_std}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.D_h = int(input_dim)
        self.M = int(num_subchannels)
        self.K = int(num_femtoCell)
        self.Pmax = float(max_power)
        self.min_std = float(min_std)
        self.eps = float(eps)

        act = _get_activation(activation)

        # Trunk: MLP
        dims: Tuple[int, ...] = (self.D_h,) + tuple(int(x) for x in trunk_layers)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
        self.trunk = nn.Sequential(*layers)
        trunk_out = dims[-1]

        # Heads
        self.logits_head = nn.Linear(trunk_out, self.M * self.K)
        self.mu_head = nn.Linear(trunk_out, self.M)
        self.log_std_head = nn.Linear(trunk_out, self.M)

        # Init
        self.apply(lambda m: _init_linear_(m, mode=init_mode))

    def forward(self, h: torch.Tensor) -> ActorOutput:
        """
        Parameters
        ----------
        h : torch.Tensor
            Node embedding, shape [B, D_h].

        Returns
        -------
        ActorOutput
            logits [B,M,K], mu [B,M], std [B,M] (>0)
        """
        if h.ndim != 2:
            raise ValueError(f"h must be rank-2 [B,D_h], got shape {tuple(h.shape)}")
        B, Dh = h.shape
        if Dh != self.D_h:
            raise ValueError(f"Expected h last dim D_h={self.D_h}, got {Dh}")

        t = self.trunk(h)  # [B, D_a]
        logits = self.logits_head(t).view(B, self.M, self.K)  # [B,M,K]
        mu = self.mu_head(t)                                  # [B,M]
        log_std = self.log_std_head(t)                        # [B,M]
        std = F.softplus(log_std) + self.min_std              # [B,M] > 0
        return ActorOutput(logits=logits, mu=mu, std=std)

    def _discrete_dist(self, logits: torch.Tensor) -> Categorical:
        # Categorical supports batch shape [B,M]
        return Categorical(logits=logits)

    def _continuous_dist(self, mu: torch.Tensor, std: torch.Tensor) -> Normal:
        # Normal supports batch shape [B,M]
        return Normal(mu, std)

    def _power_from_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        p_norm = torch.sigmoid(z)
        p = p_norm * self.Pmax
        return p_norm, p

    def _log_prob_power_from_z(
        self, z: torch.Tensor, p_norm: torch.Tensor, dist_z: Normal
    ) -> torch.Tensor:

        logp_z = dist_z.log_prob(z)  # [B,M]
        log_det = -math.log(self.Pmax) - torch.log(p_norm * (1.0 - p_norm) + self.eps)
        return logp_z + log_det  # [B,M]

    def _log_prob_power_from_p(
        self, power: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p_norm = (power / self.Pmax).clamp(self.eps, 1.0 - self.eps)
        z = logit(p_norm)
        dist_z = Normal(mu, std)
        logp_z = dist_z.log_prob(z)
        log_det = -math.log(self.Pmax) - torch.log(p_norm * (1.0 - p_norm) + self.eps)
        logp_p = logp_z + log_det
        return logp_p, p_norm, z

    @torch.no_grad()
    def get_action(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:

        out = self.forward(h)

        # Discrete: users
        dist_u = self._discrete_dist(out.logits)
        users = dist_u.sample()                 # [B,M]
        logp_u = dist_u.log_prob(users)         # [B,M]
        ent_u = dist_u.entropy()                # [B,M]

        # Continuous: power via sigmoid-squashed Gaussian
        dist_z = self._continuous_dist(out.mu, out.std)
        z = dist_z.rsample()                    # [B,M]
        p_norm, power = self._power_from_z(z)   # [B,M], [B,M]
        logp_p = self._log_prob_power_from_z(z, p_norm, dist_z)  # [B,M]

        # Joint log-prob summed over subchannels
        logp = (logp_u + logp_p).sum(dim=-1)    # [B]

        # Entropy bonus:
        #   - Discrete entropy is exact
        #   - Continuous uses pre-squash Normal entropy (common stable approximation)
        ent_p = dist_z.entropy()                # [B,M]
        entropy = (ent_u + ent_p).sum(dim=-1)   # [B]

        return {"users": users, "power": power, "logp": logp, "entropy": entropy}

    def evaluate(
        self,
        h: torch.Tensor,
        users: torch.Tensor,
        power: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if users.ndim != 2:
            raise ValueError(f"users must be rank-2 [B,M], got shape {tuple(users.shape)}")
        if power.ndim != 2:
            raise ValueError(f"power must be rank-2 [B,M], got shape {tuple(power.shape)}")

        out = self.forward(h)
        B = out.mu.shape[0]

        if users.shape[0] != B or users.shape[1] != self.M:
            raise ValueError(
                f"users shape mismatch: expected [B={B}, M={self.M}], got {tuple(users.shape)}"
            )
        if power.shape[0] != B or power.shape[1] != self.M:
            raise ValueError(
                f"power shape mismatch: expected [B={B}, M={self.M}], got {tuple(power.shape)}"
            )

        # Discrete
        dist_u = self._discrete_dist(out.logits)
        logp_u = dist_u.log_prob(users)         # [B,M]
        ent_u = dist_u.entropy()                # [B,M]

        # Continuous (invert power -> z)
        logp_p, _, _ = self._log_prob_power_from_p(power, out.mu, out.std)  # [B,M]
        dist_z = self._continuous_dist(out.mu, out.std)
        ent_p = dist_z.entropy()                # [B,M]

        logp_new = (logp_u + logp_p).sum(dim=-1)  # [B]
        entropy = (ent_u + ent_p).sum(dim=-1)     # [B]

        return {
            "logp_new": logp_new,
            "entropy": entropy,
            "logp_u_sum": logp_u.sum(dim=-1),
            "logp_p_sum": logp_p.sum(dim=-1),
        }

ActivationName = Literal["relu", "tanh", "elu", "gelu", "silu"]


@dataclass(frozen=True)
class CriticOutput:
    """
    Shared Critic (Local View) output.

    Attributes
    ----------
    V : torch.Tensor
        Value tensor with shape [B, N, 1].
    """
    V: torch.Tensor


def _get_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear_(m: nn.Module, mode: Literal["xavier", "orthogonal"] = "xavier") -> None:
    """
    Initialize Linear layers. This is optional but helps stabilize early training.
    """
    if isinstance(m, nn.Linear):
        if mode == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif mode == "orthogonal":
            nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unsupported init mode: {mode}")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SharedCriticLocal(nn.Module):


    def __init__(
        self,
        d_h: int,
        hidden_layers: Sequence[int] = (256, 256),
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
        init_mode: Literal["xavier", "orthogonal"] = "xavier",
    ) -> None:
        super().__init__()
        if d_h <= 0:
            raise ValueError(f"d_h must be positive, got {d_h}")
        if any(h <= 0 for h in hidden_layers):
            raise ValueError(f"hidden_layers must be positive ints, got {hidden_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_h = int(d_h)
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.activation_name: ActivationName = activation
        self.dropout_p = float(dropout)
        self.use_layernorm = bool(use_layernorm)

        act = _get_activation(activation)

        dims: Tuple[int, ...] = (self.d_h,) + self.hidden_layers
        blocks = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(nn.Linear(in_dim, out_dim))
            if self.use_layernorm:
                blocks.append(nn.LayerNorm(out_dim))
            blocks.append(act)
            if self.dropout_p > 0.0:
                blocks.append(nn.Dropout(p=self.dropout_p))

        self.mlp = nn.Sequential(*blocks)
        self.value_head = nn.Linear(dims[-1], 1)

        # Initialize
        self.apply(lambda m: _init_linear_(m, mode=init_mode))

    def forward(self, H: torch.Tensor) -> CriticOutput:
        """
        Forward pass.

        Parameters
        ----------
        H : torch.Tensor
            Node embeddings with shape [B, N, D_h].

        Returns
        -------
        CriticOutput
            V with shape [B, N, 1].
        """
        if H.ndim != 3:
            raise ValueError(f"H must be rank-3 [B,N,D_h], got shape {tuple(H.shape)}")
        B, N, Dh = H.shape
        if Dh != self.d_h:
            raise ValueError(f"Expected last dim D_h={self.d_h}, got {Dh}")

        # Flatten [B,N,D_h] -> [(B*N),D_h]
        x = H.reshape(B * N, Dh).contiguous()

        # Shared MLP over nodes
        x = self.mlp(x)

        # Value head -> [(B*N), 1]
        v = self.value_head(x)

        # Restore [B,N,1]
        V = v.view(B, N, 1)
        return CriticOutput(V=V)

    @torch.no_grad()
    def value_of_agent(self, H: torch.Tensor, i: int) -> torch.Tensor:
        """
        Convenience method for debugging / single-agent view.

        Returns
        -------
        torch.Tensor
            V^{(i)} with shape [B, 1].
        """
        out = self.forward(H).V  # [B,N,1]
        if not (0 <= i < out.shape[1]):
            raise IndexError(f"agent index i out of range: i={i}, N={out.shape[1]}")
        return out[:, i, :]  # [B,1]


def critic_mse_loss(V: torch.Tensor, G: torch.Tensor, reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:

    if V.shape != G.shape:
        # Allow broadcastable shapes, but enforce final rank compatibility.
        try:
            diff = V - G
        except RuntimeError as e:
            raise ValueError(f"V and G are not broadcastable: V={tuple(V.shape)}, G={tuple(G.shape)}") from e
    else:
        diff = V - G

    loss = diff.pow(2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def critic_value_clip_loss(
    V_new: torch.Tensor,
    V_old: torch.Tensor,
    G: torch.Tensor,
    eps_v: float,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:

    if eps_v <= 0:
        raise ValueError(f"eps_v must be positive, got {eps_v}")

    V_clip = torch.clamp(V_new, V_old - eps_v, V_old + eps_v)
    loss_unclipped = (V_new - G).pow(2)
    loss_clipped = (V_clip - G).pow(2)
    loss = torch.maximum(loss_unclipped, loss_clipped)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")

InitMode = Literal["xavier", "orthogonal"]
ActName = Literal["relu", "elu", "gelu", "tanh", "silu"]


def _get_activation(name: ActName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear_(m: nn.Module, mode: InitMode = "xavier") -> None:
    if isinstance(m, nn.Linear):
        if mode == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif mode == "orthogonal":
            nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unsupported init mode: {mode}")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@dataclass(frozen=True)
class GATLayerConfig:

    in_dim: int
    head_dim: int
    num_heads: int
    negative_slope: float = 0.2
    dropout: float = 0.0
    concat: bool = True
    use_edge_weight: bool = False
    eps: float = 1e-6


class DenseGATLayer(nn.Module):


    def __init__(self, cfg: GATLayerConfig, init_mode: InitMode = "xavier") -> None:
        super().__init__()
        if cfg.in_dim <= 0 or cfg.head_dim <= 0 or cfg.num_heads <= 0:
            raise ValueError(f"Invalid dims in cfg: {cfg}")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {cfg.dropout}")
        if cfg.eps <= 0:
            raise ValueError(f"eps must be > 0, got {cfg.eps}")

        self.cfg = cfg

        # Linear projection: F_l -> (K_h * d)
        self.W = nn.Linear(cfg.in_dim, cfg.num_heads * cfg.head_dim, bias=False)

        # Additive attention vectors per head: a_src, a_dst in R^d
        self.a_src = nn.Parameter(torch.empty(cfg.num_heads, cfg.head_dim))
        self.a_dst = nn.Parameter(torch.empty(cfg.num_heads, cfg.head_dim))

        self.leaky_relu = nn.LeakyReLU(cfg.negative_slope)
        self.attn_dropout = nn.Dropout(p=cfg.dropout)

        # init
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.apply(lambda m: _init_linear_(m, mode=init_mode))

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if H.ndim != 3:
            raise ValueError(f"H must be rank-3 [B,N,F], got shape {tuple(H.shape)}")
        if A.ndim != 3:
            raise ValueError(f"A must be rank-3 [B,N,N], got shape {tuple(A.shape)}")

        B, N, Fin = H.shape
        if A.shape[0] != B or A.shape[1] != N or A.shape[2] != N:
            raise ValueError(f"Shape mismatch: H={tuple(H.shape)} vs A={tuple(A.shape)}")
        if Fin != self.cfg.in_dim:
            raise ValueError(f"Expected H last dim={self.cfg.in_dim}, got {Fin}")

        # Wh: [B, N, K_h*d] -> [B, N, K_h, d]
        Wh = self.W(H).view(B, N, self.cfg.num_heads, self.cfg.head_dim)

        # s_i = a_src^T Wh_i, t_j = a_dst^T Wh_j
        # s, t: [B, N, K_h]
        s = (Wh * self.a_src.view(1, 1, self.cfg.num_heads, self.cfg.head_dim)).sum(dim=-1)
        t = (Wh * self.a_dst.view(1, 1, self.cfg.num_heads, self.cfg.head_dim)).sum(dim=-1)

        # e_{i,j} = LeakyReLU(s_i + t_j)
        # e: [B, N, N, K_h]
        e = self.leaky_relu(s.unsqueeze(2) + t.unsqueeze(1))

        # mask: A > 0 indicates an active edge
        mask = (A > 0).unsqueeze(-1)  # [B, N, N, 1]

        # optional edge weight fusion: e' = e + log(A + eps)
        if self.cfg.use_edge_weight:
            e = e + torch.log(A.clamp_min(self.cfg.eps)).unsqueeze(-1)

        # masked softmax along neighbor dimension j
        # Use a finite large negative value instead of -inf for stability in some dtypes.
        neg_large = torch.finfo(e.dtype).min if e.dtype.is_floating_point else -1e9
        e = e.masked_fill(~mask, neg_large)

        # alpha: [B, N, N, K_h]
        alpha = torch.softmax(e, dim=2)

        # attention dropout
        alpha = self.attn_dropout(alpha)

        # aggregate: O = alpha * Wh over j
        # alpha: [B,N,N,K_h] -> [B,K_h,N,N]
        # Wh:    [B,N,K_h,d] -> [B,K_h,N,d]
        alpha_h = alpha.permute(0, 3, 1, 2).contiguous()
        Wh_h = Wh.permute(0, 2, 1, 3).contiguous()

        # out: [B, K_h, N, d] -> [B, N, K_h, d]
        out = torch.matmul(alpha_h, Wh_h).permute(0, 2, 1, 3).contiguous()

        if self.cfg.concat:
            # [B, N, K_h*d]
            out = out.view(B, N, self.cfg.num_heads * self.cfg.head_dim)
        else:
            # [B, N, d]
            out = out.mean(dim=2)

        return out


class SharedGATNetwork(nn.Module):


    def __init__(
        self,
        d_obs: int,
        d_h: int,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        activation: ActName = "elu",
        dropout_attn: float = 0.0,
        use_edge_weight: bool = False,
        add_self_loops: bool = True,
        use_residual: bool = False,
        use_layernorm: bool = False,
        final_proj: bool = True,
        init_mode: InitMode = "xavier",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_obs <= 0 or d_h <= 0:
            raise ValueError(f"d_obs and d_h must be positive, got d_obs={d_obs}, d_h={d_h}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not (0.0 <= dropout_attn < 1.0):
            raise ValueError(f"dropout_attn must be in [0,1), got {dropout_attn}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.d_obs = int(d_obs)
        self.d_h = int(d_h)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.add_self_loops = bool(add_self_loops)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        self.use_edge_weight = bool(use_edge_weight)
        self.eps = float(eps)

        self.act = _get_activation(activation)

        # Determine per-head dimension
        if head_dim is None:
            # If concat=True inside layers, output dim per layer is num_heads * head_dim.
            if d_h % num_heads != 0:
                raise ValueError(
                    f"When head_dim is None, d_h must be divisible by num_heads. "
                    f"Got d_h={d_h}, num_heads={num_heads}."
                )
            head_dim = d_h // num_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self.head_dim = int(head_dim)

        layers: list[nn.Module] = []
        norms: list[nn.Module] = []

        in_dim = self.d_obs
        for _ in range(self.num_layers):
            cfg = GATLayerConfig(
                in_dim=in_dim,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                dropout=dropout_attn,
                concat=True,
                use_edge_weight=self.use_edge_weight,
                eps=self.eps,
            )
            layers.append(DenseGATLayer(cfg, init_mode=init_mode))

            out_dim = self.num_heads * self.head_dim
            if self.use_layernorm:
                norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if self.use_layernorm else None

        # Optional final projection to exactly D_h
        if final_proj and in_dim != self.d_h:
            self.proj = nn.Linear(in_dim, self.d_h)
            _init_linear_(self.proj, mode=init_mode)
        else:
            self.proj = None
            # If no proj, the output dimension is in_dim (may differ from d_h)
            self.d_h = in_dim

    @staticmethod
    def _ensure_self_loops(A: torch.Tensor) -> torch.Tensor:
        """
        Ensure diagonal entries are positive so every node has at least itself as a neighbor.
        """
        B, N, _ = A.shape
        # Create an identity mask [1,N,N] and broadcast
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        # If A is weighted: set diag to max(diag, 1); if mask: set to 1
        # Use where to preserve existing diagonal if already > 0
        diag = torch.diagonal(A, dim1=1, dim2=2)
        needs = (diag <= 0)
        if needs.any():
            A = A.clone()
            # set diag entries to 1
            A[:, torch.arange(N), torch.arange(N)] = torch.where(
                needs, torch.ones_like(diag), diag
            )
        return A

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B,N,D_obs], got shape {tuple(X.shape)}")
        if A.ndim != 3:
            raise ValueError(f"A must be rank-3 [B,N,N], got shape {tuple(A.shape)}")

        B, N, Dobs = X.shape
        if Dobs != self.d_obs:
            raise ValueError(f"Expected X last dim D_obs={self.d_obs}, got {Dobs}")
        if A.shape[0] != B or A.shape[1] != N or A.shape[2] != N:
            raise ValueError(f"Shape mismatch: X={tuple(X.shape)} vs A={tuple(A.shape)}")

        if self.add_self_loops:
            A = self._ensure_self_loops(A)

        H = X
        for idx, gat in enumerate(self.layers):
            H_in = H
            H = gat(H, A)
            H = self.act(H)

            # residual only when shape aligns
            if self.use_residual and H.shape == H_in.shape:
                H = H + H_in

            if self.use_layernorm:
                H = self.norms[idx](H)

        if self.proj is not None:
            H = self.proj(H)

        # Final shape: [B, N, D_h]
        return H

