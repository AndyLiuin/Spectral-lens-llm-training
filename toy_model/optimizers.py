from __future__ import annotations

import torch


def zeropower_via_svd(mat: torch.Tensor, steps: int | None = None, eps: float = 1e-7) -> torch.Tensor:
    del steps, eps
    u, _, vh = torch.linalg.svd(mat, full_matrices=False)
    return u @ vh


def zeropower_via_newtonschulz5(mat: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization as used in the original Muon notebooks.
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(mat.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = mat.bfloat16()
    x = x / (x.norm() + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.t()
        transposed = True

    for _ in range(int(steps)):
        A = x @ x.t()
        B = A @ x
        x = a * x + b * B + c * (A @ B)

    if transposed:
        x = x.t()
    return x.to(dtype=mat.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer with notebook-style Newton-Schulz backend."""

    _BACKENDS = {
        "newtonschulz5": zeropower_via_newtonschulz5,
        "svd": zeropower_via_svd,
    }

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
        eps: float = 1e-7,
    ):
        if backend not in self._BACKENDS:
            raise ValueError(f"Unknown Muon backend: {backend}. Expected one of {sorted(self._BACKENDS.keys())}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=int(backend_steps),
            eps=float(eps),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            backend = self._BACKENDS[group["backend"]]
            backend_steps = int(group["backend_steps"])
            eps = float(group["eps"])

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                gg = g.add(buf, alpha=momentum) if nesterov else buf

                if gg.ndim >= 2:
                    shape = gg.shape
                    mat = gg.reshape(shape[0], -1)
                    mat = backend(mat, steps=backend_steps, eps=eps)
                    mat = mat * (max(1.0, mat.size(0) / mat.size(1)) ** 0.5)
                    gg = mat.reshape(shape).to(dtype=gg.dtype)

                p.add_(gg.to(dtype=p.dtype), alpha=-lr)

        return loss
