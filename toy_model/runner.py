from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .config import DEFAULT_COLUMNS, MeasurementConfig, RunConfig
    from .data import DatasetBundle, export_dataset_bundle, make_dataset_bundle
    from .data_modarith import (
        export_modarith_dataset_bundle,
        load_modarith_dataset_bundle,
        make_modarith_dataset_bundle,
    )
    from .metrics import (
        build_metric_row,
        compute_alpha,
        compute_rankme,
        covariance_eigenspectrum,
        measurement_indices,
        normalize_trace,
        topk_share,
    )
    from .models import build_model
    from .models_modarith import build_modarith_model
    from .optimizers import Muon
    from .variant_utils import parse_variant_tokens
except ImportError:
    from config import DEFAULT_COLUMNS, MeasurementConfig, RunConfig
    from data import DatasetBundle, export_dataset_bundle, make_dataset_bundle
    from data_modarith import (
        export_modarith_dataset_bundle,
        load_modarith_dataset_bundle,
        make_modarith_dataset_bundle,
    )
    from metrics import (
        build_metric_row,
        compute_alpha,
        compute_rankme,
        covariance_eigenspectrum,
        measurement_indices,
        normalize_trace,
        topk_share,
    )
    from models import build_model
    from models_modarith import build_modarith_model
    from optimizers import Muon
    from variant_utils import parse_variant_tokens


@dataclass
class RunState:
    config: RunConfig
    run_dir: Path
    metrics_df: pd.DataFrame
    summary_row: Dict[str, float]
    test_loss: float
    model: Optional[torch.nn.Module] = None
    dataset: Optional[Any] = None


def _fmt_metric(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if np.isfinite(val):
        return f"{val:.4f}"
    return "nan"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class CombinedOptimizer:
    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = [opt for opt in optimizers if opt is not None]
        self.param_groups = []
        for opt in self.optimizers:
            for group in opt.param_groups:
                if "_base_lr" not in group:
                    group["_base_lr"] = float(group.get("lr", 0.0))
                self.param_groups.append(group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_or_default(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dataset_key(config: RunConfig) -> str:
    if config.task == "mod_arith_lm":
        return (
            f"task{config.task}_V{config.vocab_size}_seq{config.seq_len}_D{config.D}_"
            f"val{config.val_size}_test{config.test_size}_zc{config.zipf_c}_zo{config.zipf_o}_"
            f"cmin{config.c_min}_msf{config.min_step_frac}_ncp{config.noncoprime_prob}_"
            f"allow{int(config.allow_noncoprime)}_mix{config.mix_components_min}-{config.mix_components_max}_"
            f"palpha{config.component_weight_pareto_alpha}_nstd{config.token_noise_std}_"
            f"ndf{config.token_noise_t_df}_seed{config.seed}"
        )

    return (
        f"task{config.task}_d{config.d}_P{config.p}_beta{config.beta}_sigma{config.sigma}_"
        f"seq{config.seq_len}_D{config.D}_val{config.val_size}_test{config.test_size}_"
        f"noise{config.noise_std}_dist{config.latent_dist}_df{config.latent_df}_"
        f"aniso{config.latent_anisotropy}_gamma{config.latent_anisotropy_gamma}_seed{config.seed}"
    )


def _load_rff_dataset_bundle(path: Path) -> DatasetBundle:
    train = np.load(path / "train_split.npz")
    val = np.load(path / "val_split.npz")
    test = np.load(path / "test_split.npz")
    teacher = np.load(path / "teacher_params.npz")
    with (path / "teacher_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return DatasetBundle(
        z_train=train["z"],
        y_train=train["y"],
        z_val=val["z"],
        y_val=val["y"],
        z_test=test["z"],
        y_test=test["y"],
        omega=teacher["omega"],
        phase=teacher["phase"],
        teacher_a=teacher["teacher_a"],
        metadata=metadata,
    )


def prepare_dataset(config: RunConfig) -> tuple[Any, Path]:
    dataset_dir = config.output_root / "datasets" / _dataset_key(config)
    if dataset_dir.exists():
        if config.task == "mod_arith_lm":
            return load_modarith_dataset_bundle(dataset_dir), dataset_dir
        return _load_rff_dataset_bundle(dataset_dir), dataset_dir

    if config.task == "mod_arith_lm":
        bundle = make_modarith_dataset_bundle(
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            train_size=config.D,
            val_size=config.val_size,
            test_size=config.test_size,
            zipf_c=config.zipf_c,
            zipf_o=config.zipf_o,
            c_min=config.c_min,
            min_step_frac=config.min_step_frac,
            allow_noncoprime=config.allow_noncoprime,
            noncoprime_prob=config.noncoprime_prob,
            mix_components_min=config.mix_components_min,
            mix_components_max=config.mix_components_max,
            component_weight_pareto_alpha=config.component_weight_pareto_alpha,
            token_noise_std=config.token_noise_std,
            token_noise_t_df=config.token_noise_t_df,
            seed=config.seed,
        )
        export_modarith_dataset_bundle(bundle, dataset_dir)
        return bundle, dataset_dir

    bundle = make_dataset_bundle(
        d=config.d,
        p=config.p,
        beta=config.beta,
        sigma=config.sigma,
        seq_len=config.seq_len,
        train_size=config.D,
        val_size=config.val_size,
        test_size=config.test_size,
        noise_std=config.noise_std,
        latent_dist=config.latent_dist,
        latent_df=config.latent_df,
        latent_anisotropy=config.latent_anisotropy,
        latent_anisotropy_gamma=config.latent_anisotropy_gamma,
        seed=config.seed,
    )
    export_dataset_bundle(bundle, dataset_dir)
    return bundle, dataset_dir


def _evaluate_loss(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    task: str,
    vocab_size: int,
    batch_size: int = 2048,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            if task == "mod_arith_lm":
                xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
                yb = torch.as_tensor(y[start:end], dtype=torch.long, device=device)
                logits, _ = model(xb, return_repr=True)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="mean")
            else:
                xb = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y[start:end], dtype=torch.float32, device=device)
                pred, _ = model(xb, return_repr=True)
                loss = F.mse_loss(pred, yb, reduction="mean")
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def measure_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    measurement: MeasurementConfig,
    step: int,
    seed: int,
    namespace: str,
    device: str,
    task: str,
    vocab_size: int,
    modarith_measurement_pooling: str = "last",
) -> Dict[str, object]:
    model.eval()
    idx = measurement_indices(
        num_examples=len(x),
        n_samples=measurement.n_samples,
        fixed_samples=measurement.fixed_samples,
        seed=seed,
        step=step,
        namespace=namespace,
    )
    if idx.size == 0:
        raise ValueError("No measurement samples selected.")

    x_sel = x[idx]
    y_sel = y[idx]

    reps: List[np.ndarray] = []
    grads: List[np.ndarray] = []
    batch_size = min(2048, len(x_sel))

    with torch.no_grad():
        for start in range(0, len(x_sel), batch_size):
            end = min(len(x_sel), start + batch_size)
            if task == "mod_arith_lm":
                xb = torch.as_tensor(x_sel[start:end], dtype=torch.long, device=device)
                yb = torch.as_tensor(y_sel[start:end], dtype=torch.long, device=device)
                logits, h = model(xb, return_repr=True)
                if h is None:
                    raise RuntimeError("Model did not return representations for measurement.")

                pool = str(modarith_measurement_pooling).strip().lower()
                if pool not in {"token", "last", "mean"}:
                    raise ValueError(f"Unknown modarith_measurement_pooling: {modarith_measurement_pooling}")

                probs_true = torch.softmax(logits, dim=-1).gather(-1, yb.unsqueeze(-1)).squeeze(-1)
                residual_bt = probs_true - 1.0  # [B, T], true-class CE logit residual
                if pool == "token":
                    residual = residual_bt.detach().cpu().numpy().reshape(-1, 1)
                    h_np = h.detach().cpu().numpy().reshape(-1, h.shape[-1])
                elif pool == "last":
                    residual = residual_bt[:, -1].detach().cpu().numpy().reshape(-1, 1)
                    h_np = h[:, -1, :].detach().cpu().numpy()
                else:  # pool == "mean"
                    residual = residual_bt.mean(dim=1).detach().cpu().numpy().reshape(-1, 1)
                    h_np = h.mean(dim=1).detach().cpu().numpy()
                g_np = residual * h_np
            else:
                xb = torch.as_tensor(x_sel[start:end], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_sel[start:end], dtype=torch.float32, device=device)
                pred, h = model(xb, return_repr=True)
                if h is None:
                    raise RuntimeError("Model did not return representations for measurement.")
                residual = (pred - yb).detach().cpu().numpy()[:, None]
                h_np = h.detach().cpu().numpy()
                g_np = 2.0 * residual * h_np

            reps.append(h_np)
            grads.append(g_np)

    h_all = np.concatenate(reps, axis=0)
    g_all = np.concatenate(grads, axis=0)

    act_spectrum = covariance_eigenspectrum(h_all, center=True)
    grad_spectrum = covariance_eigenspectrum(g_all, center=True)
    metric_vals = build_metric_row(
        act_spectrum=act_spectrum,
        grad_spectrum=grad_spectrum,
        alpha_head_range=measurement.alpha_head_range,
        alpha_tail_range=measurement.alpha_tail_range,
        trace_normalize=measurement.trace_normalize,
    )

    return {
        "metrics": metric_vals,
        "act_spectrum": act_spectrum,
        "grad_spectrum": grad_spectrum,
        "indices": idx,
        "n_effective_samples": int(h_all.shape[0]),
    }


def _append_summary(summary_path: Path, row: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        return

    old = pd.read_csv(summary_path)
    old_cols = list(old.columns)
    new_cols = [c for c in row.keys() if c not in old_cols]

    if new_cols:
        cols = [*old_cols, *new_cols]
        old = old.reindex(columns=cols)
        records = old.to_dict(orient="records")
        records.append({c: row.get(c, np.nan) for c in cols})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(records)
        return

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_cols)
        writer.writerow({c: row.get(c, np.nan) for c in old_cols})


def _write_run_summary(run_dir: Path, row: Dict[str, object]) -> None:
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(row), f, indent=2)
    pd.DataFrame([row]).to_csv(run_dir / "run_summary.csv", index=False)


def _load_existing_run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    summary_json = run_dir / "run_summary.json"
    if summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    summary_csv = run_dir / "run_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if not df.empty:
            return df.iloc[0].to_dict()
    return None


def _config_matches_existing_run(config: RunConfig, run_dir: Path) -> bool:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return False

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            old_cfg = json.load(f)
    except Exception:
        return False

    def _same(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return int(a) == int(b)
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            af = float(a)
            bf = float(b)
            if np.isfinite(af) and np.isfinite(bf):
                return bool(np.isclose(af, bf, rtol=0.0, atol=1e-12))
            return (not np.isfinite(af)) and (not np.isfinite(bf))
        return str(a) == str(b)

    checks = {
        "task": config.task,
        "track": config.track,
        "variant": config.variant,
        "optimizer_name": config.optimizer_name,
        "d": config.d,
        "beta": config.beta,
        "p": config.p,
        "D": config.D,
        "B": config.B,
        "lr": config.lr,
        "seed": config.seed,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "zipf_c": config.zipf_c,
        "zipf_o": config.zipf_o,
        "c_min": config.c_min,
        "min_step_frac": config.min_step_frac,
        "allow_noncoprime": config.allow_noncoprime,
        "noncoprime_prob": config.noncoprime_prob,
        "mix_components_min": config.mix_components_min,
        "mix_components_max": config.mix_components_max,
        "component_weight_pareto_alpha": config.component_weight_pareto_alpha,
        "token_noise_std": config.token_noise_std,
        "token_noise_t_df": config.token_noise_t_df,
        "modarith_measurement_pooling": config.modarith_measurement_pooling,
        "probe_regime": config.probe_regime,
        "max_steps": config.max_steps,
        "target_loss": config.target_loss,
        "target_loss_metric": config.target_loss_metric,
        "target_loss_patience": config.target_loss_patience,
        "target_loss_min_steps": config.target_loss_min_steps,
        "save_checkpoints": config.save_checkpoints,
        "checkpoint_every": config.checkpoint_every,
        "window_size": config.window_size,
        "attention_scale": config.attention_scale,
        "num_layers": config.num_layers,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "ff_mult": config.ff_mult,
        "lm_head_softcap": config.lm_head_softcap,
        "embed_lr": config.embed_lr,
        "head_lr": config.head_lr,
        "scalar_lr": config.scalar_lr,
        "muon_lr": config.muon_lr,
        "muon_momentum": config.muon_momentum,
        "muon_backend": config.muon_backend,
        "muon_backend_steps": config.muon_backend_steps,
        "muon_notebook_lr_defaults": config.muon_notebook_lr_defaults,
        "weight_decay": config.weight_decay,
    }

    for key, val in checks.items():
        if key not in old_cfg:
            return False
        if not _same(old_cfg.get(key), val):
            return False

    old_paths = [str(x) for x in old_cfg.get("param_spectrum_paths", [])]
    new_paths = [str(x) for x in config.param_spectrum_paths]
    if old_paths != new_paths:
        return False
    old_checkpoint_steps = [int(x) for x in old_cfg.get("checkpoint_steps", [])]
    new_checkpoint_steps = [int(x) for x in config.checkpoint_steps]
    if old_checkpoint_steps != new_checkpoint_steps:
        return False

    old_meas = old_cfg.get("measurement", {})
    if not isinstance(old_meas, dict):
        return False
    meas_checks = {
        "n_samples": config.measurement.n_samples,
        "fixed_samples": config.measurement.fixed_samples,
        "trace_normalize": config.measurement.trace_normalize,
        "alpha_head_range": list(config.measurement.alpha_head_range),
        "alpha_tail_range": list(config.measurement.alpha_tail_range),
    }
    for key, val in meas_checks.items():
        if key not in old_meas:
            return False
        old_val = old_meas.get(key)
        if isinstance(val, list):
            if list(old_val) != val:
                return False
        elif not _same(old_val, val):
            return False

    return True


def _reconstruct_summary_from_metrics(
    *,
    config: RunConfig,
    ablation_name: str,
    metrics_df: pd.DataFrame,
    test_loss: float,
    summary_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if summary_row is not None:
        out = dict(summary_row)
        out.setdefault("ablation", ablation_name)
        out.setdefault("run_name", config.run_name())
        out.setdefault("variant", config.variant)
        out.setdefault("optimizer_name", _resolve_optimizer_name(config))
        out.setdefault("test_loss", test_loss)
        return out

    final = metrics_df.sort_values("step").iloc[-1].to_dict()
    final_step = int(final.get("step", config.max_steps))
    stop_reason = "target_loss" if final_step < int(config.max_steps) and config.target_loss is not None else "max_steps"
    return {
        **{k: final.get(k, np.nan) for k in DEFAULT_COLUMNS},
        "ablation": ablation_name,
        "run_name": config.run_name(),
        "test_loss": test_loss,
        "variant": config.variant,
        "optimizer_name": _resolve_optimizer_name(config),
        "stop_reason": stop_reason,
        "stopped_early": bool(final_step < int(config.max_steps)),
        "final_step": final_step,
    }


def _maybe_reuse_existing_run(
    *,
    config: RunConfig,
    ablation_name: str,
    run_dir: Path,
    return_model_and_data: bool,
) -> Optional[RunState]:
    if return_model_and_data:
        return None
    metrics_path = run_dir / "metrics_over_time.csv"
    if not metrics_path.exists():
        return None
    if not _config_matches_existing_run(config=config, run_dir=run_dir):
        return None
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty or "step" not in metrics_df.columns:
        return None
    final_step = int(pd.to_numeric(metrics_df["step"], errors="coerce").dropna().max())
    summary_row = _load_existing_run_summary(run_dir=run_dir)
    stop_reason = str((summary_row or {}).get("stop_reason", ""))
    is_complete = final_step >= int(config.max_steps) or stop_reason == "target_loss"
    if not is_complete:
        return None

    test_loss = float((summary_row or {}).get("test_loss", np.nan))
    summary = _reconstruct_summary_from_metrics(
        config=config,
        ablation_name=ablation_name,
        metrics_df=metrics_df,
        test_loss=test_loss,
        summary_row=summary_row,
    )
    print(f"[reuse] Reusing completed run: {run_dir}")
    return RunState(
        config=config,
        run_dir=run_dir,
        metrics_df=metrics_df,
        summary_row=summary,
        test_loss=test_loss,
        model=None,
        dataset=None,
    )


def _save_checkpoint(run_dir: Path, model: torch.nn.Module, step: int) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"model_step_{int(step):06d}.pt"
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
        },
        out_path,
    )
    return out_path


def _sample_train_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: str,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.integers(low=0, high=len(x_train), size=batch_size)
    if task == "mod_arith_lm":
        xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
        yb = torch.as_tensor(y_train[idx], dtype=torch.long, device=device)
    else:
        xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
    return xb, yb


def _resolve_optimizer_name(config: RunConfig) -> str:
    if config.optimizer_name != "adamw":
        return config.optimizer_name
    variant_tokens = parse_variant_tokens(config.variant)
    if "muon" in variant_tokens:
        return "muon"
    return "adamw"


def _iter_named_trainable_params(model: torch.nn.Module) -> List[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _split_optimizer_groups(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    named = _iter_named_trainable_params(model)
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "embedding": [],
        "head": [],
        "matrix": [],
        "scalar": [],
    }
    seen_ptrs: set[int] = set()

    def _add(group_name: str, param: torch.nn.Parameter) -> bool:
        ptr = int(param.data_ptr())
        if ptr in seen_ptrs:
            return False
        seen_ptrs.add(ptr)
        groups[group_name].append(param)
        return True

    for name, param in named:
        lname = name.lower()
        if any(key in lname for key in ("tok_embed", "wte", "vte", "pos_embed", "value_tok_embed")):
            _add("embedding", param)

    for name, param in named:
        lname = name.lower()
        if lname.endswith("lm_head.weight") or lname.endswith("readout.weight"):
            _add("head", param)

    for _, param in named:
        ptr = int(param.data_ptr())
        if ptr in seen_ptrs:
            continue
        target = "matrix" if param.ndim >= 2 else "scalar"
        _add(target, param)

    return groups


def _make_adamw(params: List[torch.nn.Parameter], lr: float, weight_decay: float) -> Optional[torch.optim.Optimizer]:
    if not params:
        return None
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _set_optimizer_lrs(optimizer, new_base_lr: float, reference_base_lr: float) -> None:
    ref = float(reference_base_lr) if float(reference_base_lr) > 0 else 1.0
    scale = float(new_base_lr) / ref
    for group in optimizer.param_groups:
        base = float(group.get("_base_lr", group.get("lr", new_base_lr)))
        group["lr"] = base * scale


def _resolve_muon_group_lrs(config: RunConfig) -> tuple[float, float, float, float, float]:
    explicit = [config.embed_lr, config.head_lr, config.scalar_lr, config.muon_lr]
    use_notebook_defaults = bool(config.muon_notebook_lr_defaults) and all(x is None for x in explicit)

    if use_notebook_defaults:
        # Preserve the notebook split ratios, but anchor them to the run's swept
        # base LR so Muon-family stages still participate in LR search.
        notebook_ref_lr = 1e-3
        scale = float(config.lr) / notebook_ref_lr if float(config.lr) > 0 else 1.0
        embed_lr = 1e-3 * scale
        head_lr = 1e-3 * scale
        scalar_lr = 1e-3 * scale
        muon_lr = 5e-2 * scale
        # Notebook split used AdamW(weight_decay=0.01) for non-Muon groups.
        adam_weight_decay = 1e-2 if float(config.weight_decay) == 0.0 else float(config.weight_decay)
        return embed_lr, head_lr, scalar_lr, muon_lr, adam_weight_decay

    embed_lr = float(config.embed_lr if config.embed_lr is not None else config.lr)
    head_lr = float(config.head_lr if config.head_lr is not None else config.lr)
    scalar_lr = float(config.scalar_lr if config.scalar_lr is not None else config.lr)
    muon_lr = float(config.muon_lr if config.muon_lr is not None else config.lr)
    return embed_lr, head_lr, scalar_lr, muon_lr, float(config.weight_decay)


def _build_optimizer(model: torch.nn.Module, config: RunConfig):
    name = _resolve_optimizer_name(config)
    if name == "muon":
        groups = _split_optimizer_groups(model)
        embed_lr, head_lr, scalar_lr, muon_lr, adam_weight_decay = _resolve_muon_group_lrs(config)

        optimizers = [
            _make_adamw(groups["embedding"], lr=embed_lr, weight_decay=adam_weight_decay),
            _make_adamw(groups["head"], lr=head_lr, weight_decay=adam_weight_decay),
            Muon(
                groups["matrix"],
                lr=muon_lr,
                momentum=float(config.muon_momentum),
                nesterov=True,
                backend=str(config.muon_backend),
                backend_steps=int(config.muon_backend_steps),
            )
            if groups["matrix"]
            else None,
            _make_adamw(groups["scalar"], lr=scalar_lr, weight_decay=adam_weight_decay),
        ]
        out = CombinedOptimizer([opt for opt in optimizers if opt is not None])
        setattr(
            out,
            "_toy_optimizer_profile",
            {
                "name": "muon_split",
                "embed_lr": float(embed_lr),
                "head_lr": float(head_lr),
                "scalar_lr": float(scalar_lr),
                "muon_lr": float(muon_lr),
                "adam_weight_decay": float(adam_weight_decay),
                "muon_backend": str(config.muon_backend),
                "muon_backend_steps": int(config.muon_backend_steps),
                "muon_momentum": float(config.muon_momentum),
            },
        )
        return out
    out = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    setattr(
        out,
        "_toy_optimizer_profile",
        {
            "name": "adamw",
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
        },
    )
    return out


def _build_model_for_task(config: RunConfig, dataset: Any, device: str) -> torch.nn.Module:
    if config.task == "mod_arith_lm":
        model = build_modarith_model(
            track=config.track,
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.num_layers,
            ff_mult=config.ff_mult,
            variant=config.variant,
            window_size=config.window_size,
            attention_scale=config.attention_scale,
            lm_head_softcap=config.lm_head_softcap,
        )
        return model.to(device)

    omega = torch.as_tensor(dataset.omega, dtype=torch.float32)
    phase = torch.as_tensor(dataset.phase, dtype=torch.float32)
    model = build_model(
        track=config.track,
        d=config.d,
        p=config.p,
        seq_len=config.seq_len,
        omega=omega,
        phase=phase,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.num_layers,
        ff_mult=config.ff_mult,
        variant=config.variant,
        window_size=config.window_size,
        attention_scale=config.attention_scale,
    )
    return model.to(device)


def _sanitize_path_for_filename(path: str) -> str:
    safe = []
    for ch in str(path):
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        elif ch == ".":
            safe.append("_")
        else:
            safe.append("-")
    return "".join(safe)


def _resolve_object_by_path(root: Any, path: str) -> Optional[Any]:
    obj = root
    for part in str(path).split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(obj, (list, tuple)) and 0 <= idx < len(obj):
                obj = obj[idx]
                continue
            if hasattr(obj, "__getitem__"):
                try:
                    obj = obj[idx]
                    continue
                except Exception:
                    return None
            return None
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _resolve_param_tensor(model: torch.nn.Module, path: str) -> Optional[torch.Tensor]:
    obj = _resolve_object_by_path(model, path)
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Parameter):
        return obj
    if torch.is_tensor(obj):
        return obj
    return None


def _default_param_spectrum_paths(config: RunConfig, model: torch.nn.Module) -> List[str]:
    paths: List[str] = []
    if config.task == "mod_arith_lm":
        paths.append("tok_embed.weight")
        paths.append("lm_head.weight")
    else:
        paths.append("input_proj.weight")
        paths.append("readout.weight")

    if hasattr(model, "blocks") and len(getattr(model, "blocks", [])) > 0:
        last = len(model.blocks) - 1
        # Prefer current scale-style modarith names.
        paths.append("blocks.0.attn.c_proj.weight")
        paths.append("blocks.0.mlp.c_proj.weight")
        if last > 0:
            paths.append(f"blocks.{last}.attn.c_proj.weight")
            paths.append(f"blocks.{last}.mlp.c_proj.weight")
        # Backward-compatible fallback names from the older block implementation.
        paths.append("blocks.0.attn.out_proj.weight")
        paths.append("blocks.0.ff.2.weight")
        if last > 0:
            paths.append(f"blocks.{last}.attn.out_proj.weight")
            paths.append(f"blocks.{last}.ff.2.weight")

    effective: List[str] = []
    seen_ptrs: set[int] = set()
    for path in paths:
        t = _resolve_param_tensor(model, path)
        if t is None or t.ndim != 2:
            continue
        ptr = int(t.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        effective.append(path)
    return effective


def _effective_param_spectrum_paths(config: RunConfig, model: torch.nn.Module) -> List[str]:
    if config.param_spectrum_paths:
        requested = [str(x).strip() for x in config.param_spectrum_paths if str(x).strip()]
    else:
        requested = _default_param_spectrum_paths(config=config, model=model)

    effective: List[str] = []
    seen_ptrs: set[int] = set()
    for path in requested:
        t = _resolve_param_tensor(model, path)
        if t is None or t.ndim != 2:
            continue
        ptr = int(t.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        effective.append(path)
    return effective


def _spectrum_stats(spectrum: np.ndarray, measurement: MeasurementConfig) -> Dict[str, float]:
    s = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return {
            "rankme": float("nan"),
            "alpha_head": float("nan"),
            "alpha_tail": float("nan"),
            "top10": float("nan"),
        }
    s = np.maximum(s, 0.0)
    s = np.sort(s)[::-1]
    if measurement.trace_normalize:
        s = normalize_trace(s)
    return {
        "rankme": float(compute_rankme(s)),
        "alpha_head": float(compute_alpha(s, *measurement.alpha_head_range)),
        "alpha_tail": float(compute_alpha(s, *measurement.alpha_tail_range)),
        "top10": float(topk_share(s, k=10)),
    }


def _compute_weight_svd_spectrum(model: torch.nn.Module, param_path: str) -> np.ndarray:
    theta = _resolve_param_tensor(model, param_path)
    if theta is None or theta.ndim != 2:
        return np.array([], dtype=np.float64)
    with torch.no_grad():
        s = torch.linalg.svdvals(theta.detach().float())
    return s.cpu().numpy()


def _compute_gradient_svd_spectrum(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    param_path: str,
    num_samples: int,
    device: str,
    task: str,
    vocab_size: int,
) -> np.ndarray:
    theta = _resolve_param_tensor(model, param_path)
    if theta is None or theta.ndim != 2:
        return np.array([], dtype=np.float64)
    if num_samples <= 0 or sample_indices.size == 0:
        return np.array([], dtype=np.float64)

    selected = [int(i) for i in sample_indices[: int(num_samples)]]
    if not selected:
        return np.array([], dtype=np.float64)

    model.eval()
    orig_requires_grad = [bool(p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    theta.requires_grad_(True)

    grad_rows: List[torch.Tensor] = []
    try:
        for idx in selected:
            model.zero_grad(set_to_none=True)
            if task == "mod_arith_lm":
                xb = torch.as_tensor(x[idx : idx + 1], dtype=torch.long, device=device)
                yb = torch.as_tensor(y[idx : idx + 1], dtype=torch.long, device=device)
                logits, _ = model(xb, return_repr=True)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="mean")
            else:
                xb = torch.as_tensor(x[idx : idx + 1], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y[idx : idx + 1], dtype=torch.float32, device=device)
                pred, _ = model(xb, return_repr=True)
                loss = F.mse_loss(pred, yb, reduction="mean")

            if not torch.isfinite(loss):
                continue
            loss.backward()
            if theta.grad is None:
                continue
            grad_rows.append(theta.grad.detach().reshape(-1).cpu())
    finally:
        model.zero_grad(set_to_none=True)
        for p, req in zip(model.parameters(), orig_requires_grad):
            p.requires_grad_(req)

    if len(grad_rows) == 0:
        return np.array([], dtype=np.float64)
    G = torch.stack(grad_rows, dim=0).float()
    with torch.no_grad():
        s = torch.linalg.svdvals(G)
    return s.cpu().numpy()


def train_toy_run(
    config: RunConfig,
    ablation_name: str,
    write_summary: bool = True,
    save_spectra: bool = True,
    device: Optional[str] = None,
    return_model_and_data: bool = False,
) -> RunState:
    if config.track.lower() not in {
        "a",
        "b",
        "transformer",
        "linear",
        "transformer_rff",
        "linear_rff",
        "transformer_lm",
        "linear_lm",
    }:
        raise ValueError("track must be Track A (transformer) or Track B (linear)")
    if config.task not in {"rff_regression", "mod_arith_lm"}:
        raise ValueError("task must be one of: rff_regression, mod_arith_lm")
    if str(config.modarith_measurement_pooling).strip().lower() not in {"token", "last", "mean"}:
        raise ValueError("modarith_measurement_pooling must be one of: token, last, mean")
    if str(config.probe_regime).strip().lower() not in {"clean", "matched", "both", "auto"}:
        raise ValueError("probe_regime must be one of: clean, matched, both, auto")
    if str(config.target_loss_metric).strip().lower() not in {"val", "train"}:
        raise ValueError("target_loss_metric must be one of: val, train")
    if config.target_loss is not None and not np.isfinite(float(config.target_loss)):
        raise ValueError("target_loss must be finite when provided")
    checkpoint_steps = tuple(
        sorted(
            {
                int(step)
                for step in config.checkpoint_steps
                if step is not None and 0 <= int(step) <= int(config.max_steps)
            }
        )
    )
    if checkpoint_steps != tuple(int(step) for step in config.checkpoint_steps):
        config = replace(config, checkpoint_steps=checkpoint_steps)
    checkpoint_step_set = set(checkpoint_steps)
    checkpointing_enabled = bool(
        config.save_checkpoints and (int(config.checkpoint_every) > 0 or bool(checkpoint_step_set))
    )

    run_root = config.output_root / ablation_name
    run_dir = run_root / config.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    reused_state = _maybe_reuse_existing_run(
        config=config,
        ablation_name=ablation_name,
        run_dir=run_dir,
        return_model_and_data=return_model_and_data,
    )
    if reused_state is not None:
        return reused_state

    _set_seed(config.seed)
    device = _device_or_default(device)

    dataset, dataset_dir = prepare_dataset(config)
    # Keep fixed-sample spectral measurements aligned across comparable runs that
    # share the same cached dataset, rather than changing the sample subset per run.
    measurement_namespace = f"{ablation_name}:{dataset_dir.resolve()}"

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(asdict(config)), f, indent=2)
    with (run_dir / "dataset_ref.txt").open("w", encoding="utf-8") as f:
        f.write(str(dataset_dir.resolve()))
        f.write("\n")

    model = _build_model_for_task(config=config, dataset=dataset, device=device)

    checkpoint_time_s = 0.0
    last_checkpoint_step: Optional[int] = None
    if checkpointing_enabled:
        ckpt_t0 = time.perf_counter()
        _save_checkpoint(run_dir=run_dir, model=model, step=0)
        checkpoint_time_s += float(time.perf_counter() - ckpt_t0)
        last_checkpoint_step = 0

    if config.task == "mod_arith_lm":
        x_train, y_train = dataset.x_train, dataset.y_train
        x_val, y_val = dataset.x_val, dataset.y_val
        x_test, y_test = dataset.x_test, dataset.y_test
    else:
        x_train, y_train = dataset.z_train, dataset.y_train
        x_val, y_val = dataset.z_val, dataset.y_val
        x_test, y_test = dataset.z_test, dataset.y_test

    optimizer = _build_optimizer(model, config)
    optimizer_profile = dict(getattr(optimizer, "_toy_optimizer_profile", {}))
    rng = np.random.default_rng(config.seed + 11)
    wall_clock_start = time.perf_counter()

    current_batch = config.B
    current_lr = config.lr
    train_time_s = 0.0
    val_time_s = 0.0
    measurement_time_s = 0.0
    tokens_seen = 0
    eval_t0 = time.perf_counter()
    latest_val_loss = _evaluate_loss(
        model=model,
        x=x_val,
        y=y_val,
        device=device,
        task=config.task,
        vocab_size=config.vocab_size,
    )
    val_time_s += float(time.perf_counter() - eval_t0)
    best_val_loss = float(latest_val_loss)
    last_eval_step = 0
    heartbeat_every = max(
        1,
        int(config.log_every),
        max(1, int(config.max_steps) // 50),
    )

    rows: List[Dict[str, float]] = []
    param_rows: List[Dict[str, object]] = []
    param_paths = _effective_param_spectrum_paths(config=config, model=model) if config.save_param_spectra else []
    target_hits = 0
    stop_reason = "max_steps"

    print(
        "[run] start "
        f"run_name={config.run_name()} "
        f"variant={config.variant} "
        f"B={current_batch} lr={current_lr:g} "
        f"max_steps={config.max_steps} "
        f"target_loss={config.target_loss} target_metric={config.target_loss_metric} "
        f"log_every={config.log_every} heartbeat_every={heartbeat_every}",
        flush=True,
    )

    def _refresh_val_loss() -> None:
        nonlocal latest_val_loss, best_val_loss, last_eval_step, val_time_s
        eval_t0 = time.perf_counter()
        latest_val_loss = _evaluate_loss(
            model=model,
            x=x_val,
            y=y_val,
            device=device,
            task=config.task,
            vocab_size=config.vocab_size,
        )
        val_time_s += float(time.perf_counter() - eval_t0)
        best_val_loss = min(float(best_val_loss), float(latest_val_loss))

    def _print_heartbeat(step: int, train_loss: float, *, event: str) -> None:
        elapsed_s = float(time.perf_counter() - wall_clock_start)
        pct = 100.0 * float(step) / float(max(int(config.max_steps), 1))
        target_suffix = ""
        if config.target_loss is not None and np.isfinite(float(config.target_loss)):
            target_suffix = (
                f" target={float(config.target_loss):g} "
                f"target_hits={int(target_hits)}"
            )
        print(
            f"[run] {event} "
            f"run_name={config.run_name()} "
            f"step={int(step)}/{int(config.max_steps)} ({pct:5.1f}%) "
            f"train={_fmt_metric(train_loss)} "
            f"last_val={_fmt_metric(latest_val_loss)} "
            f"best_val={_fmt_metric(best_val_loss)} "
            f"last_eval_step={int(last_eval_step)} "
            f"B={int(current_batch)} lr={current_lr:g} "
            f"tokens={int(tokens_seen)} "
            f"elapsed_s={elapsed_s:.1f}"
            f"{target_suffix}",
            flush=True,
        )

    def _target_metric_value(train_loss: float) -> float:
        metric_name = str(config.target_loss_metric).strip().lower()
        return latest_val_loss if metric_name == "val" else train_loss

    def _update_target_hits(train_loss: float) -> bool:
        nonlocal target_hits
        if config.target_loss is None:
            return False
        if not np.isfinite(float(config.target_loss)):
            return False
        metric_value = _target_metric_value(train_loss)
        if np.isfinite(metric_value) and metric_value <= float(config.target_loss):
            target_hits += 1
        else:
            target_hits = 0
        return target_hits >= max(1, int(config.target_loss_patience))

    def _log_measurement(step: int, train_loss: float) -> None:
        nonlocal train_time_s, tokens_seen, measurement_time_s
        meas_t0 = time.perf_counter()
        measured = measure_model(
            model=model,
            x=x_train,
            y=y_train,
            measurement=config.measurement,
            step=step,
            seed=config.seed,
            namespace=measurement_namespace,
            device=device,
            task=config.task,
            vocab_size=config.vocab_size,
            modarith_measurement_pooling=config.modarith_measurement_pooling,
        )

        if save_spectra:
            spec_dir = run_dir / "spectra"
            spec_dir.mkdir(exist_ok=True, parents=True)
            np.save(spec_dir / f"cov_spectrum_step_{step:06d}.npy", measured["act_spectrum"])
            np.save(spec_dir / f"grad_spectrum_step_{step:06d}.npy", measured["grad_spectrum"])
            if config.save_param_spectra and param_paths:
                for param_path in param_paths:
                    matrix_name = _sanitize_path_for_filename(param_path)
                    weight_spec = _compute_weight_svd_spectrum(model=model, param_path=param_path)
                    gradsvd_spec = _compute_gradient_svd_spectrum(
                        model=model,
                        x=x_train,
                        y=y_train,
                        sample_indices=np.asarray(measured["indices"], dtype=np.int64),
                        param_path=param_path,
                        num_samples=config.grad_svd_samples,
                        device=device,
                        task=config.task,
                        vocab_size=config.vocab_size,
                    )

                    if weight_spec.size > 0:
                        np.save(spec_dir / f"weight_spectrum_step_{step:06d}__{matrix_name}.npy", weight_spec)
                    if gradsvd_spec.size > 0:
                        np.save(spec_dir / f"gradsvd_spectrum_step_{step:06d}__{matrix_name}.npy", gradsvd_spec)

                    weight_stats = _spectrum_stats(weight_spec, measurement=config.measurement)
                    grad_stats = _spectrum_stats(gradsvd_spec, measurement=config.measurement)
                    param_rows.append(
                        {
                            "task": config.task,
                            "track": config.track,
                            "variant": config.variant,
                            "optimizer_name": _resolve_optimizer_name(config),
                            "step": int(step),
                            "spectrum_kind": "weight_svd",
                            "matrix_path": param_path,
                            "matrix_name": matrix_name,
                            "spectrum_rankme": weight_stats["rankme"],
                            "spectrum_alpha_head": weight_stats["alpha_head"],
                            "spectrum_alpha_tail": weight_stats["alpha_tail"],
                            "spectrum_top10": weight_stats["top10"],
                            "spectrum_size": int(weight_spec.size),
                            "grad_svd_requested_samples": int(config.grad_svd_samples),
                        }
                    )
                    param_rows.append(
                        {
                            "task": config.task,
                            "track": config.track,
                            "variant": config.variant,
                            "optimizer_name": _resolve_optimizer_name(config),
                            "step": int(step),
                            "spectrum_kind": "grad_svd",
                            "matrix_path": param_path,
                            "matrix_name": matrix_name,
                            "spectrum_rankme": grad_stats["rankme"],
                            "spectrum_alpha_head": grad_stats["alpha_head"],
                            "spectrum_alpha_tail": grad_stats["alpha_tail"],
                            "spectrum_top10": grad_stats["top10"],
                            "spectrum_size": int(gradsvd_spec.size),
                            "grad_svd_requested_samples": int(config.grad_svd_samples),
                        }
                    )

        measurement_time_s += float(time.perf_counter() - meas_t0)

        row = {
            "task": config.task,
            "track": config.track,
            "variant": config.variant,
            "optimizer_name": _resolve_optimizer_name(config),
            "d": config.d,
            "beta": config.beta,
            "P": config.p,
            "D": config.D,
            "B": current_batch,
            "lr": current_lr,
            "latent_dist": config.latent_dist,
            "latent_df": config.latent_df,
            "latent_anisotropy": config.latent_anisotropy,
            "latent_anisotropy_gamma": config.latent_anisotropy_gamma,
            "modarith_measurement_pooling": config.modarith_measurement_pooling,
            "step": step,
            "loss": latest_val_loss,
            "n_samples_measurement": int(measured["n_effective_samples"]),
            "train_loss": train_loss,
            "train_time_s": train_time_s,
            "val_time_s": val_time_s,
            "checkpoint_time_s": checkpoint_time_s,
            "measurement_time_s": measurement_time_s,
            "tokens_seen": tokens_seen,
            **measured["metrics"],
        }
        rows.append(row)

    _log_measurement(step=0, train_loss=float("nan"))

    final_step = 0
    for step in range(1, config.max_steps + 1):
        if config.transition_step is not None and step == config.transition_step:
            if config.transition_lr is not None:
                current_lr = config.transition_lr
                _set_optimizer_lrs(optimizer=optimizer, new_base_lr=current_lr, reference_base_lr=config.lr)
            if config.transition_batch is not None:
                current_batch = config.transition_batch

        model.train()
        t0 = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        t1 = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        if t0 is None:
            wall_t0 = time.perf_counter()
        else:
            t0.record()
        xb, yb = _sample_train_batch(
            x_train=x_train,
            y_train=y_train,
            batch_size=current_batch,
            rng=rng,
            device=device,
            task=config.task,
        )

        if config.task == "mod_arith_lm":
            logits, _ = model(xb, return_repr=True)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), yb.reshape(-1), reduction="mean")
        else:
            pred, _ = model(xb, return_repr=True)
            loss = F.mse_loss(pred, yb, reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if t1 is None:
            train_time_s += float(time.perf_counter() - wall_t0)
        else:
            t1.record()
            torch.cuda.synchronize()
            train_time_s += float(t0.elapsed_time(t1) / 1000.0)
        tokens_seen += int(current_batch * config.seq_len)

        train_loss = float(loss.detach().item())
        should_save_checkpoint = False
        if checkpointing_enabled and config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            should_save_checkpoint = True
        if checkpointing_enabled and step in checkpoint_step_set:
            should_save_checkpoint = True
        if should_save_checkpoint and last_checkpoint_step != step:
            ckpt_t0 = time.perf_counter()
            _save_checkpoint(run_dir=run_dir, model=model, step=step)
            checkpoint_time_s += float(time.perf_counter() - ckpt_t0)
            last_checkpoint_step = step
        should_eval = step % config.eval_every == 0 or step == config.max_steps
        if should_eval:
            _refresh_val_loss()
            last_eval_step = int(step)
            if step >= int(config.target_loss_min_steps) and _update_target_hits(train_loss=train_loss):
                # Always record the terminal step before breaking so the saved
                # metrics row, final_step, and stop_reason stay consistent.
                _log_measurement(step=step, train_loss=train_loss)
                stop_reason = "target_loss"
                final_step = int(step)
                _print_heartbeat(step=step, train_loss=train_loss, event="target_reached")
                break
        if step % config.measurement_every == 0 or step == config.max_steps:
            _log_measurement(step=step, train_loss=train_loss)
        if step == 1 or step % heartbeat_every == 0 or step == config.max_steps:
            _print_heartbeat(step=step, train_loss=train_loss, event="heartbeat")
        final_step = int(step)
    if rows:
        final_step = int(rows[-1]["step"])
    stopped_early = bool(final_step < int(config.max_steps))

    if checkpointing_enabled and last_checkpoint_step != final_step:
        ckpt_t0 = time.perf_counter()
        _save_checkpoint(run_dir=run_dir, model=model, step=final_step)
        checkpoint_time_s += float(time.perf_counter() - ckpt_t0)

    test_loss = _evaluate_loss(
        model=model,
        x=x_test,
        y=y_test,
        device=device,
        task=config.task,
        vocab_size=config.vocab_size,
    )

    metrics_df = pd.DataFrame(rows)
    for col in DEFAULT_COLUMNS:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_df = metrics_df[[*DEFAULT_COLUMNS, *[c for c in metrics_df.columns if c not in DEFAULT_COLUMNS]]]
    metrics_df.to_csv(run_dir / "metrics_over_time.csv", index=False)
    if config.save_param_spectra:
        pd.DataFrame(param_rows).to_csv(run_dir / "param_spectra_over_time.csv", index=False)

    final = metrics_df.sort_values("step").iloc[-1].to_dict()
    summary_row = {
        **{k: final.get(k, np.nan) for k in DEFAULT_COLUMNS},
        "ablation": ablation_name,
        "run_name": config.run_name(),
        "test_loss": test_loss,
        "train_time_s": train_time_s,
        "val_time_s": val_time_s,
        "checkpoint_time_s": checkpoint_time_s,
        "measurement_time_s": measurement_time_s,
        "variant": config.variant,
        "optimizer_name": _resolve_optimizer_name(config),
        "window_size": config.window_size,
        "attention_scale": config.attention_scale,
        "embed_lr": config.embed_lr,
        "head_lr": config.head_lr,
        "scalar_lr": config.scalar_lr,
        "muon_lr": config.muon_lr,
        "muon_momentum": config.muon_momentum,
        "muon_backend": config.muon_backend,
        "muon_backend_steps": config.muon_backend_steps,
        "muon_notebook_lr_defaults": config.muon_notebook_lr_defaults,
        "lm_head_softcap": config.lm_head_softcap,
        "weight_decay": config.weight_decay,
        "effective_embed_lr": optimizer_profile.get("embed_lr", np.nan),
        "effective_head_lr": optimizer_profile.get("head_lr", np.nan),
        "effective_scalar_lr": optimizer_profile.get("scalar_lr", np.nan),
        "effective_muon_lr": optimizer_profile.get("muon_lr", np.nan),
        "effective_adam_weight_decay": optimizer_profile.get("adam_weight_decay", config.weight_decay),
        "effective_optimizer_name": optimizer_profile.get("name", _resolve_optimizer_name(config)),
        "latent_dist": config.latent_dist,
        "latent_df": config.latent_df,
        "latent_anisotropy": config.latent_anisotropy,
        "latent_anisotropy_gamma": config.latent_anisotropy_gamma,
        "vocab_size": config.vocab_size,
        "zipf_c": config.zipf_c,
        "zipf_o": config.zipf_o,
        "c_min": config.c_min,
        "min_step_frac": config.min_step_frac,
        "allow_noncoprime": config.allow_noncoprime,
        "noncoprime_prob": config.noncoprime_prob,
        "mix_components_min": config.mix_components_min,
        "mix_components_max": config.mix_components_max,
        "component_weight_pareto_alpha": config.component_weight_pareto_alpha,
        "token_noise_std": config.token_noise_std,
        "token_noise_t_df": config.token_noise_t_df,
        "modarith_measurement_pooling": config.modarith_measurement_pooling,
        "probe_regime": config.probe_regime,
        "fixed_samples": config.measurement.fixed_samples,
        "trace_normalize": config.measurement.trace_normalize,
        "save_checkpoints": config.save_checkpoints,
        "checkpoint_every": config.checkpoint_every,
        "checkpoint_steps": ",".join(str(int(step)) for step in checkpoint_steps),
        "save_param_spectra": config.save_param_spectra,
        "grad_svd_samples": config.grad_svd_samples,
        "target_loss": config.target_loss,
        "target_loss_metric": config.target_loss_metric,
        "target_loss_patience": config.target_loss_patience,
        "target_loss_min_steps": config.target_loss_min_steps,
        "stop_reason": stop_reason,
        "stopped_early": stopped_early,
        "final_step": final_step,
        "transition_step": config.transition_step,
        "transition_batch": config.transition_batch,
        "transition_lr": config.transition_lr,
    }

    print(
        "[run] finish "
        f"run_name={config.run_name()} "
        f"final_step={int(summary_row['final_step'])} "
        f"stop_reason={summary_row['stop_reason']} "
        f"final_val={_fmt_metric(summary_row.get('loss', np.nan))} "
        f"test={_fmt_metric(summary_row.get('test_loss', np.nan))} "
        f"train_time_s={float(train_time_s):.1f} "
        f"val_time_s={float(val_time_s):.1f} "
        f"measurement_time_s={float(measurement_time_s):.1f} "
        f"checkpoint_time_s={float(checkpoint_time_s):.1f}",
        flush=True,
    )
    _write_run_summary(run_dir=run_dir, row=summary_row)

    if write_summary:
        _append_summary(config.output_root / "toy_summary.csv", summary_row)

    return RunState(
        config=config,
        run_dir=run_dir,
        metrics_df=metrics_df,
        summary_row=summary_row,
        test_loss=test_loss,
        model=model if return_model_and_data else None,
        dataset=dataset if return_model_and_data else None,
    )


def matched_loss_rows(df: pd.DataFrame, target_losses: Iterable[float]) -> pd.DataFrame:
    out = []
    work = df.dropna(subset=["loss"]).copy()
    if work.empty:
        return pd.DataFrame()
    for target in target_losses:
        idx = (work["loss"] - float(target)).abs().idxmin()
        row = work.loc[idx].copy()
        row["matched_loss_target"] = float(target)
        row["matched_loss_error"] = float(abs(row["loss"] - float(target)))
        out.append(row)
    return pd.DataFrame(out)
