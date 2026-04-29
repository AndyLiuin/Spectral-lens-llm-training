#!/usr/bin/env python3
import argparse
import gc
import importlib.util
import json
import pathlib
import traceback
from typing import Dict, List

import torch

from scale_arch_utils import MODEL_PROFILES, default_skip_attn_layer, default_vte_endpoint_k


VARIANT_TO_SCRIPT = {
    12: "12_better_window_scale.py",
    13: "13_sparse_embed_scale.py",
    14: "14_truncated_rope_scale.py",
    15: "15_softcap_scale.py",
    16: "16_fp8lmhead_scale.py",
}


def _load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model(module, variant: int, n_layer: int, n_head: int, n_embd: int, seq_len: int, device: torch.device):
    cfg_cls = module.GPTConfig
    cfg_fields = set(cfg_cls.__dataclass_fields__.keys())  # dataclass expected in all scripts

    cfg_kwargs = {
        "vocab_size": 50304,
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_embd": int(n_embd),
    }
    if "block_size" in cfg_fields:
        cfg_kwargs["block_size"] = 128
    if "seq_len" in cfg_fields:
        cfg_kwargs["seq_len"] = int(seq_len)
    if "seq_length" in cfg_fields:
        cfg_kwargs["seq_length"] = int(seq_len)
    if "skip_attn_layer" in cfg_fields:
        cfg_kwargs["skip_attn_layer"] = default_skip_attn_layer(n_layer)
    if "vte_endpoint_k" in cfg_fields:
        cfg_kwargs["vte_endpoint_k"] = default_vte_endpoint_k(n_layer)

    cfg = cfg_cls(**cfg_kwargs)

    if variant == 12:
        model = module.GPT(cfg)
    else:
        softcap = 15.0 if variant in (15, 16) else 30.0
        try:
            model = module.GPT(cfg, lm_head_softcap=softcap)
        except TypeError:
            model = module.GPT(cfg)

    model.train()
    model.to(device)
    return model


def _run_probe_once(module, variant: int, n_layer: int, n_head: int, n_embd: int, seq_len: int, window_tokens: int, cap_bytes: int, device: torch.device) -> Dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    idx = None
    tgt = None
    loss = None
    model = None

    try:
        model = _build_model(module, variant, n_layer, n_head, n_embd, seq_len, device)

        idx = torch.randint(0, 50304, (1, seq_len), device=device, dtype=torch.long)
        tgt = torch.randint(0, 50304, (1, seq_len), device=device, dtype=torch.long)
        idx[:, 0] = 50256
        tgt[:, 0] = 50256

        if variant == 12:
            window_arg = torch.tensor(int(window_tokens), device=device, dtype=torch.int32)
        else:
            window_blocks = max(1, int(window_tokens) // 128)
            window_arg = torch.tensor(window_blocks, device=device, dtype=torch.int32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(idx, tgt, window_arg)
        loss.backward()
        torch.cuda.synchronize(device)

        peak = int(torch.cuda.max_memory_allocated(device))
        return {
            "status": "ok",
            "pass_fw_bw": True,
            "under_vram_cap": peak < cap_bytes,
            "peak_bytes": peak,
            "peak_gib": peak / (1024 ** 3),
            "error": "",
        }
    except RuntimeError as exc:
        msg = str(exc)
        status = "oom" if "out of memory" in msg.lower() else "runtime_error"
        return {
            "status": status,
            "pass_fw_bw": False,
            "under_vram_cap": False,
            "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gib": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3),
            "error": msg,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "exception",
            "pass_fw_bw": False,
            "under_vram_cap": False,
            "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gib": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3),
            "error": f"{exc}\n{traceback.format_exc()}",
        }
    finally:
        if loss is not None:
            del loss
        if idx is not None:
            del idx
        if tgt is not None:
            del tgt
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def parse_csv_list(raw: str, cast_fn):
    return [cast_fn(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Probe max sequence length for variants 12-16 across model profiles.")
    parser.add_argument("--variants", type=str, default="12,13,14,15,16")
    parser.add_argument("--profiles", type=str, default="d12,d24,d36,d48")
    parser.add_argument("--seq_candidates", type=str, default="65536,49152,32768,24576,16384")
    parser.add_argument("--window_tokens", type=int, default=1792)
    parser.add_argument("--vram_cap_frac", type=float, default=0.90, help="Max allowed fraction of total VRAM.")
    parser.add_argument("--manifest_path", type=str, default="scale_mem_probe_12_16_manifest.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest_path).resolve()
    base_dir = pathlib.Path(__file__).resolve().parent

    variants = parse_csv_list(args.variants, int)
    profiles = parse_csv_list(args.profiles, str)
    seq_candidates = parse_csv_list(args.seq_candidates, int)

    if not torch.cuda.is_available():
        result = {
            "cuda_available": False,
            "device": args.device,
            "error": "CUDA is not available on this machine.",
            "variants": variants,
            "profiles": profiles,
            "seq_candidates": seq_candidates,
            "results": {},
        }
        manifest_path.write_text(json.dumps(result, indent=2))
        print(f"[warn] CUDA unavailable. Wrote manifest: {manifest_path}")
        return

    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    total_bytes = int(props.total_memory)
    cap_bytes = int(total_bytes * float(args.vram_cap_frac))

    out = {
        "cuda_available": True,
        "device": str(device),
        "gpu_name": props.name,
        "total_vram_bytes": total_bytes,
        "total_vram_gib": total_bytes / (1024 ** 3),
        "vram_cap_frac": float(args.vram_cap_frac),
        "vram_cap_bytes": cap_bytes,
        "vram_cap_gib": cap_bytes / (1024 ** 3),
        "window_tokens": int(args.window_tokens),
        "variants": variants,
        "profiles": profiles,
        "seq_candidates": seq_candidates,
        "results": {},
    }

    for variant in variants:
        if variant not in VARIANT_TO_SCRIPT:
            print(f"[warn] skipping unknown variant {variant}")
            continue
        script_path = base_dir / VARIANT_TO_SCRIPT[variant]
        if not script_path.exists():
            print(f"[warn] script missing for variant {variant}: {script_path}")
            continue

        module = _load_module(script_path, f"scale_probe_v{variant}")
        out["results"][str(variant)] = {}

        for profile in profiles:
            if profile not in MODEL_PROFILES:
                print(f"[warn] skipping unknown profile {profile}")
                continue
            n_layer, n_head, n_embd = MODEL_PROFILES[profile]
            print(f"[probe] variant={variant} profile={profile} (L={n_layer}, H={n_head}, D={n_embd})")

            attempts: List[Dict] = []
            selected = None

            for seq_len in seq_candidates:
                res = _run_probe_once(
                    module=module,
                    variant=variant,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    seq_len=seq_len,
                    window_tokens=args.window_tokens,
                    cap_bytes=cap_bytes,
                    device=device,
                )
                res["seq_len"] = int(seq_len)
                attempts.append(res)
                tag = "PASS" if (res["pass_fw_bw"] and res["under_vram_cap"]) else "FAIL"
                print(
                    f"  - seq={seq_len}: {tag} "
                    f"(status={res['status']}, peak={res['peak_gib']:.2f} GiB)"
                )
                if res["pass_fw_bw"] and res["under_vram_cap"]:
                    selected = int(seq_len)
                    break

            out["results"][str(variant)][profile] = {
                "n_layer": int(n_layer),
                "n_head": int(n_head),
                "n_embd": int(n_embd),
                "selected_seq_len": selected,
                "attempts": attempts,
            }

        del module
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
