"""Weights & Biases helpers.

Falls back to W&B *offline* mode when no API key is configured so training
never crashes; offline runs can be uploaded later with:

    wandb sync wandb/offline-run-*
"""

from __future__ import annotations

import os

import wandb

PROJECT = "synthetic-forgery-detector"


def resolve_mode() -> str:
    mode = os.environ.get("WANDB_MODE")
    if mode in ("online", "offline", "disabled"):
        return mode
    try:
        if wandb.api.api_key:  # deprecated but reliable & non-interactive
            return "online"
    except Exception:
        pass
    return "offline"


def init_wandb(config: dict, run_name: str | None = None,
               notes: str = "", tags: list[str] | None = None,
               enabled: bool = True):
    if not enabled:
        return None
    mode = resolve_mode()
    try:
        run = wandb.init(project=PROJECT, name=run_name, config=config,
                         notes=notes, tags=tags or [], mode=mode, reinit=True)
        print(f"[wandb] run: {run.name} (mode={mode}, id={run.id})")
        if mode == "offline":
            print("[wandb] no API key found -> offline. Later run: "
                  "`wandb login` then `wandb sync wandb/offline-run-*`")
        return run
    except Exception as e:  # never let logging break training
        print(f"[wandb] init failed ({e}); continuing without W&B")
        return None
