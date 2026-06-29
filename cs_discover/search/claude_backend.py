"""Claude-subscription backend for LLMProposer.

Calls the local Claude Code CLI in non-interactive print mode (the same mechanism the
repo's judge/ scripts use), so the discovery loop runs on your Claude subscription with
no API key plumbing. This realizes the *frozen-LLM FunSearch/EvoTune baseline*; the RL
variant later swaps this for a trained policy behind the same `call_fn` interface.

Resolution order for the binary: $CLAUDE_BIN, then PATH (`claude`), then
$HOME/.local/bin/claude.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable


def resolve_claude_bin() -> str:
    env = os.environ.get("CLAUDE_BIN")
    if env:
        return os.path.expandvars(env)
    found = shutil.which("claude")
    if found:
        return found
    return os.path.expandvars("$HOME/.local/bin/claude")


def make_claude_call_fn(
    model: str = "claude-sonnet-4-5",
    timeout: int = 120,
    retries: int = 2,
) -> Callable[[str], str]:
    """Return a `call_fn(prompt) -> str` that invokes `claude --print --model <model>`.

    Raises RuntimeError only after exhausting retries; the search loop treats a raised
    proposer call as a failed (non-improving) step rather than crashing the run.
    """
    bin_path = resolve_claude_bin()

    def call_fn(prompt: str) -> str:
        args = [bin_path, "--print", "--model", model, prompt]
        last = ""
        for _ in range(retries + 1):
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout
                last = f"rc={r.returncode} stderr={r.stderr[:200]!r}"
            except Exception as e:  # noqa: BLE001
                last = f"exc={str(e)[:200]!r}"
        raise RuntimeError(f"claude call failed after {retries + 1} tries: {last}")

    return call_fn
