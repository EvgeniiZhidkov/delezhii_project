"""One fresh Codex session per answer; preserve diagnostics before validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time

from .config import ModelConfig
from .prompts import read_json

DISABLED_FEATURES = (
    "shell_tool", "apps", "plugins", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "image_generation", "view_image",
    "multi_agent", "multi_agent_v2", "skill_search", "hooks", "memories", "goals", "tool_suggest",
)


@dataclass
class BackendReply:
    raw: str | None = None
    usage: dict | None = None
    seconds: float = 0.0
    error: str | None = None
    resolved_models: list[str] = field(default_factory=list)

    def record(self):
        return asdict(self)


def codex_command(executable, model, reply_path):
    command = [executable, "--ask-for-approval", "never", "exec", "--sandbox", "read-only",
               "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json",
               "--output-last-message", str(reply_path)]
    for feature in DISABLED_FEATURES:
        command += ["--disable", feature]
    options = {"web_search": "disabled", "project_doc_max_bytes": 0,
               "skills.bundled.enabled": False, "skills.include_instructions": False,
               "features.skip_host_skill_discovery": True, "agents.enabled": False,
               "suppress_unstable_features_warning": True}
    if model.reasoning_effort:
        options["model_reasoning_effort"] = model.reasoning_effort
    for key, value in options.items():
        command += ["-c", f"{key}={json.dumps(value)}"]
    if model.model:
        command += ["--model", model.model]
    return [*command, "-"]


def inspect_events(stdout):
    completed = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        event = read_json(line)
        if not isinstance(event, dict):
            raise ValueError("Событие CLI должно быть объектом JSON")
        if event.get("type") in {"turn.failed", "error"}:
            raise ValueError(f"CLI сообщил {event['type']}")
        if event.get("type") == "turn.completed":
            completed.append(event)
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") not in {"agent_message", "reasoning", "todo_list", "error", None}:
            raise ValueError(f"Нарушена политика без инструментов: {item.get('type')}")
    if len(completed) != 1:
        raise ValueError("Ожидалось ровно одно событие turn.completed")
    usage = completed[0].get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or any(type(v) is not int or v < 0 for v in usage.values()):
            raise ValueError("Некорректная статистика токенов")
    return usage


class CodexBackend:
    def __init__(self, executable="codex"):
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise RuntimeError("Codex CLI не найден в PATH ядра Jupyter")
        self.version = subprocess.run([self.executable, "--version"], capture_output=True,
                                      text=True, timeout=15, check=True).stdout.strip()

    def __call__(self, prompt: str, model: ModelConfig, call_dir: Path):
        call_dir = Path(call_dir).resolve()
        call_dir.mkdir(parents=True, exist_ok=False)
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        reply_path = call_dir / "reply.txt"
        reply = BackendReply()
        started = time.perf_counter()
        with TemporaryDirectory(prefix="chores-e1-codex-") as temporary:
            root = Path(temporary)
            workspace, codex_home = root / "work", root / "codex"
            workspace.mkdir(); codex_home.mkdir(mode=0o700)
            source_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
            auth = source_home / "auth.json"
            if auth.is_file():
                shutil.copyfile(auth, codex_home / "auth.json")
                (codex_home / "auth.json").chmod(0o600)
            env = {k: v for k, v in os.environ.items()
                   if k in {"PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT"}}
            env.update(CODEX_HOME=str(codex_home), NO_COLOR="1", RUST_LOG="warn")
            command = codex_command(self.executable, model, reply_path)
            stdout = stderr = ""
            try:
                login = subprocess.run([self.executable, "login", "status"], cwd=workspace, env=env,
                                       capture_output=True, text=True, timeout=15)
                if login.returncode or "chatgpt" not in (login.stdout + login.stderr).lower():
                    raise RuntimeError("Выполните codex login с входом через ChatGPT в среде ядра")
                process = subprocess.run(command, input=prompt, cwd=workspace, env=env,
                                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                                         timeout=model.timeout)
                stdout, stderr = process.stdout, process.stderr
                if process.returncode:
                    raise RuntimeError(f"Codex CLI завершился с кодом {process.returncode}; см. stderr.log")
                reply.usage = inspect_events(stdout)
                if not reply_path.is_file():
                    raise RuntimeError("CLI не сохранил финальный ответ")
                reply.raw = reply_path.read_text(encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                def decode(value):
                    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
                stdout, stderr = decode(exc.stdout), decode(exc.stderr)
                reply.error = f"Таймаут Codex ({model.timeout} с)"
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                reply.error = str(exc)
            finally:
                (call_dir / "stdout.jsonl").write_text(stdout, encoding="utf-8")
                (call_dir / "stderr.log").write_text(stderr, encoding="utf-8")
                # Rollouts carry actual model IDs even if model=None. Never copy auth.json.
                sessions = codex_home / "sessions"
                models = set()
                if sessions.is_dir():
                    for path in sessions.rglob("*.jsonl"):
                        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                            try:
                                event = json.loads(line)
                            except ValueError:
                                continue
                            payload = event.get("payload")
                            if event.get("type") == "turn_context" and isinstance(payload, dict) and isinstance(payload.get("model"), str):
                                models.add(payload["model"])
                    shutil.copytree(sessions, call_dir / "sessions")
                reply.resolved_models = sorted(models)
        reply.seconds = time.perf_counter() - started
        metadata = {"cli_version": self.version, "model_config": asdict(model), "command": command,
                    "tool_policy": "no tools; CLI items checked", **reply.record()}
        (call_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return reply
