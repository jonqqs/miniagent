"""The tools the agent can call.

A "tool" is two things sitting next to each other:

  1. A JSON schema the model sees — name, description, input shape.
  2. A plain Python function that runs when the model asks for it.

There is nothing magic here. The model never executes anything; it only ever
emits a request saying "call `read_file` with this input". This file decides
what that actually means.
"""

import subprocess
from pathlib import Path

# Everything the agent touches is confined to the directory it was started in.
WORKSPACE = Path.cwd().resolve()

# Tools the harness must ask the user about before running. This lives here as
# data, but the *gate* itself lives in the harness (agent.py) — the tool should
# not be the thing deciding whether it is allowed to run.
REQUIRES_APPROVAL = {"write_file", "run_command"}


class ToolError(Exception):
    """A tool could not run. The message is handed back to the model verbatim."""


def _safe_path(raw: str) -> Path:
    """Resolve `raw` against the workspace and refuse anything that escapes it.

    The model supplies this string, so treat it as untrusted: `../../.ssh/id_rsa`
    is a perfectly valid thing for it to ask for. Resolve first (which collapses
    `..` and follows symlinks), then check where we landed.
    """
    path = (WORKSPACE / raw).resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ToolError(f"Path is outside the workspace: {raw}")
    return path


def read_file(path: str) -> str:
    target = _safe_path(path)
    if not target.is_file():
        raise ToolError(f"Not a file: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


def write_file(path: str, content: str) -> str:
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def list_files(path: str = ".") -> str:
    target = _safe_path(path)
    if not target.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name
        for p in target.iterdir()
        if not p.name.startswith(".")
    )
    return "\n".join(entries) or "(empty)"


def run_command(command: str) -> str:
    """Run a shell command in the workspace.

    Gated behind user approval in agent.py. Without that gate this is a remote
    shell driven by a language model, which is not a thing you want lying around.
    """
    result = subprocess.run(
        command,
        shell=True,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (result.stdout + result.stderr).strip()
    return f"exit code {result.returncode}\n{output or '(no output)'}"


# The schemas the model actually sees. Descriptions matter more than they look:
# this text is the only thing telling the model *when* to reach for each tool.
TOOLS = [
    {
        "name": "read_file",
        "description": "Read a text file from the workspace. Use this before editing a file so you are working from its real contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a file, creating parent directories as needed. Overwrites the file completely, so read it first if you mean to keep any of it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "content": {"type": "string", "description": "The full new contents of the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List the contents of a directory in the workspace. Call this when you need to find out what exists before guessing at filenames.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to the workspace root. Defaults to the root."}
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the workspace and return its exit code and output. Use this for things like running tests or checking git status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."}
            },
            "required": ["command"],
        },
    },
]

_IMPLEMENTATIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "run_command": run_command,
}


def dispatch(name: str, tool_input: dict) -> str:
    """Run the tool the model asked for and return its result as text."""
    fn = _IMPLEMENTATIONS.get(name)
    if fn is None:
        raise ToolError(f"Unknown tool: {name}")
    return fn(**tool_input)
