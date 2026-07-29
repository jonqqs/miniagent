"""A minimal agent harness, written to be read.

The whole "agent" idea fits in one loop:

    send the conversation to the model
      -> the model either answers, or asks to call a tool
      -> if it asked, run the tool and append the result to the conversation
      -> send the conversation again
      -> repeat until it stops asking

That loop is `run_turn()` below. Everything else in this file is plumbing
around it: printing, an approval gate, and a REPL.

Two things are worth noticing while reading:

  * The API is stateless. `messages` is the entire memory of the agent. If it
    is not in that list, the model does not know it. "Context management" is
    just deciding what goes in this list and what gets dropped.

  * The model never runs anything. It emits a `tool_use` block — a name and
    some JSON — and this program decides whether to honour it. That decision
    point is where a harness earns its keep.
"""

import os
import sys

import anthropic

import tools

MODEL = "claude-opus-5"

# max_tokens caps thinking *and* the visible answer together. 16000 keeps
# non-streaming requests comfortably under the SDK's HTTP timeout. To go
# higher (and to raise effort to "xhigh"), switch to client.messages.stream().
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are a coding assistant working inside a single project directory.

Use your tools rather than guessing: list files before assuming what exists, and
read a file before editing it. Prefer small, verifiable steps over one large one.

Keep replies short. The user can see the tool calls scroll past, so do not narrate
what you are about to do or recap what already happened — say what you found or
what changed, and stop."""


def approve(name: str, tool_input: dict) -> bool:
    """Ask the user before running anything with side effects.

    This is the reason to write a harness at all. The model proposes; a human
    disposes. Note that a denial is not an error — it is fed back to the model
    as a normal tool result so it can try something else.
    """
    print(f"\n  ! {name} wants to run:")
    for key, value in tool_input.items():
        preview = str(value)
        if len(preview) > 500:
            preview = preview[:500] + f"... [{len(str(value))} chars total]"
        print(f"      {key}: {preview}")
    answer = input("  ! allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def run_turn(client: anthropic.Anthropic, messages: list) -> None:
    """Drive one user turn to completion, mutating `messages` as it goes."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools.TOOLS,
            messages=messages,
            # Adaptive thinking lets the model decide how much to reason per
            # step. "summarized" makes that reasoning visible below; the
            # default returns thinking blocks with empty text.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
        )

        # Safety classifiers can decline a request. That arrives as a normal
        # HTTP 200 with an empty or partial content list, so check this before
        # touching response.content or you get a confusing IndexError.
        if response.stop_reason == "refusal":
            print("\n[the model declined this request]")
            return

        # Append the response *whole*. Pulling out just the text and appending
        # that would silently drop the thinking and tool_use blocks, and the
        # next request would then be missing the tool calls it needs to match
        # results against.
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "thinking" and block.thinking:
                print(f"\n\033[2m[thinking] {block.thinking}\033[0m")
            elif block.type == "text":
                print(f"\n{block.text}")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # No tools requested: the model is done talking. Turn over.
            return

        # One tool_result per tool_use, all of them in a SINGLE user message.
        # Splitting them across several messages teaches the model to stop
        # requesting tools in parallel, and a missing result is a hard API error.
        results = []
        for block in tool_uses:
            print(f"\n  > {block.name}({block.input})")

            if block.name in tools.REQUIRES_APPROVAL and not approve(block.name, block.input):
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "The user denied this action. Do not retry it; ask them what to do instead.",
                    "is_error": True,
                })
                continue

            try:
                output = tools.dispatch(block.name, block.input)
                is_error = False
            except Exception as exc:
                # Errors go back to the model as results, not as crashes. It can
                # usually recover — a wrong path is information, not a dead end.
                output = f"{type(exc).__name__}: {exc}"
                is_error = True

            print(f"  < {output[:300]}{'...' if len(output) > 300 else ''}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,   # must match the tool_use it answers
                "content": output,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})
        # Loop back around: the model now sees its own tool calls and their
        # results, and decides what to do next.


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    messages: list = []

    print(f"miniagent — workspace: {tools.WORKSPACE}")
    print("Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(client, messages)
        except anthropic.APIError as exc:
            print(f"\n[API error] {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
