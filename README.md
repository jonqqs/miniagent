# miniagent

A minimal agent harness for the Claude API, written from scratch to understand
what an "agent" actually is. It is a small coding assistant that can read and
write files and run shell commands in one directory — with a human approval gate
in front of anything destructive.

The point of this repo is the loop, not the feature list. It is deliberately
small enough to read in one sitting.

## The idea in one paragraph

An agent is a `while` loop. You send a conversation to the model along with a
list of tools it may request. The model replies with either an answer or a
request to call a tool. If it asked for a tool, your program runs it, appends the
result to the conversation, and sends the whole thing again. Repeat until the
model stops asking. That is the entire mechanism — everything else (memory,
context management, sub-agents, permissions) is a decision about what goes into
that conversation and what you allow.

The loop is `run_turn()` in [`agent.py`](agent.py); it is about 60 lines.

## Setup

Requires an **Anthropic API key with credits**. Note that a Claude Pro or Max
subscription does *not* cover API usage — API access is billed separately, and
without credits every request fails with a 400 even on free endpoints.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Run it from the directory you want the agent to work in. That directory becomes
its workspace, and it cannot read or write outside it.

```bash
cd ~/some-project
python /path/to/miniagent/agent.py
```

```
> what does src/calc.py do, and is it correct?

  > list_files({})
  < notes.txt
    src/
  > read_file({'path': 'src/calc.py'})
  < def add(a, b):
        return a - b

`add` subtracts instead of adding — the body should be `a + b`.
```

Writes and shell commands stop and ask first:

```
  ! run_command wants to run:
      command: pytest -q
  ! allow? [y/N]
```

## What the code shows

Four things worth understanding, each marked with a comment where it happens:

**The API is stateless.** `messages` is the agent's entire memory. If something
is not in that list, the model does not know it. Everything sold as "context
management" is a policy about what to put in this list and what to drop.

**Append the response whole.** `messages.append({"role": "assistant", "content":
response.content})` — not just the text. Extracting the text and appending that
silently discards the thinking and `tool_use` blocks, and the next request then
has no tool calls for the results to attach to.

**One result per call, all in one message.** Every `tool_use` needs exactly one
matching `tool_result` carrying its `tool_use_id`. They go back in a single user
message; splitting them across several trains the model to stop calling tools in
parallel, and omitting one is a hard API error.

**Errors are results, not crashes.** A failed tool returns `is_error: True` with
the message. The model reads it and adapts — a wrong path is information. The
same channel carries a denied approval, which is why refusing an action does not
break the conversation.

## Things it deliberately does not do

Left out to keep the loop readable. Each is a reasonable next exercise:

- **Streaming.** Non-streaming with `max_tokens=16000` keeps requests under the
  SDK timeout. Switching to `client.messages.stream()` allows a much larger
  budget and `effort: "xhigh"`, which is the better setting for agentic work.
- **Refusal fallbacks.** The loop detects `stop_reason == "refusal"` and stops.
  The API can instead retry on another model server-side, via
  `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"],
  fallbacks="default")`.
- **Context management.** The conversation grows without bound until it hits the
  window. Real harnesses compact or clear old tool results.
- **Persistence.** Memory dies with the process. Writing findings to a file the
  agent can read back is the smallest useful version of memory.
- **Parallel tool execution.** Tools run one after another even when the model
  requests several at once.

## Layout

| File | What is in it |
| --- | --- |
| [`agent.py`](agent.py) | The loop, the approval gate, and the REPL |
| [`tools.py`](tools.py) | Tool schemas, their implementations, and the workspace sandbox |
