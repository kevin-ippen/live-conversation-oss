# agent_with_tools

Demonstrates an OpenAI GPT-4o agent that calls tools during a voice turn
and pushes custom SSE events to a split-pane browser UI in real time.

## What it shows

```
You: "What's the weather in Tokyo?"

agent events panel:
  THINKING   Deciding what to do...
  TOOL CALL  get_weather — city: Tokyo
  RESULT     Tokyo: Humid and sunny, 81°F
  REPLY      The weather in Tokyo is 81°F and humid...

Conversation thread:
  You said:  What's the weather in Tokyo?
  Assistant: The weather in Tokyo is 81°F and humid...
```

The event panel updates live as the agent reasons — not after the fact.

## Tools

| Tool | What it does |
|------|-------------|
| `get_weather` | Returns fake weather for major cities |
| `lookup_fact` | Returns a canned fact about a topic |
| `calculate`   | Evaluates arithmetic expressions |

All three are self-contained in `tools.py` — no external API calls needed.
Swap them for real implementations without touching `main.py`.

## Try asking

- "What's the weather in London?"
- "What can you tell me about Python?"
- "What is 137 times 42?"
- "What's 15 percent of 340?"
- "Look up a fact about voice."

## Custom events

The handler pushes these SSE events via `TurnResult.events`:

| Event | Data | Meaning |
|-------|------|---------|
| `thinking` | `{text}` | Agent decided to reason |
| `tool_call` | `{name, args}` | Tool about to be called |
| `tool_result` | `{name, result}` | Tool returned a value |
| `final` | `{text}` | Final spoken reply |
| `agent_ready` | `{}` | Session started |

These are received by the browser in `canvas.html` via the SSE `/events` stream
and rendered in the right-hand panel.

## Setup

```bash
export OPENAI_API_KEY=sk-...
cd examples/agent_with_tools
python main.py
```

Open http://localhost:8765, click **Start**, and speak.

## Swapping in real tools

Edit `tools.py` — the `dispatch()` function maps tool names to `async` coroutines.
Add to the `TOOLS` list (OpenAI tool schema) and implement the coroutine.
`main.py` does not need to change.
