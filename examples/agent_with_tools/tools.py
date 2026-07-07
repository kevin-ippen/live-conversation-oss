"""Fake tools for the agent_with_tools example.

No external APIs — everything is deterministic and instant so the demo
works out of the box. Swap these for real implementations (web search,
database queries, file lookups) without changing main.py.
"""

import asyncio
import random


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "Look up a quick fact about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to look up"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '12 * 4 + 7'"},
                },
                "required": ["expression"],
            },
        },
    },
]


WEATHER_DATA = {
    "new york":    {"temp": "72°F", "condition": "Partly cloudy"},
    "london":      {"temp": "58°F", "condition": "Overcast"},
    "tokyo":       {"temp": "81°F", "condition": "Humid and sunny"},
    "san francisco": {"temp": "64°F", "condition": "Foggy morning"},
    "paris":       {"temp": "67°F", "condition": "Clear skies"},
}

FACTS = {
    "python":    "Python was created by Guido van Rossum and first released in 1991.",
    "databricks":"Databricks was founded in 2013 by the creators of Apache Spark.",
    "voice":     "The human voice can produce frequencies from about 80 Hz to 1100 Hz.",
    "AI":        "The term 'artificial intelligence' was coined by John McCarthy in 1956.",
    "websocket": "WebSockets provide full-duplex communication channels over a single TCP connection.",
}


async def get_weather(city: str) -> str:
    await asyncio.sleep(0.3)  # simulate latency
    key  = city.lower().strip()
    data = WEATHER_DATA.get(key)
    if data:
        return f"{city.title()}: {data['condition']}, {data['temp']}"
    return f"{city.title()}: 68°F, conditions unknown"


async def lookup_fact(topic: str) -> str:
    await asyncio.sleep(0.4)
    key  = topic.lower().strip()
    # fuzzy match
    for k, v in FACTS.items():
        if k in key or key in k:
            return v
    return f"No specific fact found for '{topic}', but it's a fascinating subject."


async def calculate(expression: str) -> str:
    await asyncio.sleep(0.1)
    try:
        # Only allow safe arithmetic — no builtins, no names
        allowed = set("0123456789 +-*/.()%")
        if not all(c in allowed for c in expression):
            return "Only basic arithmetic allowed."
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


async def dispatch(name: str, args: dict) -> str:
    if name == "get_weather":
        return await get_weather(args.get("city", "unknown"))
    elif name == "lookup_fact":
        return await lookup_fact(args.get("topic", "unknown"))
    elif name == "calculate":
        return await calculate(args.get("expression", ""))
    return f"Unknown tool: {name}"
