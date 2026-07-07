"""Tests for server-level push_event / poll_command / SSE queue mechanics."""

import asyncio
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import liveconv.server as server


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_server_state():
    """Isolate tests — clear module-level state between runs."""
    server._connections.clear()
    while not server._command_queue.empty():
        try:
            server._command_queue.get_nowait()
        except Exception:
            break


@pytest.fixture(autouse=True)
def clean_server():
    _reset_server_state()
    yield
    _reset_server_state()


# ── poll_command ──────────────────────────────────────────────────────────────

def test_poll_command_returns_none_when_empty():
    assert server.poll_command() is None


def test_poll_command_returns_queued_value():
    server._command_queue.put_nowait("start_session")
    assert server.poll_command() == "start_session"


def test_poll_command_drains_in_order():
    server._command_queue.put_nowait("start_session")
    server._command_queue.put_nowait("end_session")
    assert server.poll_command() == "start_session"
    assert server.poll_command() == "end_session"
    assert server.poll_command() is None


# ── push_event ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_event_delivers_to_connected_queue():
    q: asyncio.Queue = asyncio.Queue()
    server._connections.append(q)
    server._loop = asyncio.get_running_loop()

    server.push_event("orb", {"state": "listening"})
    await asyncio.sleep(0)  # allow call_soon_threadsafe to flush

    msg = q.get_nowait()
    assert msg == {"type": "orb", "data": {"state": "listening"}}


@pytest.mark.asyncio
async def test_push_event_broadcasts_to_multiple_connections():
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    server._connections.extend([q1, q2])
    server._loop = asyncio.get_running_loop()

    server.push_event("session_start", {})
    await asyncio.sleep(0)

    assert q1.get_nowait()["type"] == "session_start"
    assert q2.get_nowait()["type"] == "session_start"


@pytest.mark.asyncio
async def test_push_event_no_op_when_no_connections():
    server._loop = asyncio.get_running_loop()
    # Should not raise
    server.push_event("orb", {"state": "idle"})


@pytest.mark.asyncio
async def test_push_event_no_op_when_loop_is_none():
    server._loop = None
    q: asyncio.Queue = asyncio.Queue()
    server._connections.append(q)
    server.push_event("orb", {"state": "idle"})
    # Nothing delivered — queue still empty
    assert q.empty()


# ── has_active_connection ─────────────────────────────────────────────────────

def test_has_active_connection_false_when_empty():
    assert not server.has_active_connection()


def test_has_active_connection_true_when_connected():
    server._connections.append(asyncio.Queue())
    assert server.has_active_connection()


# ── register_voice_handler ────────────────────────────────────────────────────

def test_register_voice_handler_stores_callable():
    async def my_handler(pcm, cancel):
        return {}

    server.register_voice_handler(my_handler)
    assert server._voice_handler is my_handler
    server._voice_handler = None  # cleanup
