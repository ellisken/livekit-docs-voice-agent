"""Tests for the docs toolset. They need network access but no LiveKit credentials.

They talk to the public LiveKit Docs MCP server, so they check the parts of the agent
that are ours (which tools are exposed, what the LLM receives) without starting a
voice session.
"""

import pytest

import agent


@pytest.fixture
async def tools():
    toolset = agent.docs_toolset()
    await toolset.setup()
    yield {t.info.name: t for t in toolset.tools}
    await toolset.aclose()


async def test_only_read_only_tools_are_exposed(tools):
    assert set(tools) == set(agent.READ_ONLY_TOOLS)
    assert "submit_docs_feedback" not in tools


async def test_search_result_is_clean_bounded_text(tools):
    result = await tools["docs_search"]({"query": "noise cancellation for agents"})

    assert isinstance(result, str)
    assert "**" not in result  # search highlighting stripped
    assert not result.startswith("{")  # plain text, not the default JSON dump
    assert len(result) <= agent.MAX_TOOL_RESULT_CHARS + len("\n[truncated]")
    assert "docs.livekit.io" in result  # still carries page links for the LLM to cite
