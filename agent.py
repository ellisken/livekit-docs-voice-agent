"""Docs Buddy: a voice agent that answers LiveKit questions from LiveKit's own docs.

The voice pipeline is the one from the LiveKit voice AI quickstart. The addition is a
read-only connection to the LiveKit Docs MCP server, so answers come from the current
docs instead of the model's training data.
"""

import logging
import re

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    mcp,
    room_io,
)
from livekit.plugins import ai_coustics

logger = logging.getLogger("docs-buddy")

load_dotenv(".env.local")

DOCS_MCP_URL = "https://docs.livekit.io/mcp"

# The Docs MCP server exposes nine tools. Only expose the read-only ones:
# `submit_docs_feedback` writes to LiveKit, and a voice agent shouldn't do that on its
# own initiative. Fewer tools also means a shorter prompt and faster tool choice.
READ_ONLY_TOOLS = ["docs_search", "get_pages", "get_changelog"]

# Search results are ~10k characters of Markdown with every hit wrapped in **bold**.
# Cap what goes to the LLM so a single lookup can't balloon the context.
MAX_TOOL_RESULT_CHARS = 6000

INSTRUCTIONS = """\
You are Docs Buddy, a voice guide to the LiveKit documentation. Developers talk to you
while they build, often with their hands on the keyboard.

How to answer:
- For any question about LiveKit, look it up with docs_search first, then answer from
  what you found. Use get_pages when a search hit looks right but the snippet is thin.
  Use get_changelog for "what's new" or version questions.
- If the docs don't cover it, say so plainly. Never invent API names or parameters.
- Speak in plain sentences. No markdown, lists, emojis, or code.
- Keep it to two or three sentences, then offer to go deeper.
- Never read code or URLs aloud character by character. Say the class or function name
  once, naturally, and name the docs page so they can find it, for example "the Turns
  overview page under Build Agents".
"""


def _clean_tool_result(ctx: mcp.MCPToolResultContext) -> str:
    """Turn an MCP result into compact plain text for the LLM.

    The framework's default resolver returns the content item as escaped JSON. For a
    voice agent we want plain text, without search-highlight markup, and bounded in size.
    """
    text = "\n".join(getattr(item, "text", "") for item in ctx.result.content).strip()
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
    return text or f"{ctx.tool_name} returned no results."


def docs_toolset() -> mcp.MCPToolset:
    return mcp.MCPToolset(
        id="livekit-docs",
        mcp_server=mcp.MCPServerHTTP(
            DOCS_MCP_URL,
            allowed_tools=READ_ONLY_TOOLS,
            timeout=10,
            tool_result_resolver=_clean_tool_result,
        ),
    )


class DocsBuddy(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS, tools=[docs_toolset()])


server = AgentServer()


@server.rtc_session(agent_name="docs-buddy")
async def docs_buddy(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inference.STT(model="assemblyai/assemblyai-stt-3-7-realtime-agent"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice="fa4c9eb3dccc4806b382b40d61c6b10a",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=DocsBuddy(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user in one sentence and ask what they're building."
    )


if __name__ == "__main__":
    cli.run_app(server)
