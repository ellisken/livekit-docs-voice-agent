# Docs Buddy: a voice agent that reads the LiveKit docs for you

Docs Buddy is a small voice agent built with [LiveKit Agents](https://github.com/livekit/agents). Ask it a LiveKit question out loud while you code. It looks up the answer in LiveKit's own documentation through the [LiveKit Docs MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp.md), then tells you in two or three sentences.

This README is a tutorial. It starts from the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md), adds one feature, and notes the rough edges I hit along the way.

**Why this agent?** A voice agent's weakest point is answering from stale training data. Pointing it at the docs over MCP fixes that, and it raises the design questions a real deployment raises too: which tools the agent should be allowed to call, and what a tool result should look like before an LLM turns it into speech.

## What you'll build

```
you (mic) ──> STT ──> LLM ──> TTS ──> you (speaker)
                       │
                       └──> LiveKit Docs MCP server
                            docs_search · get_pages · get_changelog
```

- The voice pipeline from the quickstart: AssemblyAI speech-to-text, Gemma 4 31B as the LLM, and Fish Audio text-to-speech, all through [LiveKit Inference](https://docs.livekit.io/agents/models/inference.md), so you only need LiveKit credentials. It also uses the LiveKit turn detector and ai-coustics noise cancellation.
- A read-only connection to the Docs MCP server.
- A custom tool-result resolver that turns search results into compact plain text.
- Two tests that check the docs integration without needing LiveKit credentials.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs a supported Python for you. I started from macOS's built-in Python 3.9, which is too old (the SDK needs 3.10+), and uv handled it.
- A free [LiveKit Cloud](https://cloud.livekit.io/) project, for the `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` that LiveKit Inference uses.
- Optional: the [LiveKit CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli.md) (`lk`). The quickstart assumes it. The steps below show how to get by without it.

## Step 1: Start from the quickstart

The quickstart's path is `lk cloud auth`, then `lk agent init my-agent --template agent-starter-python`, then `lk agent dev`. If you have Homebrew, that path is the smoothest.

I didn't have Homebrew, so I cloned the starter directly:

```bash
git clone https://github.com/livekit-examples/agent-starter-python
cd agent-starter-python
uv sync
```

`uv sync` installed `livekit-agents` 1.8.13 on Python 3.14 in about a minute.

> **Rough edge:** the quickstart has no path for readers without the CLI. The starter
> repo works fine on its own, but you have to notice the link in the middle of a
> paragraph. Also, the CLI's "From Source" instructions say
> `git clone github.com/livekit/livekit-cli`, without `https://`. git treats that as a
> local path and fails, and the next line (`make install`) also needs a
> `cd livekit-cli` first. My [docs code tester](https://github.com/ellisken/docs-code-tester) found this
> automatically. It appears on three pages.

Look at `src/agent.py`. Three ideas matter:

- An **`Agent`** holds the instructions (the system prompt) and tools.
- An **`AgentSession`** wires up the voice pipeline: `stt`, `llm`, `tts`, and turn handling.
- An **`AgentServer`** registers the entrypoint (`@server.rtc_session`). LiveKit dispatches a job to it whenever a room needs this agent.

> **Rough edge:** the quickstart's appendix says "here's a complete working version of
> the quickstart agent" and that the starter "includes this agent". But the starter's
> `agent.py` is noticeably different. It sets the LLM on the `Agent` instead of the
> session, adds `STTContextOptions` keyterms, `preemptive_generation`, and
> `expressive=True`, and calls `await ctx.connect()`. The Python appendix omits that call
> even though the TypeScript version makes it. None of this is wrong, but a new reader
> comparing the two will wonder which one to trust.

## Step 2: Add the docs as tools

LiveKit Agents has first-class MCP support. Install the extra. Note the quotes:

```bash
uv add 'livekit-agents[mcp]~=1.8'
```

> **Rough edge:** the MCP page shows `uv add livekit-agents[mcp]~=1.8` without quotes.
> In zsh, the default shell on macOS, that fails with
> `zsh: no matches found: livekit-agents[mcp]~=1.8` because the square brackets are a
> glob pattern.

Before writing any agent code, I connected to the server with the SDK's own MCP client
to see what it offers:

```python
import asyncio
from livekit.agents import mcp

async def main():
    server = mcp.MCPServerHTTP("https://docs.livekit.io/mcp")
    await server.initialize()
    for tool in await server.list_tools():
        print(tool.info.name)
    await server.aclose()

asyncio.run(main())
```

It lists nine tools: `get_docs_overview`, `get_sdks`, `get_pages`, `docs_search`, `get_changelog`, `get_python_agent_example`, `get_pricing_info`, `submit_docs_feedback`, and `code_search`.

That raises two design decisions.

**1. Don't hand a voice agent every tool.** `submit_docs_feedback` *writes* to LiveKit. A voice agent shouldn't file feedback because it misheard something. More tools also means a longer prompt and slower, less reliable tool selection, which you feel in a voice conversation. So I allow only the three read-only tools a spoken Q&A needs, using [server-level filtering](https://docs.livekit.io/agents/logic/tools/mcp.md#filter-server):

```python
READ_ONLY_TOOLS = ["docs_search", "get_pages", "get_changelog"]

mcp.MCPServerHTTP(DOCS_MCP_URL, allowed_tools=READ_ONLY_TOOLS, ...)
```

**2. Shape tool results for an LLM that will speak.** I called `docs_search` directly to see what comes back. It took 0.6 seconds to return about 9,600 characters of Markdown, with every matched word wrapped in `**bold**` for highlighting. The SDK's default resolver then serializes that content item to JSON, so the LLM receives escaped Markdown inside a JSON envelope. For a voice agent I'd rather send clean, bounded plain text. That's what [`tool_result_resolver`](https://docs.livekit.io/agents/logic/tools/mcp.md) is for:

```python
MAX_TOOL_RESULT_CHARS = 6000

def _clean_tool_result(ctx: mcp.MCPToolResultContext) -> str:
    text = "\n".join(getattr(item, "text", "") for item in ctx.result.content).strip()
    text = text.replace("**", "")              # drop search highlighting
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
    return text or f"{ctx.tool_name} returned no results."
```

Then wrap the server in an `MCPToolset` and give it to the agent:

```python
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
```

> The docs mark the older `mcp_servers=` parameter as deprecated, because it generates
> toolset IDs that change between sessions. Use an `MCPToolset` with an explicit `id`.

## Step 3: Write instructions for the ear, not the eye

The instructions do the rest of the voice-specific work. See `INSTRUCTIONS` in [`agent.py`](agent.py). The key rules:

- Search before answering any LiveKit question, and say so when the docs don't cover something. Never invent API names.
- Two or three sentences, then offer to go deeper. The user is listening, not skimming.
- Never read code or URLs aloud. Say a class name once, naturally, and name the docs page ("the Turns overview page under Build Agents") so the user can find it.

The full agent is [`agent.py`](agent.py). The session setup is the quickstart's, unchanged.

## Step 4: Test the parts you own, without credentials

A voice session needs LiveKit credentials, but the docs integration doesn't. [`tests/test_docs_tools.py`](tests/test_docs_tools.py) sets up the real toolset against the public MCP server and checks two things:

1. Only the three read-only tools are exposed, and `submit_docs_feedback` isn't.
2. A real `docs_search` result reaches the LLM as plain text: no `**` markup, not a JSON dump, within the size cap, and still containing docs URLs the agent can cite.

```bash
uv sync
uv run pytest
```

```
2 passed in 4.13s
```

To evaluate what the agent says (does it search first, does it keep answers short), the next step is LiveKit's [testing framework](https://docs.livekit.io/testing.md) with an LLM judge, or [simulations](https://docs.livekit.io/testing/simulations.md). Both need LiveKit Inference credentials.

## Step 5: Talk to it

Create `.env.local` with your LiveKit Cloud project's values (copy `.env.example`):

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<key>
LIVEKIT_API_SECRET=<secret>
```

With the CLI, `lk cloud auth` and then `lk app env -w` writes this file for you. Then run:

```bash
lk agent dev        # then open the Agent Console link it prints
```

Without the CLI:

```bash
uv run python agent.py dev
```

Then open the Agent Console from your project's dashboard. Try "How do I add noise cancellation to my agent?" or "What changed in the latest Python agents release?"

> **Rough edge:** `python agent.py console` still works, but prints a deprecation notice
> pointing to `lk agent console`. That makes the CLI a hard requirement for talking to
> an agent in the terminal. Also, when I ran it before setting credentials, the job
> failed immediately with
> `ValueError: api_key is required, either as argument or set LIVEKIT_API_KEY`, but the
> worker kept running until I killed it. A check at startup, with a hint about
> `.env.local`, would save new users some confusion.

## What I verified, and what I didn't

Verified on November 24, 2026, with `livekit-agents` 1.8.13 on Python 3.14 (macOS):

- Dependencies install with `uv sync`, `agent.py` imports cleanly, and the CLI loads.
- The Docs MCP server's tool list, the `allowed_tools` filter, and the resolver's output, against the live server (`pytest`).

Not verified: a full voice conversation. That needs LiveKit Cloud credentials, which I didn't use for this write-up. The session configuration is copied from the quickstart.

## Ideas for next steps

- **Say it, then show it.** Send the docs URL to the frontend as a text stream while the agent says the page name, so links are clickable instead of read aloud.
- **Tune the result cap by measurement.** Use the Agent Console's per-turn model timing to compare 3k, 6k, and 10k character caps for latency versus answer quality.
- **Version awareness.** Read the user's installed `livekit-agents` version and tell the agent, so `get_changelog` answers can say "that's newer than your version".

## Files

```
agent.py                  the agent
tests/test_docs_tools.py  credential-free tests for the docs toolset
pyproject.toml            dependencies (livekit-agents[mcp], ai-coustics plugin)
.env.example              the variables .env.local needs
```

The voice pipeline configuration is adapted from LiveKit's MIT-licensed [agent-starter-python](https://github.com/livekit-examples/agent-starter-python).
