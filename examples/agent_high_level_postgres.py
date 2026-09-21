"""agent_high_level_postgres.py — High-Level (create_agent) demonstration with GoatMiddleware.

Same integration pattern as agent_high_level.py, but backed by PostgreSQL for
persistent memory: a PostgresSaver checkpointer for conversation state
and a PostgresStore for long-term memory.

For the in-process version, see agent_high_level.py.
For low-level StateGraph integration, see agent_low_level.py.
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from langgraph_goat import GoatMiddleware, GoatMiddlewareConfig, GoatLogger

# Putting examples/ folder on sys.path
CURR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR_DIR))

# Import tools and world state
from res.toolsNworld import ALL_TOOLS
from customllm import CustomLLM

load_dotenv()

# LangGraph warns when deserializing custom GoAT types from checkpoints.
# They are warnings, not errors — quiet them so the chat loop stays readable.
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
logging.getLogger("langgraph").setLevel(logging.ERROR)

log = GoatLogger("middleware", level=logging.INFO)


# ── 1. Define Context Schema (for user identification) ────────────────────────

@dataclass
class Context:
    user_id: str


# ── 2. Embedding function for PostgreSQL Store semantic search ────────────────

# GoAT's middleware uses store.search() to retrieve memories for better plan
# generation. That requires the store to have an embedding index.
# This uses the Ollama /api/embed endpoint (same server as CustomLLM).

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Embedding dimensions (nomic-embed-text = 768; adjust for a different model)
EMBED_DIMS = int(os.getenv("OLLAMA_EMBED_DIMS", "768"))


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with Ollama, for the store's vector index.

    LangGraph hands the index callable a list of texts and expects one vector
    per text, so this must not swallow errors: returning fewer vectors than
    texts makes the store's INSERT fail with a placeholder/parameter mismatch.
    """
    resp = requests.post(
        f"{OLLAMA_ENDPOINT}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


# ── 3. Initialize Model ───────────────────────────────────────────────────────

model = CustomLLM()

# Output directory for all GoAT artifacts (graphs, telemetry, explanations)
OUTPUT_DIR = str(CURR_DIR / "output")


# ── 4. Save GoAT Logs Helper ──────────────────────────────────────────────────

def save_goat_logs(goat_output, user_input, ai_response):
    """Save response.json and graph.json into <OUTPUT_DIR>/<graph_id>/.

    Args:
        goat_output: The GoatOutput from the agent response.
        user_input: The user's input string.
        ai_response: The extracted AI response string.
    """
    goat_graph = goat_output.graph if goat_output else None
    goat_graph_id = goat_graph.graph_id if goat_graph else "unknown"

    print(f"[LOG] GoAT Graph ID: {goat_graph_id}")
    print(f"[LOG] Winning Plan ID: {goat_output.winning_plan_id if goat_output else None}")
    complexity = goat_output.complexity if goat_output else None
    print(f"[LOG] Complexity K: {complexity.suggested_K if complexity else 'N/A'}")

    output_dir = Path(OUTPUT_DIR) / goat_graph_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save response.json ────────────────────────────────────────────────────

    goat_response = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "goat_graph_id": goat_graph_id,
        **(goat_output.to_dict() if goat_output else {}),
    }

    response_file = output_dir / "response.json"
    with open(response_file, "w", encoding="utf-8") as f:
        json.dump(goat_response, f, indent=2, default=str, ensure_ascii=False)
    print(f"[LOG] Saved response to: {response_file}")

    # ── Save graph.json (from GoAT graph) ─────────────────────────────────────

    if goat_graph:
        goat_dict = goat_output.to_dict()
        graph_data = {
            **goat_dict.get("graph", {}),
            "plan_complexity": goat_dict.get("complexity"),
        }

        graph_file = output_dir / "graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"[LOG] Saved graph to: {graph_file}")


# ── 5. Main Chat Loop ─────────────────────────────────────────────────────────

def main():
    """Chat with the agent using PostgreSQL persistent memory."""
    # "postgres://<username>:<password>@<host>:<port>/<database>?<options>"
    conninfo = (
        f"postgres://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
        f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
        f"?sslmode={os.getenv('PSQL_SSLMODE')}"
    )

    with ConnectionPool(
        conninfo=conninfo,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        # Persistent chat memory (checkpointer for conversation state)
        memory = PostgresSaver(pool)
        memory.setup()

        # Persistent long-term memory store. The embedding index needs the
        # pgvector extension; without it GoAT falls back from semantic search
        # (search) to exact key lookups (get).
        try:
            store = PostgresStore(
                pool,
                index={"dims": EMBED_DIMS, "embed": _ollama_embed, "fields": ["text"]},
            )
            store.setup()
            print("[LOG] PostgreSQL store initialized with pgvector embedding index")
        except Exception as e:
            logging.getLogger(__name__).warning(
                "pgvector not available (%s) — falling back to a store without an "
                "embedding index.", e
            )
            store = PostgresStore(pool)
            store.setup()

        # Save a preference to Long-Term Memory. The "text" key is what the
        # embedding index reads, so store.search() can match on it.
        store.put(
            ("preferences",),
            "user_123",
            {"text": "User prefers detailed and scientific style", "style": "detailed and scientific"},
        )
        print("[LOG] Saved user preference to PostgreSQL store")

        # ── Configure GoatMiddleware ──────────────────────────────────────────

        goat_config = GoatMiddlewareConfig(
            default_K=3,
            max_K=3,
            store=store,  # Pass store so GoAT can read memory for better plan generation
            output_dir=OUTPUT_DIR,  # All GoAT outputs will be saved here
            render_goat_graph=True,
            render_goat_tele=True,
            render_goat_trace=True,
        )
        goat_middleware = GoatMiddleware(goat_config)

        # ── Create the Agent with GoAT middleware ─────────────────────────────

        agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            middleware=[goat_middleware],
            context_schema=Context,
            store=store,
            checkpointer=memory,
        )

        print("=" * 72)
        print("GoAT Middleware with PostgreSQL Persistent Memory")
        print("Type 'quit' to exit.")
        print("=" * 72)

        thread_config = {"configurable": {"thread_id": "1"}}

        while True:
            user_input = input("\nUser:\n")
            if user_input.lower() == "quit":
                break

            print("\nAgent:")
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                thread_config,
                context=Context(user_id="user_123"),
            )

            # Extract the AI response
            ai_response = ""
            for msg in reversed(response.get("messages", [])):
                if getattr(msg, "content", ""):
                    ai_response = msg.content
                    break
            print(ai_response)

            # Extract GoatOutput from response, or from the middleware instance
            goat_output = response.get("goat_output") or goat_middleware.last_goat_output
            save_goat_logs(goat_output, user_input, ai_response)

            # Show that the conversation state really is persisted
            latest = memory.get_tuple(thread_config)
            if latest:
                turns = len(latest.checkpoint["channel_values"].get("messages", []))
                print(f"[LOG] Checkpoint step {latest.metadata.get('step', 'N/A')}, "
                      f"{turns} message(s) in persisted state")


if __name__ == "__main__":
    main()
