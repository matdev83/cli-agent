# CLI Agent (MVP)

This project aims to build a command‑line software development assistant inspired by the Cline VSCode extension. The agent will run entirely in the terminal and communicate with a remote language model to execute development tasks such as running shell commands and editing files.

Key ideas:

- **Reuse Cline prompts** – Cline provides detailed system and command prompts describing the agent's tools and workflow. These prompts are ported to Python so that the CLI agent can leverage the same language and capabilities.
- **Lightweight orchestration** – Instead of relying on the CrewAI framework, the agent implements its own simple loop in Python to send messages to the model, parse tool instructions, and apply the requested actions.
- **Extensible tools** – Tools like `execute_command`, `read_file`, and `write_to_file` will be implemented in Python. Additional features such as MCP integration or browser actions can be added later.

The repository contains planning documents in `docs/` and reference code from the original Cline extension under `vendor/`. The `src/` directory will host the Python implementation.

This MVP is a work in progress. See `docs/MVP-TASKS.md` for the ordered list of development steps.
