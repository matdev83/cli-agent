import shutil
from pathlib import Path
from typing import Dict, cast
from src.cli import run_agent

# Reconstruct APP_DIR as in the test file
# Assuming current working directory is cli-agent root
APP_DIR = Path(__file__).resolve().parent / "dev" / "app1"

# Create a temporary directory for the agent's work
tmp_path = Path("./temp_agent_work")
if tmp_path.exists():
    shutil.rmtree(tmp_path)
tmp_path.mkdir()

workdir = tmp_path / "app"
shutil.copytree(APP_DIR, workdir)

print("Running agent with OpenRouter LLM...")
result, history = run_agent(
    "List the files in this directory using the list_files tool and then finish.",
    responses_file=None,
    auto_approve=True,
    cwd=str(workdir),
    model="deepseek/deepseek-chat-v3-0324:free",
    return_history=True,
)

print("\n--- Agent History (Dialogue with LLM) ---")
for i, item in enumerate(history):
    msg = cast(Dict[str, str], item)
    print(f"--- Message {i+1} ({msg['role']}) ---")
    print(msg['content'])
    print("-" * 30)

print("\n--- Final Result ---")
print(result)

# Clean up temporary directory
shutil.rmtree(tmp_path)
