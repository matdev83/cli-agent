import os
import platform
from src.prompts.system import get_system_prompt


def test_placeholder_replacement(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(os.path, "expanduser", lambda x: "/home/test")
    prompt = get_system_prompt("/work")
    assert "/work" in prompt
    assert "Darwin" in prompt
    assert "/bin/zsh" in prompt
    assert "/home/test" in prompt


def test_browser_sections(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    prompt_no = get_system_prompt("/work", supports_browser_use=False)
    prompt_yes = get_system_prompt("/work", supports_browser_use=True, browser_settings={"viewport": {"width": 800, "height": 600}})
    phrase = "Puppeteer-controlled browser when you feel it is necessary"
    assert phrase not in prompt_no
    assert phrase in prompt_yes
    assert "800x600" in prompt_yes

