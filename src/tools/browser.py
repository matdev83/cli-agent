from __future__ import annotations

import json
from pathlib import Path # Not strictly needed for this initial impl, but good practice
from typing import Dict, Any, List, Optional

try:  # Optional Playwright import so tests don't require heavy dependency
    from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright
except Exception:  # pragma: no cover - dependency may not be installed
    sync_playwright = None  # type: ignore
    Page = BrowserContext = Playwright = object  # fallbacks for type hints

from .tool_protocol import Tool

# Store playwright instance at module level, or manage via agent_memory/class instance.
# For simplicity in a stateless tool call, re-initializing might be easier if state isn't persisted by agent.
# However, to maintain browser state (launched page), agent_memory is better.
# Let's assume agent_memory will hold 'playwright_sync_instance', 'playwright_browser_context', 'playwright_page'

class BrowserActionTool(Tool):
    """A tool to interact with a web browser using Playwright."""

    # Viewport dimensions as per Cline's prompt (can be made configurable later)
    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 1024

    @property
    def name(self) -> str:
        return "browser_action"

    @property
    def description(self) -> str:
        return (
            "Perform an action in a web browser. Actions include: "
            "launch (open URL), click (at x,y coordinate), type (text into focused element), "
            "scroll_down, scroll_up, close. "
            f"The browser viewport will be {self.VIEWPORT_WIDTH}x{self.VIEWPORT_HEIGHT}. "
            "Screenshots are returned after most actions."
        )

    @property
    def parameters_schema(self) -> Dict[str, str]:
        return {
            "action": "Action to perform (e.g., launch, click, type, scroll_down, scroll_up, close)",
            "url": "URL for the 'launch' action (optional)",
            "coordinate": "x,y coordinates for 'click' (optional)",
            "text": "Text for 'type' (optional)"
        }

    def _get_playwright_page(self, agent_tools_instance: Any) -> Optional[Page]:
        if agent_tools_instance and hasattr(agent_tools_instance, 'playwright_page'):
            return agent_tools_instance.playwright_page
        return None

    def _get_playwright_context(self, agent_tools_instance: Any) -> Optional[BrowserContext]:
        if agent_tools_instance and hasattr(agent_tools_instance, 'playwright_browser_context'):
            return agent_tools_instance.playwright_browser_context
        return None

    def _get_playwright_sync_instance(self, agent_tools_instance: Any) -> Optional[Playwright]:
        if agent_tools_instance and hasattr(agent_tools_instance, 'playwright_sync_instance'):
            return agent_tools_instance.playwright_sync_instance
        return None

    def _set_playwright_state(self, agent_tools_instance: Any, sync_instance: Optional[Playwright] = None, context: Optional[BrowserContext] = None, page: Optional[Page] = None):
        if agent_tools_instance:
            # Allow setting None to clear state
            agent_tools_instance.playwright_sync_instance = sync_instance
            agent_tools_instance.playwright_browser_context = context
            agent_tools_instance.playwright_page = page

    def execute(self, params: Dict[str, Any], agent_tools_instance: Any) -> str:
        """
        Executes the browser action.
        Expects 'action' in params, and other params based on the action.
        Manages Playwright browser instance and page via agent_tools_instance.
        Manages Playwright browser instance and page via agent_tools_instance.
        """
        action = params.get("action")
        if not action:
            return json.dumps({"status": "error", "message": "Missing required parameter 'action'."})

        # For screenshotting (stubbed for now)
        screenshot_data = "screenshot_data_not_implemented"

        try:
            if action == "launch":
                url = params.get("url")
                if not url:
                    return json.dumps({"status": "error", "message": "Missing 'url' for 'launch' action."})
                if sync_playwright is None:
                    return json.dumps({"status": "error", "message": "Playwright not installed"})

                # Close existing browser if any, before launching a new one
                existing_context = self._get_playwright_context(agent_tools_instance)
                if existing_context:
                    try:
                        existing_context.close()
                    except Exception: # Ignore errors on closing old context
                        pass
                existing_sync_instance = self._get_playwright_sync_instance(agent_tools_instance)
                if existing_sync_instance:
                    try:
                        existing_sync_instance.stop()
                    except Exception:
                        pass
                self._set_playwright_state(agent_tools_instance, None, None, None)

                print("[DEBUG] Attempting to start Playwright...")
                p_sync = sync_playwright().start()
                print(f"[DEBUG] p_sync type: {type(p_sync)}")
                if hasattr(p_sync, '_mock_name'): print(f"[DEBUG] p_sync mock name: {p_sync._mock_name}")
                elif hasattr(p_sync, 'name'): print(f"[DEBUG] p_sync name: {p_sync.name}")


                print(f"[DEBUG] p_sync.chromium type: {type(p_sync.chromium)}")
                if hasattr(p_sync.chromium, '_mock_name'): print(f"[DEBUG] p_sync.chromium mock name: {p_sync.chromium._mock_name}")
                elif hasattr(p_sync.chromium, 'name'): print(f"[DEBUG] p_sync.chromium name: {p_sync.chromium.name}")

                print(f"[DEBUG] Is p_sync.chromium.launch callable? {callable(p_sync.chromium.launch)}")

                browser = p_sync.chromium.launch(headless=True) # Consider headless option
                print("[DEBUG] Browser launched.")
                context = browser.new_context(
                    viewport={'width': self.VIEWPORT_WIDTH, 'height': self.VIEWPORT_HEIGHT}
                )
                page = context.new_page()
                page.goto(url)

                self._set_playwright_state(agent_tools_instance, p_sync, context, page)

                # Screenshot logic would go here. For now, just a message.
                return json.dumps({"status": "success", "message": f"Browser launched at {url}.", "screenshot": screenshot_data})

            elif action == "close":
                context = self._get_playwright_context(agent_tools_instance)
                sync_instance = self._get_playwright_sync_instance(agent_tools_instance)

                if context:
                    context.close()
                if sync_instance:
                    sync_instance.stop()

                self._set_playwright_state(agent_tools_instance, None, None, None)
                return json.dumps({"status": "success", "message": "Browser closed."})

            # Actions requiring an active page
            page = self._get_playwright_page(agent_tools_instance)
            if not page:
                return json.dumps({"status": "error", "message": "Browser not launched or page not available."})

            if action == "click":
                coordinate_str = params.get("coordinate")
                if not coordinate_str:
                    return json.dumps({"status": "error", "message": "Missing 'coordinate' for 'click' action."})
                try:
                    x_str, y_str = coordinate_str.split(',')
                    x, y = int(x_str), int(y_str)
                except ValueError:
                    return json.dumps({"status": "error", "message": "Invalid coordinate format. Expected 'x,y'."})

                page.mouse.click(x, y)
                # Screenshot logic
                return json.dumps({"status": "success", "message": f"Clicked at {x},{y}.", "screenshot": screenshot_data})

            elif action == "type":
                text_to_type = params.get("text")
                if text_to_type is None: # Allow empty string for typing
                    return json.dumps({"status": "error", "message": "Missing 'text' for 'type' action."})

                page.keyboard.type(text_to_type)
                # Screenshot logic
                return json.dumps({"status": "success", "message": f"Typed text: '{text_to_type}'.", "screenshot": screenshot_data})

            elif action == "scroll_down":
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                # Screenshot logic
                return json.dumps({"status": "success", "message": "Scrolled down.", "screenshot": screenshot_data})

            elif action == "scroll_up":
                page.evaluate("window.scrollBy(0, -window.innerHeight)")
                # Screenshot logic
                return json.dumps({"status": "success", "message": "Scrolled up.", "screenshot": screenshot_data})

            else:
                return json.dumps({"status": "error", "message": f"Unknown browser action: '{action}'."})

        except Exception as e:
            # Attempt to clean up playwright resources on error
            current_context = self._get_playwright_context(agent_tools_instance)
            if current_context:
                try:
                    current_context.close()
                except Exception:
                    pass # best effort
            current_sync_instance = self._get_playwright_sync_instance(agent_tools_instance)
            if current_sync_instance:
                try:
                    current_sync_instance.stop()
                except Exception:
                    pass
            self._set_playwright_state(agent_tools_instance, None, None, None)
            return json.dumps({"status": "error", "message": f"An error occurred: {str(e)}"})
