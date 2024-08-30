import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class TerminalGUI:
    def __init__(self):
        self.console = Console()
        self.session = PromptSession()
        self.is_terminal_mode = False

    async def main(self):
        while True:
            with patch_stdout():
                if self.is_terminal_mode:
                    user_input = await self.session.prompt_async("Terminal> ")
                else:
                    user_input = await self.session.prompt_async("Chat> ")

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "terminal":
                self.is_terminal_mode = not self.is_terminal_mode
                mode = "Terminal" if self.is_terminal_mode else "Chat"
                self.console.print(f"Switched to {mode} mode")
            else:
                if self.is_terminal_mode:
                    # Handle terminal commands here
                    self.console.print(f"Executing: {user_input}")
                else:
                    # Send input to LLM app and get response
                    response = self.send_to_llm(user_input)
                    self.display_response(response)

    def send_to_llm(self, user_input):
        # Replace this with your actual LLM app integration
        return f"LLM response to: {user_input}"

    def display_response(self, response):
        panel = Panel(
            Text(response, style="green"),
            title="LLM Response",
            border_style="blue",
        )
        self.console.print(panel)


if __name__ == "__main__":
    app = TerminalGUI()
    asyncio.run(app.main())
