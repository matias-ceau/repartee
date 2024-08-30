from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Static


class LLMResponse(Static):
    """A custom widget to display LLM responses."""


class ModeDisplay(Static):
    """A widget to display the current mode."""


class TerminalGUIApp(App):
    """A Textual app that simulates a terminal-like interface with LLM integration."""

    TITLE = "Repartee"
    CSS_PATH = "repartee.tcss"
    BINDINGS = [("ctrl+t", "toggle_mode", "Toggle Mode")]

    mode = reactive("Chat")

    def compose(self) -> ComposeResult:
        yield Header()
        yield ModeDisplay(id="mode_display")
        yield ScrollableContainer(id="chat_container")
        yield Input(placeholder="Enter your message...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self.update_mode_display()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.lower() == "exit":
            self.exit()
        elif self.mode == "Terminal" and event.value.strip():
            self.handle_terminal_command(event.value)
        else:
            self.handle_chat_input(event.value)
        event.input.value = ""

    def action_toggle_mode(self) -> None:
        self.mode = "Terminal" if self.mode == "Chat" else "Chat"
        self.update_mode_display()

    def update_mode_display(self) -> None:
        self.query_one("#mode_display").update(f"Mode: {self.mode}")

    def handle_chat_input(self, user_input: str) -> None:
        container = self.query_one("#chat_container")
        container.mount(Static(f"You: {user_input}"))
        response = self.send_to_llm(user_input)
        container.mount(LLMResponse(response))
        container.scroll_end(animate=False)

    def handle_terminal_command(self, command: str) -> None:
        container = self.query_one("#chat_container")
        container.mount(Static(f"$ {command}"))
        # Here you would actually execute the command and get the output
        output = f"Simulated output for: {command}"
        container.mount(Static(output))
        container.scroll_end(animate=False)

    def send_to_llm(self, user_input: str) -> str:
        # Replace this with your actual LLM app integration
        return f"LLM response to: {user_input}"


if __name__ == "__main__":
    app = TerminalGUIApp()
    app.run()
