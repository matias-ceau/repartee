# To modify the `TerminalGUIApp` to resemble a markdown viewer example, we can arrange the layout so that user prompts and LLM responses are displayed side-by-side. In the context given, we want to use Markdown widgets for formatting and separate the chat and response areas.
#
# Below is the updated code to achieve this layout:

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Markdown


class UserPrompt(Markdown):
    """Markdown for the user's input prompt."""


class LLMResponse(Markdown):
    """Markdown for displaying the LLM's response."""

    BORDER_TITLE = "LLM Response"


class ModeDisplay(Markdown):
    """A widget to display the current mode."""


class TerminalGUIApp(App):
    """A Textual app that simulates a terminal-like interface with LLM integration."""

    TITLE = "Repartee"
    CSS_PATH = "repartee.tcss"
    BINDINGS = [("ctrl+t", "toggle_mode", "Toggle Mode")]

    mode = reactive("Chat")

    CSS = """
    UserPrompt {
        background: $primary 10%;
        color: $text;
        margin: 1;        
        margin-right: 8;
        padding: 1 2 0 2;
    }

    LLMResponse {
        border: wide $success;
        background: $success 10%;
        color: $text;
        margin: 1;
        margin-left: 8; 
        padding: 1 2 0 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            with VerticalScroll(id="chat-container"):
                yield ModeDisplay(f"Mode: {self.mode}", id="mode_display")
                yield UserPrompt("Welcome! Type your message and press Enter.")
            with VerticalScroll(id="response-container"):
                yield LLMResponse("Awaiting response...")
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
        chat_container = self.query_one("#chat-container")
        chat_container.mount(UserPrompt(f"You: {user_input}"))

        response_container = self.query_one("#response-container")
        response = self.send_to_llm(user_input)
        response_container.mount(LLMResponse(response))
        chat_container.scroll_end(animate=False)
        response_container.scroll_end(animate=False)

    def handle_terminal_command(self, command: str) -> None:
        chat_container = self.query_one("#chat-container")
        chat_container.mount(UserPrompt(f"$ {command}"))

        response_container = self.query_one("#response-container")
        output = f"Simulated output for: {command}"
        response_container.mount(LLMResponse(output))
        chat_container.scroll_end(animate=False)
        response_container.scroll_end(animate=False)

    def send_to_llm(self, user_input: str) -> str:
        # Mock response for illustration
        return f"LLM response to: {user_input}"


if __name__ == "__main__":
    app = TerminalGUIApp()
    app.run()

# ### Key Modifications:
# - **Markdown Widgets**: `UserPrompt` and `LLMResponse` widgets use the `Markdown` class to benefit from more sophisticated formatting.
# - **Container and VerticalScroll**: Used a `Container` to host two `VerticalScroll` widgets, ensuring inputs and responses are separated.
# - **CSS for Styling**: Applied simple TCSS for background and layout styling to differentiate prompts and responses.
#
# This updated layout provides a clearer separation of user inputs and LLM responses in a side-by-side format, similar to a markdown viewer, while preserving terminal-like interaction.
