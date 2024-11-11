from textual.app import App, ComposeResult
from textual_terminal import Terminal


class TerminalApp(App):
    def compose(self) -> ComposeResult:
        """Embed two terminal widgets running different commands"""
        yield Terminal(command="nvim", id="terminal_nvim")
        yield Terminal(command="bash", id="terminal_bash")

    def on_ready(self) -> None:
        """# Start the terminal widgets"""
        terminal_nvim: Terminal = self.query_one("#terminal_nvim")
        terminal_nvim.start()
        terminal_bash: Terminal = self.query_one("#terminal_bash")
        terminal_bash.start()


if __name__ == "__main__":
    app = TerminalApp()
    app.run()
