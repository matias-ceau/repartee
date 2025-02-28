"""
Command Line Interface for Repartee.

Provides a simple CLI for interacting with AI models through the Repartee
architecture, with support for memory and context-aware conversations.
"""

import os
import sys
import uuid
import argparse
from typing import Dict, List, Optional, Any

from rich.console import Console
from rich.markdown import Markdown

from ..config import get_api_key, ReparteeDefaults
from ..models.openai_models import OpenAIModel
from ..models.anthropic_models import AnthropicModel
from ..memory.short_term_memory import ShortTermMemory
from ..memory.episodic_memory import EpisodicMemory
from ..memory.semantic_memory import SemanticMemory


class CLI:
    """
    Command-line interface for Repartee.
    
    Provides a simple interface for conversing with AI models
    with support for memory and context awareness.
    """
    
    def __init__(self):
        # Initialize console for rich output
        self.console = Console()
        
        # Initialize memory systems
        self.short_term_memory = ShortTermMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Default model settings
        self.current_model = None
        self.model_name = ""
        self.conversation_id = str(uuid.uuid4())
        
    def _check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        keys = {
            "openai": bool(get_api_key("OPENAI_API_KEY")),
            "anthropic": bool(get_api_key("ANTHROPIC_API_KEY")),
            "google": bool(get_api_key("GOOGLE_API_KEY")),
            "perplexity": bool(get_api_key("PERPLEXITY_API_KEY")),
        }
        return keys
        
    def _initialize_model(self, provider: str, model_name: str = None) -> None:
        """Initialize model based on provider and optional model name."""
        available_keys = self._check_api_keys()
        
        if provider == "openai":
            if not available_keys["openai"]:
                self.console.print("[bold red]Error:[/bold red] OpenAI API key not found.")
                self.console.print("Set it with: [blue]export OPENAI_API_KEY=your-key[/blue]")
                sys.exit(1)
                
            model = model_name or ReparteeDefaults.models.openai
            self.current_model = OpenAIModel(model_name=model)
            self.model_name = model
            self.console.print(f"[green]Using OpenAI model:[/green] {model}")
            
        elif provider == "anthropic":
            if not available_keys["anthropic"]:
                self.console.print("[bold red]Error:[/bold red] Anthropic API key not found.")
                self.console.print("Set it with: [blue]export ANTHROPIC_API_KEY=your-key[/blue]")
                sys.exit(1)
                
            model = model_name or ReparteeDefaults.models.claude
            self.current_model = AnthropicModel(model_name=model)
            self.model_name = model
            self.console.print(f"[green]Using Anthropic model:[/green] {model}")
            
        else:
            self.console.print(f"[bold red]Error:[/bold red] Provider '{provider}' not supported yet.")
            available = [k for k, v in available_keys.items() if v]
            self.console.print(f"Available providers: {', '.join(available)}")
            sys.exit(1)
            
    def _format_system_prompt(self) -> str:
        """Format a system prompt with relevant context."""
        # Start with default system prompt
        system_prompt = ReparteeDefaults.system_prompt
        
        # Add relevant knowledge from semantic memory
        # This is a placeholder for more sophisticated context retrieval
        system_prompt += "\n\nImportant information about the user:"
        
        # TODO: Add more sophisticated context retrieval from semantic memory
        
        return system_prompt
        
    def _get_conversation_context(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for the current conversation.
        
        Args:
            prompt: The current user prompt
            
        Returns:
            List of messages to include in the context
        """
        # Get messages from short-term memory (current conversation)
        messages = self.short_term_memory.get_messages_for_model()
        
        # Search episodic memory for relevant past conversations
        similar_messages = self.episodic_memory.search_similar(prompt, limit=3)
        
        if similar_messages:
            # Add a note about relevant past conversations
            context_note = {
                "role": "system",
                "content": "The following historical conversation is relevant to the current query:"
            }
            messages.insert(1, context_note)  # Insert after initial system message
            
            # Add the most relevant historical message
            best_match = similar_messages[0]
            historical_msg = {
                "role": "system",
                "content": f"Previous relevant exchange: {best_match['content']}"
            }
            messages.insert(2, historical_msg)
        
        return messages
        
    def send_prompt(self, prompt: str) -> str:
        """
        Send a prompt to the current model and get a response.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            The model's response
        """
        if not self.current_model:
            self.console.print("[bold red]Error:[/bold red] No model initialized.")
            return ""
            
        # Add user message to short-term memory
        self.short_term_memory.add_user_message(prompt)
        
        # Get conversation context
        system_prompt = self._format_system_prompt()
        conversation_history = self._get_conversation_context(prompt)
        
        # Send to model
        self.console.print("\n[Assistant]:", style="bold blue")
        
        response = self.current_model.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )
        
        # Add response to memories
        self.short_term_memory.add_assistant_message(response)
        self.episodic_memory.add_message(
            role="assistant",
            content=response,
            conversation_id=self.conversation_id
        )
        
        return response
        
    def run(self, args: Optional[List[str]] = None) -> None:
        """
        Run the CLI application.
        
        Args:
            args: Command line arguments (if None, uses sys.argv)
        """
        parser = argparse.ArgumentParser(description="Repartee: A conversational AI assistant")
        parser.add_argument("-m", "--model", help="Model to use (e.g., gpt-4, claude-3.5-sonnet)")
        parser.add_argument("-p", "--provider", default="openai", help="Provider (openai, anthropic)")
        parser.add_argument("--import-obsidian", help="Import Obsidian vault from directory")
        parser.add_argument("--list-conversations", action="store_true", help="List recent conversations")
        parser.add_argument("prompt", nargs="*", help="Prompt for one-shot query")
        
        parsed_args = parser.parse_args(args)
        
        # Handle special commands
        if parsed_args.import_obsidian:
            try:
                concepts, relations = self.semantic_memory.import_from_obsidian(parsed_args.import_obsidian)
                self.console.print(f"[green]Imported {concepts} concepts and {relations} relations from Obsidian vault[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error importing Obsidian vault:[/bold red] {e}")
            return
            
        if parsed_args.list_conversations:
            conversations = self.episodic_memory.list_conversations()
            self.console.print("[bold]Recent conversations:[/bold]")
            for conv in conversations:
                self.console.print(f"[bold]{conv['title']}[/bold] ({conv['conversation_id']})")
                self.console.print(f"  Messages: {conv['message_count']}")
                self.console.print(f"  Date: {conv['timestamp']}")
            return
            
        # Initialize the model
        self._initialize_model(parsed_args.provider, parsed_args.model)
        
        # Add system message to short-term memory
        system_prompt = self._format_system_prompt()
        self.short_term_memory.add_system_message(system_prompt)
        
        # One-shot mode vs interactive mode
        if parsed_args.prompt:
            # One-shot query mode
            prompt = " ".join(parsed_args.prompt)
            self.console.print(f"[User]: {prompt}", style="bold green")
            
            # Track in episodic memory
            self.episodic_memory.add_message(
                role="user",
                content=prompt,
                conversation_id=self.conversation_id
            )
            
            response = self.send_prompt(prompt)
            
        else:
            # Interactive mode
            self.console.print("[bold]Repartee Interactive Mode[/bold]")
            self.console.print(f"Using model: [bold]{self.model_name}[/bold]")
            self.console.print("Type 'exit', 'quit', or Ctrl+D to exit")
            
            while True:
                try:
                    prompt = input("\n[User]: ")
                    
                    if prompt.lower() in ["exit", "quit"]:
                        break
                        
                    # Track in episodic memory
                    self.episodic_memory.add_message(
                        role="user",
                        content=prompt,
                        conversation_id=self.conversation_id
                    )
                    
                    response = self.send_prompt(prompt)
                    
                except (KeyboardInterrupt, EOFError):
                    self.console.print("\nExiting...")
                    break
                    
        return


def main():
    """Entry point for the CLI application."""
    cli = CLI()
    cli.run()
    
    
if __name__ == "__main__":
    main()
