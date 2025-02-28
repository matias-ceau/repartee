"""
Main entry point for Repartee.

Initializes the application and starts the conversational assistant.
"""

import sys
from typing import List, Optional

from .ui.cli import CLI


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for Repartee application.
    
    Args:
        args: Optional command line arguments (if None, uses sys.argv[1:])
    """
    if args is None:
        args = sys.argv[1:]
        
    cli = CLI()
    cli.run(args)
    
    
if __name__ == "__main__":
    main()
