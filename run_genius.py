"""Root launcher for Genius Deep-Reasoning Agent."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chat_cli import main

if __name__ == "__main__":
    main()
