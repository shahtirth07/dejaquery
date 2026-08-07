"""Déjà Query application package."""

from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env before any submodule reads os.environ.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
