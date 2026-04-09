import os

"""
Lightweight configuration defaults.

The original codebase imported `API_BASE` and `API_KEY` from this module, but
they were not defined. We keep these as optional overrides while still
supporting standard environment variables in `Raccoon/utils.py`.
"""

# For OpenAI-compatible APIs. Use "Default" to let the loader choose.
API_BASE = os.getenv("RACCOON_API_BASE", "Default")

# For OpenAI-compatible APIs. Use "Default" to load from provider-specific env vars.
API_KEY = os.getenv("RACCOON_API_KEY", "Default")
