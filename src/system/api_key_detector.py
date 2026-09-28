"""
Genius AI — Universal API Key Auto-Detector & Environment Manager.

Detects API key providers automatically from key format / prefix:
  - Google AI Studio / Gemini: starts with 'AIzaSy'
  - OpenRouter: starts with 'sk-or-v1-' or OpenRouter format
  - Anthropic Claude: starts with 'sk-ant-'
  - Groq Cloud: starts with 'gsk_'
  - OpenAI: starts with 'sk-proj-' or 'sk-'
  - Together AI: starts with 'together-' or format
  - DeepSeek: starts with 'sk-' (with DeepSeek identifier) or deepseek config
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

PROVIDER_METADATA = {
    "gemini": {
        "name": "Google AI Studio (Gemini 2.0 Flash / Pro)",
        "env_var": "GEMINI_API_KEY",
        "alt_env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash-exp",
        "website": "https://aistudio.google.com/app/apikey",
    },
    "openrouter": {
        "name": "OpenRouter (Unified Claude 3.7 / Llama 3.3 / Gemini / DeepSeek)",
        "env_var": "OPENROUTER_API_KEY",
        "default_model": "google/gemini-2.0-flash-001",
        "website": "https://openrouter.ai/keys",
    },
    "claude": {
        "name": "Anthropic Claude (Claude 3.7 Sonnet with Extended Thinking)",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-7-sonnet-20250219",
        "website": "https://console.anthropic.com/settings/keys",
    },
    "groq": {
        "name": "Groq Cloud (Ultra-Fast Llama 3.3 70B & DeepSeek R1)",
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "website": "https://console.groq.com/keys",
    },
    "openai": {
        "name": "OpenAI (GPT-4o / GPT-4o-mini)",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "website": "https://platform.openai.com/api-keys",
    },
}


def detect_provider_from_key(key: str) -> Tuple[str, str]:
    """
    Analyzes an API key and returns (provider_id, confidence_reason).
    """
    clean_key = key.strip().strip("'\"")

    # 1. Google AI Studio / Gemini: starts with AIzaSy
    if clean_key.startswith("AIzaSy"):
        return "gemini", "Key starts with 'AIzaSy' (standard Google AI Studio format)"

    # 2. OpenRouter: starts with sk-or-v1-
    if clean_key.startswith("sk-or-v1-") or "openrouter" in clean_key.lower():
        return "openrouter", "Key matches OpenRouter 'sk-or-v1-' prefix"

    # 3. Anthropic Claude: starts with sk-ant-
    if clean_key.startswith("sk-ant-"):
        return "claude", "Key starts with 'sk-ant-' (Anthropic Claude API format)"

    # 4. Groq: starts with gsk_
    if clean_key.startswith("gsk_"):
        return "groq", "Key starts with 'gsk_' (Groq Cloud format)"

    # 5. OpenAI project key: starts with sk-proj-
    if clean_key.startswith("sk-proj-"):
        return "openai", "Key starts with 'sk-proj-' (OpenAI Project key format)"

    # 6. Generic OpenAI style key (sk-...)
    if clean_key.startswith("sk-"):
        # Could be OpenAI or DeepSeek or other OpenAI-compatible
        return "openai", "Key starts with 'sk-' (Standard OpenAI-compatible format)"

    # Fallback heuristic: check length and character distribution
    if len(clean_key) == 39 and clean_key.isalnum():
        return "gemini", "Key matches Google Cloud / AI Studio alphanumeric pattern"

    return "unknown", "Could not conclusively identify provider from prefix alone"


def get_current_configured_keys() -> Dict[str, str]:
    """Returns dictionary of currently configured provider keys from .env and environment."""
    configured = {}
    for prov, meta in PROVIDER_METADATA.items():
        val = os.getenv(meta["env_var"]) or os.getenv(meta.get("alt_env_var", ""))
        if val:
            # Mask key for privacy
            if len(val) > 8:
                masked = val[:4] + "..." + val[-4:]
            else:
                masked = "***"
            configured[prov] = masked
    return configured


def save_key_to_env(key: str, provider: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Saves the key to .env file and sets current environment variable.
    Returns (success, provider_id, message).
    """
    clean_key = key.strip().strip("'\"")
    if not clean_key:
        return False, "", "Empty key provided."

    detected_prov, reason = detect_provider_from_key(clean_key)
    target_prov = provider or (detected_prov if detected_prov != "unknown" else "gemini")

    meta = PROVIDER_METADATA.get(target_prov)
    if not meta:
        return False, "", f"Unknown provider: {target_prov}"

    env_var_name = meta["env_var"]
    os.environ[env_var_name] = clean_key
    os.environ["DEFAULT_MODEL_PROVIDER"] = target_prov

    # Read existing .env if present
    env_lines = []
    found_var = False
    found_default = False

    if ENV_FILE_PATH.exists():
        try:
            with open(ENV_FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith(f"{env_var_name}="):
                        env_lines.append(f"{env_var_name}={clean_key}\n")
                        found_var = True
                    elif stripped.startswith("DEFAULT_MODEL_PROVIDER="):
                        env_lines.append(f"DEFAULT_MODEL_PROVIDER={target_prov}\n")
                        found_default = True
                    else:
                        env_lines.append(line)
        except Exception as e:
            return False, target_prov, f"Failed to read .env: {e}"

    if not found_var:
        env_lines.append(f"{env_var_name}={clean_key}\n")
    if not found_default:
        env_lines.append(f"DEFAULT_MODEL_PROVIDER={target_prov}\n")

    try:
        with open(ENV_FILE_PATH, "w", encoding="utf-8") as f:
            f.writelines(env_lines)
    except Exception as e:
        return False, target_prov, f"Failed to write .env: {e}"

    display_name = meta["name"]
    return True, target_prov, f"Successfully saved {display_name} key to .env ({reason})"


def interactive_setup() -> Optional[str]:
    """
    Runs interactive terminal prompt for pasting and auto-detecting keys.
    Returns the selected provider or None.
    """
    # Load existing .env first
    try:
        import dotenv
        dotenv.load_dotenv(ENV_FILE_PATH)
    except ImportError:
        pass

    print("\n" + "=" * 62)
    print("      GENIUS AI - SMART API KEY AUTO-DETECTOR")
    print("=" * 62)

    current = get_current_configured_keys()
    if current:
        print("\nCurrently Active API Keys:")
        for prov, masked in current.items():
            name = PROVIDER_METADATA[prov]["name"]
            print(f"  * {name}: {masked}")
    else:
        print("\nNo API keys configured yet. Running in fallback local mode.")

    print("\nPaste any API key from:")
    print("  1. Google AI Studio (Gemini 2.0 Flash / Pro) - starts with 'AIzaSy...'")
    print("  2. OpenRouter (Unified API)                 - starts with 'sk-or-v1-...'")
    print("  3. Anthropic Claude                         - starts with 'sk-ant-...'")
    print("  4. Groq Cloud                               - starts with 'gsk_...'")
    print("  5. OpenAI (GPT-4o)                          - starts with 'sk-...'")
    print("\nPress ENTER without typing to keep current settings and launch.")
    print("-" * 62)

    try:
        user_input = input("Paste API Key here > ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return None

    if not user_input:
        # User pressed enter without pasting
        def_prov = os.getenv("DEFAULT_MODEL_PROVIDER")
        if def_prov and def_prov in PROVIDER_METADATA:
            print(f"\nLaunching with active provider: {PROVIDER_METADATA[def_prov]['name']}")
            return def_prov
        # Check if any key is set
        for p in ["gemini", "openrouter", "claude", "groq", "openai"]:
            if os.getenv(PROVIDER_METADATA[p]["env_var"]):
                print(f"\nLaunching with detected provider: {PROVIDER_METADATA[p]['name']}")
                return p
        print("\nLaunching with LOCAL provider.")
        return "local"

    # Analyze input
    detected, reason = detect_provider_from_key(user_input)

    if detected == "unknown":
        print(f"\nCould not automatically identify provider from key prefix.")
        print("Select the provider manually:")
        print("  1. Google AI Studio (Gemini)")
        print("  2. OpenRouter")
        print("  3. Anthropic Claude")
        print("  4. Groq")
        print("  5. OpenAI")
        choice = input("Select [1-5] (default 1): ").strip()
        prov_map = {"1": "gemini", "2": "openrouter", "3": "claude", "4": "groq", "5": "openai"}
        detected = prov_map.get(choice, "gemini")

    ok, final_prov, msg = save_key_to_env(user_input, detected)
    if ok:
        print(f"\n[OK] {msg}")
        print(f"[OK] Genius will now use: {PROVIDER_METADATA[final_prov]['name']}")
        return final_prov
    else:
        print(f"\n[ERROR] {msg}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--key":
        # Direct CLI key pass: python -m src.system.api_key_detector --key <key>
        if len(sys.argv) > 2:
            k = sys.argv[2]
            ok, prov, msg = save_key_to_env(k)
            print(f"{prov}|{msg}")
            sys.exit(0 if ok else 1)
        else:
            print("Missing key argument")
            sys.exit(1)

    selected_provider = interactive_setup()
    if selected_provider:
        # Exit with provider name on stdout for batch scripts to read
        print(f"PROVIDER:{selected_provider}")
