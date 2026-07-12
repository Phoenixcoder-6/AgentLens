"""
verify_groq.py — Day 1 smoke test
Confirms the Groq API key is configured and a basic call works.
Run: python verify_groq.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_key_here":
        print("[FAIL] GROQ_API_KEY not set. Copy .env.example -> .env and fill in your key.")
        sys.exit(1)

    print(f"[OK]   GROQ_API_KEY found: {api_key[:8]}...{api_key[-4:]}")

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: AgentLens Day 1 OK"
                }
            ],
            max_tokens=20,
            temperature=0.0,
        )
        reply = response.choices[0].message.content.strip()
        print(f"[OK]   Groq API call succeeded. Response: '{reply}'")
        print("\nDay 1 Groq verification PASSED.")

    except ImportError:
        print("[FAIL] groq package not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Groq API call failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
