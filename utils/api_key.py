import os

# OpenAI API Key Configuration
# Set your OpenAI API key here or via environment variable
api_key = os.getenv("OPENAI_API_KEY", "")

# Alternative: Set the API key directly (not recommended for production)
# api_key = "your-openai-api-key-here"

# Validate API key
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    print("Or set it directly in this file (not recommended for production)")