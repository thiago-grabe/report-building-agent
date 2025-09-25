"""Core package for the DocDacity Document Assistant.

Provides higher-level orchestration via `DocumentAssistant` and utilities for
retrieval, tools, prompts, and LangGraph workflow construction.

Quick start:
    from src.assistant import create_assistant
    assistant = create_assistant(api_key="your_key")
    session_id = assistant.start_session("demo_user")
    result = assistant.process_message("What's the total in invoice INV-001?")
    print(result["response"])  # Structured response dict
"""

# This makes src a proper Python package