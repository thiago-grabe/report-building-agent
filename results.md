# DocDacity Assistant - End-to-End Results

This document shows real, reproducible outputs from running the assistant locally via your environment. It includes the exact commands used, terminal transcript excerpts, and annotated references to the code paths responsible for the behavior.

## How we ran it

We executed a full interactive session non-interactively using your .venv-managed environment via `uv`. This feeds a sequence of prompts to the CLI and captures outputs for documentation.

```bash
# From the project root
printf "demo_user\nWhat's the total amount in invoice INV-001?\nSummarize all contracts\nCalculate the sum of all invoice totals\n/docs\n/history\n/logs\n/quit\n" | uv run main.py | cat
```

Notes:

- The CLI prompts for a user id; we pass `demo_user`.
- Then we send three tasks (QA → Summarization → Calculation), list documents, view history, export logs, and quit.

## Transcript excerpts (sanitized)

Below are verbatim slices of the output to illustrate the behavior.

### Q&A: Invoice total (INV-001)

```text
🤖 Assistant: The total amount in invoice INV-001 is $22,000. This includes a subtotal of $20,000 for services rendered, plus a 10% tax amounting to $2,000.

SOURCES: INV-001
TOOLS USED: document_reader
```

Why this output? The Q&A agent found the invoice and cited `INV-001`. The `document_reader` tool read the document contents to support the answer.

### Summarization: All contracts

```text
🤖 Assistant: The Service Agreement (Document ID: CON-001) is a contract between DocDacity Solutions Inc. and a client, effective from January 1, 2024. The agreement outlines the provision of specified services by DocDacity Solutions Inc. to the client for a total amount of $180,000. The contract includes terms regarding the scope of services, payment schedules, and responsibilities of both parties.

KEY POINTS:
  • The contract is a Service Agreement effective from January 1, 2024.
  • DocDacity Solutions Inc. is the service provider.
  • The total contract amount is $180,000.
  • The agreement specifies the scope of services and payment terms.
  • Responsibilities of both parties are outlined.

TOOLS USED: document_search
```

Why this output? The summarization agent searched for contracts and summarized the `CON-001` document, extracting key points and indicating the relevant document ID.

### Calculation: Sum of all invoice totals

```text
🤖 Assistant: To find the sum of all invoice totals, we add the amounts from each invoice:

1. Invoice #12345 has a total of $22,000.
2. Invoice #12346 has a total of $69,300.
3. Invoice #12347 has a total of $214,500.

Adding these amounts together gives:
$22,000 + $69,300 + $214,500 = $305,800.

RESULT: 305800.0
TOOLS USED: document_search
```

Why this output? The calculation agent identifies invoice amounts and composes a step-by-step explanation that adds to `305,800`. The CLI also prints the numeric `RESULT` for convenience.

### Documents listing

```text
AVAILABLE DOCUMENTS:
----------------------------------------
ID: INV-001
Title: Invoice #12345
Type: invoice
Total: $22,000.00
----------------------------------------
ID: CON-001
Title: Service Agreement
Type: contract
Value: $180,000.00
----------------------------------------
ID: CLM-001
Title: Insurance Claim #78901
Type: claim
Amount: $2,450.00
----------------------------------------
ID: INV-002
Title: Invoice #12346
Type: invoice
Total: $69,300.00
----------------------------------------
ID: INV-003
Title: Invoice #12347
Type: invoice
Total: $214,500.00
----------------------------------------
```

### Conversation history

```text
CONVERSATION HISTORY (3 turns):
============================================================

Turn 1 ...
Intent: qa
User: What's the total amount in invoice INV-001?
Assistant: The total amount in invoice INV-001 is $22,000. ...
Sources: INV-001
Tools used: document_reader
------------------------------------------------------------

Turn 2 ...
Intent: summarization
User: Summarize all contracts
Assistant: The Service Agreement (Document ID: CON-001) ...
Key Points:
  • The contract is a Service Agreement effective from January 1, 2024.
  • DocDacity Solutions Inc. is the service provider.
  • The total contract amount is $180,000.
  • The agreement specifies the scope of services and payment terms.
  • Responsibilities of both parties are outlined.
Tools used: document_search
------------------------------------------------------------

Turn 3 ...
Intent: calculation
User: Calculate the sum of all invoice totals
Assistant: To find the sum of all invoice totals, we add the amounts ...
Result: 305800.0
Tools used: document_search
------------------------------------------------------------
```

### Logs

An export is created under `logs/tool_usage_YYYYMMDD_HHMMSS.json`.

```text
LOGS EXPORTED TO logs/tool_usage_20250925_192814.json
```

## Code references

These code references show where key behaviors are implemented.

Intent classification prompt construction:

```19:53:src/prompts.py
# This prompt should help the LLM classify user intents into qa, summarization, calculation, or unknown.
# Refer to README.md Task 3.1 for detailed implementation requirements.
def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template
    """
    template = """You are an intent classification assistant for a document analysis system. 
Classify the user's intent into one of: qa, summarization, calculation, or unknown.

Provide:
- intent_type: one of [qa, summarization, calculation, unknown]
- confidence: a number between 0 and 1
- reasoning: a brief explanation for your decision
"""
    return PromptTemplate(
```

Calculator tool factory:

```72:110:src/tools.py
def create_calculator_tool(logger: ToolLogger):
    """
    Creates a calculator tool that safely evaluates basic mathematical expressions.
    Allowed: digits, spaces, parentheses, + - * / % **, decimal points.
    Returns a string result and logs all invocations.
    """
    # Strict validation: only numbers, operators, parentheses, and whitespace
    allowed_pattern = re.compile(r"^[0-9\s\+\-\*\/\%\(\)\.]+$")
    def _is_safe_expression(expr: str) -> bool:
        ...
```

Workflow creation:

```455:461:src/agent.py
def create_workflow(llm, tools):
    """
    Creates the LangGraph workflow
    """
    graph = StateGraph(AgentState)
```

## Reproducing locally

- Ensure your `.env` contains `OPENAI_API_KEY` and the virtual environment is active.
- Run the same command to produce a transcript for your session:

```bash
printf "demo_user\nWhat's the total amount in invoice INV-001?\nSummarize all contracts\nCalculate the sum of all invoice totals\n/docs\n/history\n/logs\n/quit\n" | uv run main.py | cat
```

## What to look for in the outputs

- Q&A should cite a specific document ID in `SOURCES`.
- Summarization should list several key points and identify document IDs.
- Calculation should show the steps and the final numeric `RESULT`.
- `/docs` should enumerate all sample documents with amounts/values.
- `/history` should show intents, tools used, and content for each turn.
- `/logs` should export a JSON file under `logs/`.

## Appendix: Sample data semantics

- Invoices include `Total Due` and are identified as `invoice` type; we parse totals through metadata and content.
- Contracts include a `Total Contract Value`; summarization collapses key terms.
- Claims list itemized medical expenses and a `Total Claim Amount`.

This completes the runbook-style results with real outputs and code anchors for quick auditing and verification.
