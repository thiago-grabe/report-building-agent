# Document Assistant Project Instructions

Welcome to the Document Assistant project! This project will help you build a sophisticated document processing system using LangChain and LangGraph. You'll create an AI assistant that can answer questions, summarize documents, and perform calculations on financial and healthcare documents.

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:

- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:

```bash
cd <repository_path>
```

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Assistant

```bash
python main.py
```

## Project Structure

```text
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic models
│   ├── retrieval.py      # Document retrieval
│   ├── tools.py          # Agent tools
│   ├── prompts.py        # Prompt templates
│   ├── agent.py          # LangGraph workflow
│   └── assistant.py      # Main agent
├── sessions/             # Saved conversation sessions
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Agent Architecture

The LangGraph agent follows this workflow:

![LangGraph Agent Architecture](./docs/langgraph_agent_architecture.png)

## Implementation Tasks

### 1. Schema Implementation (schemas.py)

#### Task 1.1: AnswerResponse Schema

Create a Pydantic model for structured Q&A responses with the following fields:

- `question`: The original user question (string)
- `answer`: The generated answer (string)
- `sources`: List of source document IDs used (list of strings)
- `confidence`: Confidence score between 0 and 1 (float)
- `timestamp`: When the response was generated (datetime)

**Purpose**: This schema ensures consistent formatting of answers and tracks which documents were referenced.

#### Task 1.2: UserIntent Schema

Create a Pydantic model for intent classification with these fields:

- `intent_type`: The classified intent ("qa", "summarization", "calculation", or "unknown")
- `confidence`: Confidence in classification (float between 0 and 1)
- `reasoning`: Explanation for the classification (string)

**Purpose**: This schema helps the system understand what type of request the user is making and route it to the appropriate agent.

### 2. Agent State Implementation (agent.py)

#### Task 2.1: AgentState Properties

The `AgentState` class is already defined, but you need to understand its structure:

- `messages`: Conversation messages with LangGraph message annotation
- `user_input`: Current user input
- `intent`: Classified user intent
- `next_step`: Next node to execute in the graph
- `conversation_history`: Previous conversation turns
- `conversation_summary`: Summary of recent conversation
- `active_documents`: Document IDs currently being discussed
- `current_response`: The response being built
- `tools_used`: List of tools used in current turn
- `session_id` and `user_id`: Session management

#### Task 2.2: Intent Classification Function

Implement the `classify_intent` function that:

1. Formats the conversation history for context
2. Uses the LLM with structured output to classify the user's intent
3. Sets the `next_step` based on the classified intent:
   - "qa" --> "qa_agent"
   - "summarization" --> "summarization_agent"
   - "calculation" --> "calculation_agent"
   - default --> "qa_agent"

**Key concepts**:

- Use `llm.with_structured_output(UserIntent)` for structured responses
- Include conversation history for better context understanding
- The function should return the updated state

#### Task 2.3: Calculation Agent Completion

Complete the `calculation_agent` function by implementing the final response generation:

1. Use `llm.with_structured_output(CalculationResponse)` to get a structured response
2. Create a prompt asking for a clear explanation of the calculation
3. Ensure the response includes the expression and step-by-step explanation
4. Update the state with the response and return it

#### Task 2.4: Memory Update Function

Implement the `update_memory` function that:

1. Creates a `ConversationTurn` object from the current interaction
2. Adds the turn to the conversation history
3. Updates the message history with user input and agent response
4. Tracks active documents from the response
5. Sets `next_step` to "end"

**Purpose**: This function maintains conversation context and tracks document references across turns.

#### Task 2.5: Workflow Creation

Implement the `create_workflow` function that:

1. Creates a `StateGraph` with the `AgentState`
2. Adds all agent nodes (classify_intent, qa_agent, summarization_agent, calculation_agent, update_memory)
3. Sets "classify_intent" as the entry point
4. Adds conditional edges from classify_intent to the appropriate agents
5. Adds edges from each agent to update_memory
6. Adds edge from update_memory to END
7. Returns the compiled workflow

**Graph Structure**:

```text
classify_intent --> [qa_agent|summarization_agent|calculation_agent] --> update_memory --> END
```

### 3. Prompt Implementation (prompts.py)

#### Task 3.1: Intent Classification Prompt

Implement the `get_intent_classification_prompt` function that returns a `PromptTemplate` with:

- Input variables: `["user_input", "conversation_history"]`
- Template that instructs the LLM to classify intent into qa, summarization, calculation, or unknown
- Clear examples and guidelines for each intent type
- Instructions to provide confidence score and reasoning

**Purpose**: This prompt helps the LLM accurately classify user intents for proper routing.

#### Task 3.2: Chat Prompt Template

Implement the `get_chat_prompt_template` function that:

1. Takes an `intent_type` parameter
2. Selects the appropriate system prompt based on intent type
3. Returns a `ChatPromptTemplate` with system message, chat history placeholder, and human message
4. Uses the existing system prompts (QA_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, CALCULATION_SYSTEM_PROMPT)

**Purpose**: This provides context-aware prompts for different types of tasks.

### 4. Tool Implementation (tools.py)

#### Task 4.1: Calculator Tool

Implement the `create_calculator_tool` function that:

1. Uses the `@tool` decorator to create a LangChain tool
2. Takes a mathematical expression as input
3. Validates the expression for safety (only allow basic math operations)
4. Evaluates the expression using Python's `eval()` function
5. Logs the tool usage with the ToolLogger
6. Returns a formatted result string
7. Handles errors gracefully

## Key Concepts for Success

### 1. LangChain Tool Pattern

Tools are functions decorated with `@tool` that can be called by LLMs. They must:

- Have clear docstrings describing their purpose and parameters
- Handle errors gracefully
- Return string results
- Log their usage for debugging

### 2. LangGraph State Management

The state flows through nodes and gets updated at each step. Key principles:

- Always return the updated state from node functions
- Use the state to pass information between nodes
- The state persists conversation context and intermediate results

### 3. Structured Output

Use `llm.with_structured_output(YourSchema)` to get reliable, typed responses from LLMs instead of parsing strings.

### 4. Conversation Memory

The system maintains conversation context by:

- Storing conversation turns with metadata
- Tracking active documents
- Summarizing long conversations
- Providing context to subsequent requests

## Testing Your Implementation

1. **Unit Testing**: Test individual functions with sample inputs
2. **Integration Testing**: Test the complete workflow with various user inputs
3. **Edge Cases**: Test error handling and edge cases

## Common Pitfalls to Avoid

1. **Missing Error Handling**: Always wrap external calls in try-catch blocks
2. **Incorrect State Updates**: Ensure you're updating and returning the state correctly
3. **Prompt Engineering**: Make sure your prompts are clear and specific
4. **Tool Security**: Validate all inputs to prevent security issues

## Expected Behavior

After implementation, your assistant should be able to:

- Classify user intents correctly
- Search and retrieve relevant documents
- Answer questions with proper source citations
- Generate comprehensive summaries
- Perform calculations on document data
- Maintain conversation context across turns

Good luck with your implementation! Remember to test thoroughly and refer to the existing working code for guidance on patterns and best practices.

## How This Implementation Meets the Rubric

- Schema Implementation:
  - `AnswerResponse` and `UserIntent` are fully defined with correct field types and constraints.
  - Confidence values are constrained to 0–1; `intent_type` is restricted to `qa|summarization|calculation|unknown`.
  - `logs/` auto-creates and stores per-session tool usage via `ToolLogger`; `sessions/` stores session state.

- Workflow Creation and Routing:
  - `create_workflow` builds a `StateGraph` with all nodes and conditional edges based on classified intent.
  - State flows through: `classify_intent` → `[qa_agent|summarization_agent|calculation_agent]` → `update_memory` → `END`.

- Tool Implementation:
  - Calculator tool uses `@tool`, validates expressions with a strict allowlist, evaluates safely via restricted `eval`, returns string results, and logs all calls.

- Prompt Engineering:
  - `get_intent_classification_prompt` provides categories, examples, and instructions for confidence and reasoning.
  - `get_chat_prompt_template` selects the appropriate system prompt based on intent type.

- Integration and Testing:
  - `main.py` provides an interactive CLI with `/help`, `/history`, `/docs`, `/logs`.
  - The assistant maintains conversation memory and document context; session data persists under `sessions/`.

## Memory and State Design

- `AgentState` carries `messages`, `user_input`, `intent`, `next_step`, `conversation_history`, `conversation_summary`, `active_documents`, `current_response`, `tools_used`, `session_id`, and `user_id`.
- `update_memory` extracts active document IDs from responses, appends the latest user/assistant messages, and routes to `END`.

## Example Conversations

- Calculation:
  - User: "Calculate the sum of all invoice totals"
  - Tools: `document_search` (amount range), `document_reader` (if needed), `calculator("22000 + 69300 + 214500")`
  - Assistant: returns explanation with result `305800` and cites `INV-001`, `INV-002`, `INV-003`.

- Q&A:
  - User: "What's the total amount in invoice INV-001?"
  - Tools: `document_reader("INV-001")`
  - Assistant: returns the amount `22000` with `INV-001` cited.

- Summarization:
  - User: "Summarize all contracts"
  - Tools: `document_search(search_type="type", doc_type="contract")`, `document_reader` on each result
  - Assistant: returns a concise summary with key points and document IDs.

## Running Locally

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Set your OpenAI API key:

```bash
cp .env.example .env
echo "OPENAI_API_KEY=YOUR_KEY" >> .env
```

1. Run the assistant:

```bash
python main.py
```

1. Try these:

```text
What's the total amount in invoice INV-001?
Summarize all contracts
Calculate the sum of all invoice totals
Find documents with amounts over $50,000
```

## Notes

- If you see dependency resolution warnings, ensure `langchain`, `langgraph`, and `langchain-openai` versions match those in `requirements.txt`.
- The CLI prevents missing API key usage; ensure `.env` contains `OPENAI_API_KEY`.

## CLI Usage

Once running `python main.py`, the interactive CLI supports commands:

- **/help**: Show available commands and example queries
- **/history**: Print the current session's conversation turns with intent, response, and tools used
- **/docs**: List the in-memory sample documents with IDs, types, and amounts
- **/logs**: Export the per-session tool usage logs to `logs/tool_usage_YYYYMMDD_HHMMSS.json`
- **/quit**: Exit the assistant

Enter free-form questions or requests at the prompt, e.g.:

```text
What's the total amount in invoice INV-001?
Summarize all contracts
Calculate the sum of all invoice totals
Find documents with amounts over $50,000
```

## Configuration

- **Environment**: Set `OPENAI_API_KEY` in `.env`.
- **Model**: The assistant uses `gpt-4o` via `langchain-openai`.
- **Base URL**: The OpenAI base URL is configured to `https://openai.vocareum.com/v1` in `src/assistant.py`. Adjust if using a different endpoint.
- **Sessions Path**: Session files are stored under `./sessions`. Change via `DocumentAssistant(session_storage_path=...)`.

## Schemas Reference

- **AnswerResponse**
  - **question**: string
  - **answer**: string
  - **sources**: list[string] (document IDs)
  - **confidence**: float in [0,1]
  - **timestamp**: datetime (auto)

- **UserIntent**
  - **intent_type**: one of `qa | summarization | calculation | unknown`
  - **confidence**: float in [0,1]
  - **reasoning**: string explanation

Additional structured responses:

- **SummarizationResponse**: `original_length`, `summary`, `key_points[]`, `document_ids[]`, `timestamp`
- **CalculationResponse**: `expression`, `result`, `explanation`, `units?`, `timestamp`

## Tools Reference

- **calculator(expression: str) -> str**
  - Validates a basic math expression and evaluates it in a restricted `eval` environment
  - Allowed: digits, whitespace, parentheses, `+ - * / % **`, decimals
  - Returns a string: "The result of the calculation is X."
  - All invocations are logged via `ToolLogger`

- **document_search(query: str, search_type: "keyword|type|amount|amount_range", doc_type?, min_amount?, max_amount?, comparison?, amount?) -> str**
  - Keyword, type, and amount-aware search across sample documents
  - Understands natural language like "over $50,000", "between $20,000 and $80,000"
  - Returns a formatted text list of matching documents

- **document_reader(doc_id: str) -> str**
  - Returns the full content of a specific document by ID with amount info when present

- **document_statistics() -> str**
  - Collection-level counts by type and financial aggregates (total, average, min, max)

## Sessions and Logs

- **Sessions**: Each session is saved as JSON under `sessions/<session_id>.json`, capturing conversation turns, document context, and timestamps. Sessions are auto-saved after each successful turn.
- **Logs**: Tool usage is captured per session under `logs/session_<session_id>.json`. When using `/logs`, an additional export is written as `logs/tool_usage_<timestamp>.json`.

## Sample Documents

Preloaded documents (IDs and notable amounts):

- **INV-001**: Invoice — Total Due $22,000
- **INV-002**: Invoice — Total Due $69,300
- **INV-003**: Invoice — Total Due $214,500
- **CON-001**: Contract — Total Contract Value $180,000
- **CLM-001**: Claim — Total Claim Amount $2,450

You can add more documents at runtime via `DocumentAssistant.add_document(...)`.

## Troubleshooting

- **No API key**: Ensure `.env` contains `OPENAI_API_KEY=...` before running.
- **Dependency issues**: Reinstall with `pip install -r requirements.txt`. Ensure versions align with this repo.
- **Empty tool results**: Try broader `document_search` queries or check sample documents with `/docs`.
- **File permissions**: Ensure the process can create and write to `./logs` and `./sessions`.
