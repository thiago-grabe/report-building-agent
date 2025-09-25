"""Prompt templates for intent classification, chat system setup, and formatting.

Includes:
- Intent classification prompt (`get_intent_classification_prompt`)
- Chat prompt selector (`get_chat_prompt_template`)
- Memory summary and response formatting prompts

Example:
    prompt = get_intent_classification_prompt()
    text = prompt.format(user_input="Calculate total", conversation_history="")
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# TODO: Implement the intent classification prompt.
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

Guidance:
- qa: The user asks a specific question to be answered from documents.
- summarization: The user asks to summarize one or more documents.
- calculation: The user asks to compute, total, add, subtract, multiply, divide, or otherwise calculate values (often from invoices, contracts, or claims).
- unknown: Anything else or ambiguous.

Examples:
1) "What's the total amount in invoice INV-001?" -> qa (confidence ~0.8+)
2) "Summarize all contracts" -> summarization (confidence ~0.9)
3) "Calculate the sum of all invoice totals" -> calculation (confidence ~0.9)
4) "Tell me about our performance" -> unknown (confidence depends on context)

Conversation History:
{conversation_history}

User Input:
{user_input}

Return the fields above.
"""
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template=template,
    )

# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

Current conversation context:
{conversation_summary}
"""

# Summarization System Prompt  
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information

Current conversation context:
{conversation_summary}
"""

# Calculation System Prompt
CALCULATION_SYSTEM_PROMPT = """You are a precise calculator assistant for document-related computations.

Your responsibilities:
- Perform accurate calculations
- Show your work step-by-step
- Extract numbers from documents when needed
- Verify calculations

Guidelines:
1. Search documents for required numbers if not provided
2. Use the calculator tool for all computations
3. Explain each calculation step
4. Include units where applicable
5. Double-check results for accuracy

Current conversation context:
{conversation_summary}
"""

# Agent Decision Prompt
AGENT_DECISION_PROMPT = PromptTemplate(
    input_variables=["intent", "user_input", "available_tools"],
    template="""Based on the classified intent and user input, decide which tools to use and in what order.

Intent: {intent}
User Input: {user_input}
Available Tools: {available_tools}

Think step by step:
1. What information do I need to answer this request?
2. Which tools will help me get this information?
3. What order should I use the tools in?

Respond with your reasoning and tool usage plan.
"""
)

# Response Formatting Prompts
QA_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["question", "answer", "sources", "confidence"],
    template="""Format the Q&A response:

Question: {question}
Answer: {answer}
Sources: {sources}
Confidence: {confidence}

Provide a natural, conversational response that includes the answer and cites the sources.
"""
)

SUMMARY_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["documents", "key_points", "summary"],
    template="""Format the summarization response:

Documents Analyzed: {documents}
Key Points: {key_points}
Summary: {summary}

Create a well-structured summary that highlights the most important information.
"""
)

CALCULATION_RESPONSE_FORMAT = PromptTemplate(
    input_variables=["expression", "result", "explanation", "sources"],
    template="""Format the calculation response:

Calculation: {expression}
Result: {result}
Step-by-step: {explanation}
Data Sources: {sources}

Present the calculation clearly with all steps shown.
"""
)

# TODO: Implement the chat prompt template function.
# This function should return appropriate prompts based on the intent type.
# Refer to README.md Task 3.2 for detailed implementation requirements.
def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent
    """
    system_prompt = QA_SYSTEM_PROMPT
    if intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT
    else:
        system_prompt = QA_SYSTEM_PROMPT

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["conversation_history", "max_length"],
    template="""Summarize the following conversation history into a concise summary (max {max_length} words):

{conversation_history}

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions

Summary:"""
)
