Rubric
Use this project rubric to understand and assess the project criteria.

Schema Implementation
Criteria	Submission Requirements
Implement structured output schemas using Pydantic.

The agent displays successful execution of code that defines both the AnswerResponse and 

UserIntent schemas with all required fields, proper field types, and appropriate default values.

Automatically generated logs directory showing history of tool calls for each session

Auto-generated sessions directory with history of individual sessions

Validate schema implementation for proper type enforcement.

Both schemas must properly validate input data and enforce type constraints, with AnswerResponse ensuring confidence is between 0-1 and UserIntent restricting intent_type to valid options.

Workflow Creation and Routing
Criteria	Submission Requirements
Create a complete LangGraph workflow.

The create_workflow function must instantiate a StateGraph, add all required nodes, set up proper routing with conditional edges, and return a compiled workflow.

Implement proper state flow and routing logic.

The workflow must correctly route requests based on intent classification and ensure state flows properly through all nodes to completion.

Tool Implementation
Criteria	Submission Requirements
Implement a functional calculator tool.

The calculator tool must use the @tool decorator, validate expressions for safety, evaluate mathematical expressions, log usage, and handle errors gracefully. The tool should return a string representation of the number (e.g “5” not 5)

Prompt Engineering
Criteria	Submission Requirements
Create an effective intent classification prompt.

The intent classification prompt must guide the LLM to classify intents accurately with clear categories, examples, and instructions for confidence scoring and reasoning.

Implement dynamic chat prompt selection.

The get_chat_prompt_template function must select appropriate system prompts based on intent type and return a properly structured ChatPromptTemplate.

Integration and Testing
Criteria	Submission Requirements
Demonstrate complete system functionality.

The implementation must work end-to-end, handling various user inputs across all three intent types with proper tool usage, memory management, and response generation.

Provide comprehensive documentation.

The submission must include a README or documentation explaining implementation decisions, how state and memory works, how structured outputs are enforced, and example conversations demonstrating all features.