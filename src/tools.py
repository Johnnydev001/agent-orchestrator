from langchain_core.tools import tool
from langsmith import traceable
    
@tool
def fail_on_purpose():
    """
    A tool that always fails when invoked. Used for testing error handling.
    This function is traceable for debugging and analysis purposes.
    """
    raise Exception("This tool is designed to fail on purpose.")

@traceable(run_type="prompt",name="parse_prompt", description="Parses the input prompt without leading or trailing whitespace.")
def parse_prompt_to_messages(prompt: str) -> str:
    """
    Parses the input prompt without leading or trailing whitespace.
    This function is traceable for debugging and analysis purposes.
    """
    parsed_prompt = prompt.strip()
    return parsed_prompt
   