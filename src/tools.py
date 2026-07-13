from langchain_core.tools import tool

@tool
def write_to_file(filename: str, content: str) -> str:
    """Write content to a file. Creates a new file or overwrites existing file.
    
    Args:
        filename: The path to the file to write
        content: The content to write to the file
    
    Returns:
        A success message with the filename or an error message
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Content successfully written to {filename}"
    except Exception as e:
        return f"An error occurred: {str(e)}"