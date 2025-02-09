def clean_code(file_path, output_path=None):
    """
    Cleans Python code by removing empty lines while preserving basic code structure.
    
    Parameters:
    file_path (str): Path to the input Python file
    output_path (str, optional): Path where the cleaned code will be saved. 
                               If None, creates a new file with '_cleaned' suffix
    
    Returns:
    str: Path to the cleaned file
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try with utf-8-sig (for files with BOM)
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            # If both fail, use latin-1 which can read all 8-bit values
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()

    # Clean the lines
    cleaned_lines = []
    prev_line_was_code = False
    in_function_def = False
    
    for line in lines:
        # Strip whitespace
        stripped_line = line.rstrip()
        current_line = stripped_line.strip()
        
        # Skip completely empty lines
        if not current_line:
            continue
            
        # Check if we're entering a function definition
        if current_line.startswith('def '):
            in_function_def = True
            if prev_line_was_code:
                cleaned_lines.append('')  # Add single blank line before function
            cleaned_lines.append(stripped_line)
            prev_line_was_code = True
            continue
            
        # Check if we're entering a class definition
        if current_line.startswith('class '):
            if prev_line_was_code:
                cleaned_lines.append('')  # Add single blank line before class
            cleaned_lines.append(stripped_line)
            prev_line_was_code = True
            continue

        # Handle comments
        if current_line.startswith('#'):
            if not prev_line_was_code:
                cleaned_lines.append(stripped_line)
            else:
                cleaned_lines.append('')
                cleaned_lines.append(stripped_line)
            prev_line_was_code = False
            continue

        # Add the line if it contains code
        cleaned_lines.append(stripped_line)
        prev_line_was_code = True
        
        # Reset function definition flag if we've moved past the function
        if in_function_def and not current_line.startswith((' ', '\t')):
            in_function_def = False
    
    # Determine output path
    if output_path is None:
        base_name = file_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_cleaned.py"
    
    # Write cleaned code to file with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(cleaned_lines))
    
    return output_path




# Example usage:
if __name__ == "__main__":
    # Example usage with a file
    input_file = "multicell.py"
    cleaned_file = clean_code(input_file)
    print(f"Cleaned code saved to: {cleaned_file}")
    
    input_file = "singlecell.py"
    cleaned_file = clean_code(input_file)
    print(f"Cleaned code saved to: {cleaned_file}")