import os

def modify_filename(filename):
    """
    Modify the filename based on the specified rules:
    - If the filename contains "-", add "_" before "-"
    - Otherwise, add "_" before the 8th last character
    """
    if "-" in filename:
        return filename.replace("-", "_-", 1)
    else:
        return filename[:-8] + "_" + filename[-8:]

# Define the root directory to start the search
root_directory = '/Users/zhangwenbo/Documents/RVSS_Need4Speed/data'  # Replace with your target directory

# Walk through the directory
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        # Process only jpg files that do not contain "_"
        if filename.endswith('.jpg') and "_" not in filename:
            # Modify the filename based on the rules
            new_filename = modify_filename(filename)
            
            # Build the full old and new file paths
            old_file_path = os.path.join(dirpath, filename)
            new_file_path = os.path.join(dirpath, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}" in directory "{dirpath}"')
