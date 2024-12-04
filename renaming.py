import os
import re

def rename_files(directory):
    """
    Rename files by removing numbers and 'nhandnguyen' from filenames
    Parameters:
        directory (str): Path to the directory containing files
    """
    try:
        count = 1
        # Sort the filenames numerically
        filenames = sorted([f for f in os.listdir(directory) if f.endswith('.wav')], 
                         key=lambda x: int(re.search(r'^\d+', x).group()))
        
        for filename in filenames:
            # Create simple numbered filename
            new_filename = f"{count}.wav"
            count += 1
            
            # Create full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} → {new_filename}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Get user input
    directory = input("Enter directory path: ")
    
    # Check if directory exists
    if os.path.exists(directory):
        rename_files(directory)
    else:
        print("Directory does not exist!")
