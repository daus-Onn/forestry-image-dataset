import os
import hashlib

def remove_duplicates(folder_path):
    unique_hashes = set()
    duplicates_removed = 0

    # List all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it is a file (not a sub-directory)
        if os.path.isfile(file_path):
            # Read the file and generate MD5 Hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check if the hash already exists in the set
            if file_hash not in unique_hashes:
                unique_hashes.add(file_hash) # Store the unique hash
            else:
                os.remove(file_path) # Delete the file if it's a duplicate
                duplicates_removed += 1
                print(f"Removed (Duplicate): {filename}")

    print(f"\nProcess completed! Total duplicate images removed: {duplicates_removed}")

# Specify the path to the dataset folder
# Example below targets the bamboo forest class directory
remove_duplicates('dataset_forestry/bamboo_forest')