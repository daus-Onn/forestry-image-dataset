import os

def rename_files(folder_path, prefix):
    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return

    # Process to rename files sequentially
    for count, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it is a file, not a sub-directory
        if os.path.isfile(file_path):
            # Get the file extension (e.g., .jpg)
            file_extension = os.path.splitext(filename)[1]
            
            # Create the new filename (e.g., mangrove_1.jpg)
            new_name = f"{prefix}_{count + 1}{file_extension}"
            new_path = os.path.join(folder_path, new_name)
            
            try:
                os.rename(file_path, new_path)
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")
                
    print(f"Successfully renamed all files in directory: {prefix}")

# --- CHANGE THE PATH BELOW TO MATCH YOUR DIRECTORY ---
base_dir = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset"

# Execute the function for each specific class
rename_files(os.path.join(base_dir, "mangrove_forest"), "mangrove")
rename_files(os.path.join(base_dir, "coniferous_pine_forest"), "coniferous")
rename_files(os.path.join(base_dir, "tropical_rainforest"), "tropical")
rename_files(os.path.join(base_dir, "bamboo_forest"), "bamboo")

print("\nFile renaming process completed successfully!")