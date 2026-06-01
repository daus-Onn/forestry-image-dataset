import os
import uuid

def sequentially_rename_files(base_dir):
    datasets = ['train', 'val', 'test']
    classes = {
        'mangrove_forest': 'mangrove',
        'coniferous_pine_forest': 'coniferous',
        'tropical_rainforest': 'tropical',
        'bamboo_forest': 'bamboo'
    }

    for dataset_type in datasets:
        for folder_name, prefix in classes.items():
            folder_path = os.path.join(base_dir, dataset_type, folder_name)
            
            if not os.path.exists(folder_path):
                continue
                
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            # PHASE 1: Rename all to random temporary names to prevent WinError 183 collision
            temp_files = []
            for filename in files:
                file_path = os.path.join(folder_path, filename)
                file_extension = os.path.splitext(filename)[1]
                
                # Generate a completely unique random string
                temp_name = f"TEMP_{uuid.uuid4().hex}{file_extension}"
                temp_filepath = os.path.join(folder_path, temp_name)
                
                os.rename(file_path, temp_filepath)
                temp_files.append(temp_name)
                
            # PHASE 2: Rename the temporary files into neat sequential numbers
            count = 1
            for temp_name in temp_files:
                temp_filepath = os.path.join(folder_path, temp_name)
                file_extension = os.path.splitext(temp_name)[1]
                
                final_name = f"{prefix}_{count}{file_extension}"
                final_filepath = os.path.join(folder_path, final_name)
                
                os.rename(temp_filepath, final_filepath)
                count += 1
                
            print(f"Successfully sequenced: {dataset_type} -> {folder_name}")

# --- PATH CONFIGURATION ---
dataset_dir = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted"

# Execute the function
sequentially_rename_files(dataset_dir)
print("\nProcess completed! All files are sequentially numbered without any collisions.")