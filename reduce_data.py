import os
import random

def delete_random_images(folder_path, num_to_delete):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Check if we have enough files to delete
    if len(files) <= num_to_delete:
        print(f"Not enough files in {folder_path} to delete {num_to_delete} images.")
        return

    # Randomly select files to delete
    files_to_delete = random.sample(files, num_to_delete)
    
    # Delete the selected files
    deleted_count = 0
    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            
    print(f"Successfully deleted {deleted_count} images from {os.path.basename(folder_path)}.")

# --- PATH CONFIGURATION ---
# Target the 'train' folder since it holds the majority of the dataset
base_train_dir = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted\train"

# We will delete 250 images from each of the 4 classes (Total deleted = 1,000 images)
# 10,500 - 1,000 = ~9,500 images left
delete_random_images(os.path.join(base_train_dir, "mangrove_forest"), 250)
delete_random_images(os.path.join(base_train_dir, "coniferous_pine_forest"), 250)
delete_random_images(os.path.join(base_train_dir, "tropical_rainforest"), 250)
delete_random_images(os.path.join(base_train_dir, "bamboo_forest"), 250)

print("\nProcess to reduce dataset size is complete!")