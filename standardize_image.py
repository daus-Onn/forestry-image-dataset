import os
from PIL import Image

def convert_all_to_jpg(folder_path):
    total_converted = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is NOT already in jpg/jpeg format
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            
            # Ensure it is a file and not a sub-directory
            if os.path.isfile(file_path):
                try:
                    # Open the image file
                    img = Image.open(file_path)
                    
                    # Convert color mode to RGB (mandatory for JPG format)
                    rgb_im = img.convert('RGB')
                    
                    # Construct a new filename with the .jpg extension
                    new_name = os.path.splitext(filename)[0] + '.jpg'
                    new_path = os.path.join(folder_path, new_name)
                    
                    # Save the new image as JPG
                    rgb_im.save(new_path, 'JPEG')
                    
                    # Remove the original file (e.g., .webp or .png) to save space
                    os.remove(file_path)
                    
                    total_converted += 1
                    print(f"Successfully converted: {filename} -> {new_name}")
                    
                except Exception as e:
                    print(f"Failed to convert {filename}: Error - {e}")
                    # Optional: Uncomment the line below to delete corrupted files
                    # os.remove(file_path) 

    print(f"\nProcess completed! {total_converted} images have been standardized to JPG format.")

# Specify the path to the dataset folder
# Execute this function for each specific class directory
convert_all_to_jpg('dataset_forestry/bamboo_forest')