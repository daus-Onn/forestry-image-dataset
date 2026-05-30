import os
import pandas as pd

def create_dataset_excel(base_folder, output_filename):
    dataset_info = []
    
    # Traverse through train, val, and test directories
    for dataset_type in ['train', 'val', 'test']:
        type_folder = os.path.join(base_folder, dataset_type)
        
        # Check if the directory exists
        if not os.path.exists(type_folder):
            continue
            
        # Traverse through each class folder (e.g., mangrove, bamboo, etc.)
        for class_label in os.listdir(type_folder):
            class_folder = os.path.join(type_folder, class_label)
            
            # Ensure it is a directory
            if os.path.isdir(class_folder):
                # Loop through all image files in the class folder
                for filename in os.listdir(class_folder):
                    file_path = os.path.join(class_folder, filename)
                    
                    # Ensure it is a file
                    if os.path.isfile(file_path):
                        # Append the details to our list
                        dataset_info.append({
                            'Filename': filename,
                            'Class_Label': class_label,
                            'Dataset_Type': dataset_type,
                            'File_Path': file_path
                        })
                        
    # Convert the compiled list into a Pandas DataFrame (Table format)
    df = pd.DataFrame(dataset_info)
    
    # Save the DataFrame to a CSV file (readable by Microsoft Excel)
    df.to_csv(output_filename, index=False)
    print(f"Success! Dataset summary has been saved to: {output_filename}")

# --- PATH CONFIGURATION ---
# Path to your fully processed and splitted dataset
dataset_path = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted"

# Name of the output Excel/CSV file
output_excel_file = "forestry_dataset_summary.csv"

# Execute the function
create_dataset_excel(dataset_path, output_excel_file)