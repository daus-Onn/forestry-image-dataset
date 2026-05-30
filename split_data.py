import splitfolders

# Exact path to your current raw dataset folder
input_directory = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset"

# Path to create a new folder for the split dataset
output_directory = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted"

print("Processing and splitting the dataset... Please wait.")

# Splitting the data into 70% Train, 15% Validation, and 15% Test
# move=False means it copies the files, keeping your original 10,520 images intact
splitfolders.ratio(input_directory, 
                   output=output_directory, 
                   seed=1337, 
                   ratio=(.7, .15, .15), 
                   group_prefix=None, 
                   move=False)

print("Success! Dataset has been successfully split into train, val, and test folders.")