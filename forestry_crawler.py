from bing_image_downloader import downloader

# The 4 classes we established
forestry_classes = [
    "mangrove_forest", 
    "coniferous_pine_forest", 
    "tropical_rainforest", 
    "bamboo_forest"
]

# Loop to download images for each class
for f_class in forestry_classes:
    print(f"Downloading images for: {f_class}")
    
    # Since the target is up to 10,000 images, we set 2500 images per class (2500 x 4 = 10,000)
    downloader.download(f_class, 
                        limit=2500,  
                        output_dir='forestry_dataset', 
                        adult_filter_off=True, 
                        force_replace=False, 
                        timeout=60)
                        
print("Download complete! Please check the 'forestry_dataset' folder.")