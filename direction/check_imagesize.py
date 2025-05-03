import os
from PIL import Image
import numpy as np
from collections import Counter

def check_image_sizes(data_dir, class_names):
    """
    Check and report the sizes of all images in the dataset
    """
    all_sizes = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    # Open the image and get its size
                    with Image.open(img_path) as img:
                        size = img.size  # Returns (width, height)
                        all_sizes.append(size)
    
    # Count unique sizes
    size_counter = Counter(all_sizes)
    
    print(f"Image size statistics for {data_dir}:")
    print(f"Total number of images: {len(all_sizes)}")
    print(f"Number of unique sizes: {len(size_counter)}")
    
    # Print the most common sizes
    print("\nMost common sizes (width x height):")
    for size, count in size_counter.most_common(10):
        print(f"  {size[0]} x {size[1]}: {count} images ({count/len(all_sizes)*100:.1f}%)")
    
    # If there are many different sizes, show some statistics
    if len(size_counter) > 1:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        
        print("\nWidth statistics:")
        print(f"  Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}, Median: {np.median(widths)}")
        
        print("Height statistics:")
        print(f"  Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}, Median: {np.median(heights)}")

# Add this to your main code
if __name__ == "__main__":
    # Define directories
    train_dir = "./train"
    val_dir = "./valid"
    test_dir = "./test"
    
    # Define classes
    class_names = [
        'Siren_L', 'Siren_R', 'Bike_L', 'Bike_R', 'Horn_L', 'Horn_R'
    ]
    
    # Check image sizes in each directory
    check_image_sizes(train_dir, class_names)
    check_image_sizes(val_dir, class_names)
    check_image_sizes(test_dir, class_names)