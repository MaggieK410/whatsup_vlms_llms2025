import os
import json
from PIL import Image, ImageEnhance
import numpy as np

def add_noise(image, noise_level=40):
    """
    Add Gaussian noise to a PIL image.
    """
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def change_color(image, brightness_factor=1.2, contrast_factor=1.2, saturation_factor=1.2):
    """
    Change the image color by adjusting brightness, contrast, and saturation.
    """
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    # Adjust saturation (color)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)
    return image

def horizontal_flip(image):

    return image.transpose(Image.FLIP_LEFT_RIGHT)

def augment_image(image):
    """
    Apply a set of augmentations to the image.
    Returns a dictionary mapping augmentation names to augmented PIL images.
    """
    augmented_images = {}
    augmented_images['vertical_flip'] = horizontal_flip(image)
    augmented_images['noise'] = add_noise(image)
    augmented_images['color_change'] = change_color(image)
    return augmented_images

def create_augmented_dataset(json_file, output_images_dir, output_json_file):
    """
    Loads the original JSON file, applies augmentations to each image,
    saves the augmented images in output_images_dir, and writes a new JSON file.
    
    Args:
        json_file (str): Path to the original JSON file (e.g., on_under_images.json).
        output_images_dir (str): Folder where augmented images will be saved.
        output_json_file (str): Path for the output JSON file.
    """
    # Load the original JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    augmented_data = []
    
    for sample in data:
        # The sample should contain "image_path" and "caption_options"
        orig_image_path = sample["image_path"]
        
        try:
            image = Image.open(orig_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {orig_image_path}: {e}")
            continue
        
        # Get augmentations
        aug_images = augment_image(image)
        for aug_name, aug_image in aug_images.items():
            # Build a new filename: original name + '_' + aug_name + extension
            base_name, ext = os.path.splitext(os.path.basename(orig_image_path))
            new_filename = f"{base_name}_{aug_name}{ext}"
            new_image_path = os.path.join(output_images_dir, new_filename)
            
            # Save the augmented image
            aug_image.save(new_image_path)
            
            # Create a new sample with updated image path
            # Here we store relative paths to the output_images_dir folder
            new_sample = {
                "image_path": "data/" + os.path.join(os.path.basename(output_images_dir), new_filename),
                "caption_options": sample["caption_options"]
            }
            augmented_data.append(new_sample)
    
    # Save the new augmented dataset JSON file
    with open(output_json_file, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    
    print(f"Augmented dataset created with {len(augmented_data)} samples.")
    

if __name__ == '__main__':
    # Modify these paths as needed
    original_json = "data/on_under_images.json"        # Your original JSON file
    output_images_dir = "data/on_under_images_augmented"  # New folder for augmented images
    output_json = "data/on_under_images_augmented.json"       # New JSON file with augmented samples
    
    create_augmented_dataset(original_json, output_images_dir, output_json)
