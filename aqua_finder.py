import os
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from pillow_heif import register_heif_opener
register_heif_opener()

# hardware configuration
device = torch.device("mps")
print(f"Executing on: {device}")

# load vision language model
model_id = "openai/clip-vit-base-patch32"
# load neural network weights and sends to apu
model = CLIPModel.from_pretrained(model_id).to(device)
# resize photos for model
processor = CLIPProcessor.from_pretrained(model_id)

# -------- PATHS --------
reference_image = ""
dataset_dir = ""
output_dir = ""

# 1.0 is 1-1 pixel match
similarity_threshold = 0.85 

#create dir if doesnt exit
os.makedirs(output_dir, exist_ok=True)

print("encoding reference image...")
# open image and process to tensors
ref_image = Image.open(reference_image).convert("RGB")
ref_inputs = processor(images=ref_image, return_tensors="pt").to(device)


# disables training!!!
with torch.no_grad():
    ref_features = model.get_image_features(**ref_inputs)

print(f"Scanning directory: {dataset_dir}")
valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".heic") #check docs for .heic compatibilty

# os.walk traverses the main directory and all subdirectories
for root, dirs, files in os.walk(dataset_dir):
    for filename in files:
        # Check if the file matches our valid extensions
        if not filename.lower().endswith(valid_extensions):
            continue
            
        # Join the current folder path (root) with the filename
        img_path = os.path.join(root, filename)
        
        try:
            # Process current image
            target_image = Image.open(img_path).convert("RGB")
            target_inputs = processor(images=target_image, return_tensors="pt").to(device)
            
            # Disable learning 
            with torch.no_grad():
                target_features = model.get_image_features(**target_inputs)
                
            # Calculate similarity
            similarity = F.cosine_similarity(ref_features, target_features).item()
            
            if similarity >= similarity_threshold:
                print(f"Match: {img_path} | Score: {similarity:.4f}")
                # Use img_path to copy from the exact subfolder
                shutil.copy2(img_path, os.path.join(output_dir, filename))
                
        except Exception as e:
            # Catch errors
            print(f"Failed to process {img_path}: {e}")

print("Search Complete!")
