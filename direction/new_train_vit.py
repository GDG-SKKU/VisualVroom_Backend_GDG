import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import vit_b_16
import sys
import os

def load_model(model_path, num_classes):
    # Load a ViT model
    model = vit_b_16(weights=None)
    
    # Modify the classifier head for our number of classes
    model.heads = nn.Sequential(
        nn.Linear(model.hidden_dim, num_classes)
    )
    
    # Adapt model to our image size (241x428)
    # Calculate new patch dimensions
    new_img_size = (241, 428)
    patch_size = 16  # Standard ViT patch size
    
    # Calculate number of patches in each dimension
    num_patches_height = new_img_size[0] // patch_size
    num_patches_width = new_img_size[1] // patch_size
    num_patches = num_patches_height * num_patches_width
    
    # Update patch projection to match our new size
    model.conv_proj = nn.Conv2d(
        in_channels=3,
        out_channels=model.hidden_dim,
        kernel_size=patch_size,
        stride=patch_size
    )
    
    # Load trained weights (will include the adapted position embeddings)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print(f"Loaded model adapted for image size {new_img_size} with {num_patches} patches")
    
    return model

def predict_image(model, image_path, class_names):
    # Image preprocessing - use original image size
    transform = transforms.Compose([
        transforms.Resize((241, 428)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    # Display image with predictions
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[top3_indices[0]]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    bars = plt.barh([class_names[idx] for idx in top3_indices], top3_prob.numpy())
    plt.xlabel("Probability")
    plt.title("Top 3 Predictions")
    plt.xlim(0, 1)
    
    for bar, score in zip(bars, top3_prob.numpy()):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.4f}', 
                 va='center')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    return class_names[top3_indices[0]], dict(zip([class_names[idx] for idx in top3_indices], top3_prob.numpy()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'vit_model.pth'
    
    # Get class names (assuming the train folder contains the class folders)
    train_dir = './train'
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, num_classes)
    
    print(f"Predicting image {image_path}...")
    pred_class, top3_preds = predict_image(model, image_path, class_names)
    
    print(f"\nPredicted class: {pred_class}")
    print("\nTop 3 predictions:")
    for cls, prob in top3_preds.items():
        print(f"{cls}: {prob:.4f}")