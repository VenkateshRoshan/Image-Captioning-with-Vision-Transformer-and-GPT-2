from PIL import Image
from models.model import ImageCaptioningModel
from torchvision import transforms
import torch

import torch
from transformers import ViTModel, ViTFeatureExtractor, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
from config.config import Config

class ImageCaptioningInference:
    def __init__(self, model):
        self.model = model
        self.device = Config.DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def infer_image(self, image):
        # Load and preprocess the image
        # image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Extract image features
        image_features = self.model.extract_image_features(image)

        # Generate caption
        caption = self.generate_caption(image_features)
        return caption
    
    def generate_caption(self, image_features, num_beams=3, max_length=50):
        # Prepare the image features for input
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Generate caption using beam search
        output = self.model.gpt2_model.generate(
            inputs_embeds=image_features,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            bos_token_id=self.model.tokenizer.bos_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id
        )
        
        # Decode the generated caption
        caption = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
        return caption

if __name__ == "__main__":
    # Path to the saved model directory
    model_dir = 'model'
    
    # Initialize inference class
    model = ImageCaptioningModel()
    model.load(model_dir)

    inference_model = ImageCaptioningInference(model)
    
    # Path to the input image
    image_path = 'test_img.jpg'

    image = Image.open(image_path)
    
    # Perform inference and print the generated caption
    caption = inference_model.infer_image(image)
    print("Generated Caption:", caption)

