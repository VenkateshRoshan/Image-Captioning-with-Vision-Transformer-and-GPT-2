import torch
from transformers import ViTModel, ViTFeatureExtractor, GPT2LMHeadModel, GPT2Tokenizer
from config.config import Config
from torchsummary import summary
from torchvision import transforms

class ImageCaptioningModel:
    def __init__(self):
        """Initialize the ViT and GPT-2 models for image captioning."""
        self.device = Config.DEVICE
        self.vit_model = ViTModel.from_pretrained(Config.VIT_MODEL).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(Config.VIT_MODEL)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(Config.GPT2_MODEL).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_image_features(self, images):
        """Extract features from images using ViT."""
        pixel_values = self.feature_extractor(images=images, return_tensors="pt", do_rescale=False).pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    
    def prepare_gpt2_inputs(self, image_features, captions):
        """Prepare GPT-2 inputs."""
        # Tokenize the captions
        tokenized_captions = self.tokenizer(captions, padding="longest", truncation=True,
                                            max_length=Config.MAX_SEQ_LEN, return_tensors="pt").to(self.device)

        # Get the word embeddings for the tokens
        token_embeddings = self.gpt2_model.transformer.wte(tokenized_captions['input_ids'])

        # Concatenate image features with token embeddings
        image_features = image_features.unsqueeze(1)  # Reshape to [batch_size, 1, hidden_size]
        inputs_embeds = torch.cat((image_features, token_embeddings), dim=1)  # Concatenate along the sequence dimension

        # Adjust input_ids to account for the image feature token
        batch_size = image_features.shape[0]
        image_token_id = torch.full((batch_size, 1), fill_value=self.tokenizer.bos_token_id, device=self.device)
        input_ids = torch.cat((image_token_id, tokenized_captions['input_ids']), dim=1)

        # Adjust attention_mask to account for the image feature token
        image_attention = torch.ones((batch_size, 1), device=self.device)
        attention_mask = torch.cat((image_attention, tokenized_captions['attention_mask']), dim=1)

        return inputs_embeds, input_ids, attention_mask
    
    def save(self, path):
        """Save model to disk."""
        self.gpt2_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        """Load model from disk."""
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path).to(self.device)
