import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset class to handle loading and transforming image-caption pairs
    where image paths and captions are provided in a CSV file.
    
    Attributes:
        caption_file (str): Path to the CSV file containing image paths and captions.
        transform (torchvision.transforms.Compose): Transformations to apply on the images.
    """

    def __init__(self, caption_file: str, file_path: str, transform=None):
        """
        Initialize dataset with caption CSV file and optional transform.
        
        Args:
            caption_file (str): Path to the CSV file where each row has an image path and caption.
            transform (callable, optional): Optional transform to apply on an image.
        """
        self.df = pd.read_csv(caption_file)
        self.image_path = file_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),   # Resize to 224x224 for ViT
            transforms.ToTensor(),           # Convert to tensor
            # Normalize to have values in the range [0, 1]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding caption by index.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: (image, caption) where image is the transformed image tensor and caption is the associated text.
        """
        img_path = self.df.iloc[idx, 0]  # The first column contains image paths
        caption = self.df.iloc[idx, 1]   # The second column contains captions
        # Load image
        image = Image.open(self.image_path+img_path).convert('RGB')

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, caption
