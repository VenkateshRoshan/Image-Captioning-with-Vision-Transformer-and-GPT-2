import torch
class Config:
    IMAGE_SIZE = (224, 224)
    MAX_SEQ_LEN = 64
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    GPT2_MODEL = 'gpt2'
    LEARNING_RATE = 1e-4 #5e-5
    EPOCHS = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    AWS_S3_BUCKET = 'your-s3-bucket-name'
    DATASET_PATH = '../Datasets/Flickr8K/'
    BATCH_SIZE = 32