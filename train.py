import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from data.dataLoader import ImageCaptionDataset
from config.config import Config
from models.model import ImageCaptioningModel

import mlflow
import mlflow.pytorch

# TODO : Implementing Weights and Biases to for project tracking and evaluation and TODO : DVC also for data versioning


def train_model(model,dataLoader, optimizer, loss_fn):

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "device": Config.DEVICE
        })

    model.gpt2_model.train()
    for epoch in range(Config.EPOCHS):
        epoch_loss = 0
        for batch_idx, (images, captions) in tqdm(enumerate(dataLoader)):
            print(f'\rBatch {batch_idx + 1}/{len(dataLoader)} , Loss : {epoch_loss/(batch_idx+1):.4f}\t', end='')
            images = images.to(Config.DEVICE)
            captions = [caption for caption in captions]

            # extract image features
            image_features = model.extract_image_features(images)
            # print("Image Features shape:", image_features.shape)
            input_embeds, input_ids, attention_mask = model.prepare_gpt2_inputs(image_features, captions)

            # print("Input Embeds shape:", input_embeds.shape)
            # print("Input IDs shape:", input_ids.shape)
            # print("Attention Mask shape:", attention_mask.shape)
            # Match Inputs Embeds and Input Ids and Attention Masks
            assert input_embeds.shape[1] == input_ids.shape[1] == attention_mask.shape[1]

            optimizer.zero_grad()
            outputs = model.gpt2_model(inputs_embeds=input_embeds, labels=input_ids, attention_mask=attention_mask)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss/len(dataLoader):.4f}')
        mlflow.log_metric('loss', epoch_loss/len(dataLoader), step=epoch)

    # Save the model    
    model.save('model')
    # save the artifacts
    mlflow.log_artifacts('model')
    mlflow.pytorch.log_model(model.gpt2_model, "models")

    # return model


if __name__ == '__main__':
    # Initialize dataset using the CSV file
    dataset = ImageCaptionDataset(
        caption_file=Config.DATASET_PATH + 'captions.csv',    # Path to captions CSV file
        file_path = Config.DATASET_PATH+ '/images/', # Path to images folder
    )

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, # Specify the batch size
        shuffle=True, # Shuffle the data
        num_workers=4 # Number of subprocesses for data loading
    )

    # # Iterate over the dataloader
    # for batch_idx, (images, captions) in enumerate(dataloader):
    #     print(f'Batch {batch_idx + 1}:')
    #     print(f'Images shape: {images.shape}')
    #     print(f'Captions: {captions}')
    #     # Pass 'images' and 'captions' to your model for training/validation

    # Initialize the ImageCaptioningModel
    model = ImageCaptioningModel()
    optimizer = torch.optim.Adam(model.gpt2_model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    mlflow.set_experiment('ImageCaptioning')
    train_model(model, dataloader, optimizer, loss_fn)
