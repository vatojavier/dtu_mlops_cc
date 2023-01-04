# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch
from torchvision import transforms


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train_images = []
    train_labels = []
    for i in range(5):
        data = np.load(f"{input_filepath}/train_{i}.npz")
        [train_images.append(img) for img in data["images"]]
        [train_labels.append(label) for label in data["labels"]]

    train_images = np.array(train_images)
    # reshape to  (n_immgs, channels, pixels, pixels)
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    train_images = torch.from_numpy(train_images).type(torch.float32)

    train_labels = np.array(train_labels)
    train_labels = torch.from_numpy(train_labels).long()

    # Normalize
    for idx, image in enumerate(train_images):
        mean = image.mean().item()
        std = image.std().item()

        transform_norm = transforms.Compose([transforms.Normalize(mean, std)])
        img_normalized = transform_norm(image)
        train_images[idx] = img_normalized

    torch.save(train_images, f"{output_filepath}/images.pt")
    torch.save(train_labels, f"{output_filepath}/labels.pt")

    # Now labels
    


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
