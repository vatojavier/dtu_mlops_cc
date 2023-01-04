from torch.utils.data import DataLoader, TensorDataset
from mlops_cc.models import model
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from torchvision import transforms


def get_data(path):
    test_images = []
    test_labels = []
    data = np.load(f"{path}")
    [test_images.append(img) for img in data["images"]]
    [test_labels.append(label) for label in data["labels"]]

    test_images = np.array(test_images)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    test_labels = np.array(test_labels)
    test = [
        torch.tensor(test_images).type(torch.float32),
        torch.tensor(test_labels).long(),
    ]

    return test

def normalize_data(data):

    # Normalize
    for idx, image in enumerate(data):
        mean = image.mean().item()
        std = image.std().item()

        transform_norm = transforms.Compose([transforms.Normalize(mean, std)])
        img_normalized = transform_norm(image)
        data[idx] = img_normalized

    return data



@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
def main(model_checkpoint, data_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Using model in inference")
    print(model_checkpoint)

    mymodel = model.MyAwesomeModel(784, 10)
    mymodel.load_state_dict(torch.load(model_checkpoint))

    images, labels = get_data(data_path)
    images = normalize_data(images)

    test_dataset = TensorDataset(images, labels)  # create your datset
    testloader = DataLoader(
        test_dataset, batch_size=64
    )  # create your dataloader

    with torch.no_grad():
        for images, labels in testloader:
            # resize images
            images = images.view(images.shape[0], -1)

            mymodel.eval()
            logps = mymodel.forward(images)
            ps = torch.exp(logps)

            # Take max from the probs
            top_p, top_class = ps.topk(1, dim=1)

            # Compare with labels
            equals = top_class == labels.view(*top_class.shape)

            # mean
            accuracy = torch.mean(equals.type(torch.FloatTensor))

    print(f'Accuracy: {accuracy.item()*100}%')


    


    


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
