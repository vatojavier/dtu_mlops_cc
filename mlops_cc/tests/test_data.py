from torch.utils.data import DataLoader, TensorDataset
from mlops_cc.models.predict_model import get_data
import torch
import pytest
import os


@pytest.mark.skipif(not os.path.exists("data/processed/images.pt"), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/labels.pt"), reason="Data files not found")
def test_data():
    N_TRAIN = 2500
    N_TEST = 5000

    train_images = torch.load("data/processed/images.pt")
    train_labels = torch.load("data/processed/labels.pt")

    train_dataset = TensorDataset(train_images, train_labels)  # create your datset
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)


    test_path = "data/raw/test.npz"
    test_images, test_labels = get_data(test_path)

    test_dataset = TensorDataset(test_images, test_labels)
    testloader = DataLoader(
        test_dataset, batch_size=64, num_workers=8
    )  # create your dataloader

    assert len(train_dataset) == N_TRAIN and len(test_dataset) == N_TEST, "Data length not correct"

    labels_set = set()
    # Checking shape of the train data
    for images, labels in trainloader:
        assert images.shape[1:] == (1,28,28), "Image shape incorrect" # first element (batch size coudl change)
        [labels_set.add(l.item()) for l in labels]

    # Checking shape of the test data
    for images, labels in testloader:
        assert images.shape[1:] == (1,28,28), "Image shape incorrect"

    # Checking all labels are in training data
    all_labels = [i for i in range(10)]
    for l in all_labels:
        assert l in labels_set


    # breakpoint()