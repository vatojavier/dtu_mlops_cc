
from mlops_cc.models import model
from torch.utils.data import DataLoader, TensorDataset
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from pytorch_lightning import Callback, Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def test_training():

    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)  # create your datset
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)


    class MetricTracker(Callback):
        def __init__(self):
            self.collection = []

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
            elogs = trainer.logged_metrics
            self.collection.append(elogs["train_loss"])
            # print(elogs)


    cb = MetricTracker()
    mymodel = model.MyAwesomeModel(784, 10)

    trainer = Trainer(max_epochs=5, callbacks=[cb], limit_train_batches=0.2, logger=WandbLogger(project="dtu_mlops"))

    trainer.fit(mymodel, trainloader)

    # Create plot for training loss
    losses = [i.item() for i in cb.collection]
    steps = [i for i in range(len(losses))]

    # Check that loss metrics has been collected
    assert len(losses) > 0 and len(losses) == len(steps), "Training losses metrics not registered"