import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import wandb
import torch

class MyAwesomeModel(pl.LightningModule):
    # input_size = input_size
    # output_size = output_size
    criterion = nn.NLLLoss()

    losses = []
    def __init__(self, input_size, output_size):
        


        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.output = nn.Linear(64, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = F.log_softmax(self.output(x), dim=1)
        return x

    def configure_optimizers(self):
        # params = 
        optimizer = optim.SGD(self.parameters(), lr=0.3)
        return optimizer
       
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        images = images.view(images.shape[0], -1)
        # optimizer.zero_grad()
        output = self(images)
        loss = self.criterion(output, labels)
        self.log('train_loss', loss)

        # self.logger.experiment.log({'logits': wandb.Histrogram(output)}) # No furrula
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        # resize images
        images = images.view(images.shape[0], -1)

        logps = self(images)
        # logps = mymodel.forward(images)
        ps = torch.exp(logps)

        # Take max from the probs
        top_p, top_class = ps.topk(1, dim=1)

        # Compare with labels
        equals = top_class == labels.view(*top_class.shape)

        # mean
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        self.log('test_accuracy', accuracy)
        return accuracy





        


