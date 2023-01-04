from mlops_cc.models import model
from torch.utils.data import DataLoader, TensorDataset
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

model =  model.MyAwesomeModel(784, 10)

images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")

train_dataset = TensorDataset(images, labels)  # create your datset
trainloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)  

n_epochs = 40

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.3)

steps = 0
losses = []
for e in range(n_epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    else:
        losses.append(running_loss/len(trainloader))
        steps += 1
        print(f"Training loss: {running_loss/len(trainloader)}")

steps = [i for i in range(steps)]

# Use the plot function to draw a line plot
plt.plot(steps, losses)

# Add a title and axis labels
plt.title("Training Loss vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")

# Save the plot
plt.savefig("reports/figures/lossV1.png")

torch.save(model.state_dict(), 'models/trained_modelV1.pt')