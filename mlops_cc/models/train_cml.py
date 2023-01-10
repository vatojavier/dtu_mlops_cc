# assume we have a trained model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from mlops_cc.models import model
from torch.utils.data import DataLoader, TensorDataset


model_checkpoint = "models/trained_modelV2.pt"
mymodel = model.MyAwesomeModel(784, 10)
mymodel.load_state_dict(torch.load(model_checkpoint))

images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")

train_dataset = TensorDataset(images, labels)  # create your datset
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

preds, target = [], []
for batch in trainloader:
    x, y = batch

    # resize images
    x = x.view(x.shape[0], -1)

    probs = mymodel(x)

    preds.append(probs.argmax(dim=-1))
    target.append(y.detach())

target = torch.cat(target, dim=0)
preds = torch.cat(preds, dim=0)

report = classification_report(target, preds)

with open("reports/classification_report.txt", 'w') as outfile:
    outfile.write(report)

confmat = confusion_matrix(target, preds, labels=[i for i in range(10)])
disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=[i for i in range(10)])
disp.plot()
plt.savefig('reports/figures/confusion_matrix.png')
