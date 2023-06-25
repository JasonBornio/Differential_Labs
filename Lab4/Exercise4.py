# Execute this code block to install dependencies when running on colab

import torchbearer
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

#flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])

def loadData(batch_size: int=128):
    trainset = MNIST(".", train=True, download=True, transform=transform)
    testset = MNIST(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out

from torchbearer.callbacks import LiveLossPlot

def trainMLP2(model, train_data, test_data):
    # define the loss function and the optimiser
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    #%matplotlib inline
    callbacks = [LiveLossPlot()]
    
    # Construct a trial object with the model, optimiser and loss.
    # Also specify metrics we wish to compute.
    trial = torchbearer.Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'], callbacks=callbacks).to(device)

    # Provide the data to the trial
    trial.with_generators(train_generator=train_data,  val_generator=test_data,test_generator=test_data)

    # Run 10 epochs of training
    history = trial.run(epochs=10, verbose=0)
    
    return history, trial

def evaluate(history, trial):
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print("HISTORY:")
    print(history)
    print("RESULTS:")
    print(results)
    return

def question1(hidden_length: int = 10):
    train, test = loadData()
    model = BaselineModel(784, hidden_length, 10)
    return trainMLP2(model, train, test)

#------------------------------------------
def main():
    print("hidden units:: 1")
    history, trial = question1(1)
    evaluate(history, trial)
    
    print("hidden units:: 10")
    history, trial = question1(10)
    evaluate(history, trial)
    
    print("hidden units:: 100")
    history, trial = question1(100)
    evaluate(history, trial)
    
    print("hidden units:: 1,000")
    history, trial = question1(1000)
    evaluate(history, trial)
    
    print("hidden units:: 10,000")
    history, trial = question1(10000)
    evaluate(history, trial)
    
    print("hidden units:: 100,000")
    history, trial = question1(100000)
    evaluate(history, trial)
    
    print("hidden units:: 150,000")
    history, trial = question1(150000)
    evaluate(history, trial)
    return 0

if __name__ == "__main__":
    main()
    