import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
n_epochs = 30
batch_size = 64
batch_size_test = 10240
learning_rate = 0.0005
momentum = 0.1
random_seed = 1 # We are setting some seeds by hand BUT we use CUDA for faster training, so training is not deterministic see: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(random_seed)

# #Custom datasets are cool but, how about not reinventing the wheel
# class ColorsDataset(Dataset):
#     def __init__(self, transforms=None):
#         self.colors = []
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.)

#     def __getitem__(self, idx):
#         data = torch.from_numpy(data).float()
#         target = torch.from_numpy(target).float()
#         return data, target

inps = torch.from_numpy(np.load("104811_positions_data.npy")).float()
tgts = torch.from_numpy(np.divide(np.load("104811_positions_targets.npy"),1000)).float()
tgts = tgts[:,None]
test_data = TensorDataset(inps, tgts)
del(inps,tgts)

inps = torch.from_numpy(np.load("1050788_positions_data.npy")).float()
tgts = torch.from_numpy(np.divide(np.load("1050788_positions_targets.npy"),1000)).float()
tgts = tgts[:,None]
training_data = TensorDataset(inps, tgts)
del(inps,tgts)

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size_test)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

for x, y in test_dataloader:
    print("Shape of test X [N, rgb]: ", x.shape)
    print("Shape of test y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()

#         self.convA1 = nn.Conv2d(7, 24, 3, stride= 1, padding= 1)
#         self.convA2 = nn.Conv2d(24, 24, 3, stride= 1, padding= 0)


#         self.convB1 = nn.Conv2d(24, 32, 3, stride= 1, padding= 1)
#         self.convB2 = nn.Conv2d(32, 32, 3, stride= 1, padding= 0)

#         self.convC1 = nn.Conv2d(32, 48, 3, stride= 1, padding= 1)
#         self.convC2 = nn.Conv2d(48, 48, 3, stride= 1, padding= 0)

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(512, 64)
#         self.fc2 = nn.Linear(64, 1)
        

#     def forward(self, x):
#         x = self.convA1(x)
#         x = self.convA2(x)
#         x = F.relu(x)
#         x = self.convB1(x)
#         x = self.convB2(x)
#         x = F.relu(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = torch.sigmoid(x)
#         return output

#with
# batch_size = 64
# batch_size_test = 1024
# learning_rate = 0.0005


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.convA1 = nn.Conv2d(7, 32, 3, stride= 1, padding= 1)
        self.convA2 = nn.Conv2d(32, 32, 3, stride= 1, padding= 0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, 1)
        

    def forward(self, x):
        x = self.convA1(x)
        x = self.convA2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(448, 256)
#         self.fc2 = nn.Linear(256, 1)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = torch.sigmoid(x)
#         return output

writer = SummaryWriter(f'runs/lr_{learning_rate}m_{momentum}double conv 64_large_dataset')
model = NeuralNetwork()
dataiter = iter(train_dataloader)
inps,tgts = dataiter.next()
writer.add_graph(model, inps)
del(inps,tgts)

model = model.to(device)

print(model)

loss_fn = nn.MSELoss()
loss_fn_lin = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer, epoch):

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        lossLin = loss_fn_lin(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss',
                        lossLin,
                        (epoch* len(train_dataloader)+ batch)/10)

        if batch % 64 == 0:
            print(f"loss: {loss:>7f}")


def test(dataloader, model, epoch):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, test_loss_lin, correct = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y)
            test_loss_lin += loss_fn_lin(pred,y)
    print(f"Test loss: {test_loss:>7f}")

    writer.add_scalar("test loss",
                test_loss_lin,
                epoch* len(test_dataloader))

test(test_dataloader, model, 0)

for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t+1)
    test(test_dataloader, model, t+1)
print("Done!")
writer.close()

torch.save(model, 'model.pt')

for x, y in test_dataloader:
    # print(f"X:{x[31:32,:,:,:]}")
    print(f"y:{y[64:65,:]}")

    model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        pred = model(x[64:65,:,:,:])
        print(f"predicted: y:{pred}")
    print(x.shape)
    print(y.shape)
    break
