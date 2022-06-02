import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import copy
import random
import time
from torch.utils.data import Dataset
from torchvision import datasets
import tqdm
from Dataset import Objects
# import wandb
from models.benchmark import Benchmark, IterativeBenchmark
# from loss import EMDLosspy
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("Using Cuda , Hhheeh")
else:
    device = torch.device("cpu")
EPOCHS = 100

best_valid_loss = float('inf')

BATCH_SIZE = 1
dataset_dir = '/home/sombit/Downloads/objects/dataset/'
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 100),nn.ReLU(),nn.Dropout(0.5),nn.Linear(100, 50),nn.ReLU(),nn.Dropout(0.5),nn.Linear(50, 50),nn.ReLU(),nn.Dropout(0.5),nn.Linear(50, output_dim))  
    def forward(self, x):


        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.encoder(x)
        return x
# model = MLP(input_dim=24*3, output_dim=4).to(device)
model = IterativeBenchmark(in_dim=3,niters=4,gn="True").to(device)
model.float()
train_data = Objects(dataset_dir=dataset_dir, transform=transforms.ToTensor())
train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)
print("Train dataset size: ", len(train_iterator))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
train_set, val_set = torch.utils.data.random_split(train_data, [int(len(train_data)*0.8), len(train_data) - int(0.8*len(train_data))])

train_iterator=torch.utils.data.DataLoader(train_set,
        batch_size=BATCH_SIZE, shuffle=True)
val_iterator=torch.utils.data.DataLoader(val_set,
        batch_size=8, shuffle=True)
# loss_fn = EMDLosspy()
# loss_fn = loss_fn.cuda()
def compute_loss(ref_cloud, pred_ref_clouds, loss_fn):
    losses = []
    discount_factor = 0.5
    for i in range(8):
        loss = loss_fn(ref_cloud[..., :3].contiguous(),
                       pred_ref_clouds[i][..., :3].contiguous())
        losses.append(discount_factor**(8 - i)*loss)
    return torch.sum(torch.stack(losses))

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    
    for i, (batch) in enumerate(iterator):
        optimizer.zero_grad()
        # print(batch)
        x = batch[0]['frag'].float().to(device)
        y = batch[0]['other'].float().to(device)
        z = batch[0]['all'].float().to(device)
        print(x.shape,y.shape,z.shape)
        R = batch[1]
        t = batch[2]
        R, t, pred_cloud = model(x.permute(0, 2, 1).contiguous(),
                                     y.permute(0, 2, 1).contiguous())
        a = torch.cat((y,pred_cloud[0]),axis = 1)
        print(a.shape,z.shape)
        # loss = compute_loss(z, a, loss_fn)
        loss_chamfer, _ = chamfer_distance(z, a)

        loss_chamfer.backward()
        epoch_loss += loss_chamfer.item()
        optimizer.step()
    

        
    #     # print(type(x))

    #     y = batch['target']
    #     # y = y.type(torch.LongTensor))
    #     y = y.long().cuda()
    #     optimizer.zero_grad()

    #     y_pred = model(x)
    #     loss = criterion(y_pred, y)

    #     acc = calculate_accuracy(y_pred, y)

    #     epoch_loss += loss.item()
    #     epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion, device):
    ... 

    # epoch_loss = 0
    # epoch_acc = 0

    # model.eval()

    # with torch.no_grad():

    #     for i, batch in enumerate(iterator):
    #         x = batch['representation'].to(device)
    #         # print(type(x))

    #         y = batch['target']
    #         # y = y.type(torch.LongTensor))
    #         y = y.long().cuda()
    #         y_pred = model(x)
    #         loss = criterion(y_pred, y)

    #         acc = calculate_accuracy(y_pred, y)

    #         epoch_loss += loss.item()
    #         epoch_acc += acc

    # return epoch_loss / len(iterator), epoch_acc / len(iterator)

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()

    train_loss = train(model, train_iterator, optimizer, criterion, device)
    # valid_loss, valid_acc = evaluate(model, val_iterator, criterion, device)

    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'model'+epoch+'.pth')

    end_time = time.monotonic()

    # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss})

    # wandb.log({'val_accuracy': valid_acc, 'valid_loss': valid_loss})

    # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print("Epoch" , epoch+1, "EpochLoss",train_loss)
    # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    # print(f'\tVal Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}%')


