import numpy as np
import copy
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from DBCTNet_1_MMH import DBCTNet
from utils.FocalLoss import FocalLoss
from utils.CreateCube import create_patch
from utils.SplitDataset import split_dataset
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='PU',help='dataset')
parser.add_argument('--train_ratio', default=0.007,help='ratio of train')
parser.add_argument('--epochs',default=300,help='epoch')
parser.add_argument('--lr',default=0.001,help='learning rate')
parser.add_argument('--gamma',default=0.4,help='gamma of focal loss')
args = parser.parse_args()

config={'batch_size':128,
        'dataset':args.dataset,
        'model':'DBCTNet',
        'train_ratio':float(args.train_ratio),
        'weight_decay':0.001,
        'epoch':int(args.epochs),
        'patch_size':9,
        'fc_dim':16,
        'heads':2,
        'drop':0.1,
        'lr':float(args.lr),
        'gamma':float(args.gamma),
        }


def load_data(dataset):
    if dataset == 'PU':
        data = sio.loadmat('./data/Pavia_university/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./data/Pavia_university/PaviaU_gt.mat')['paviaU_gt']
    elif dataset == 'SA':
        data = sio.loadmat('./data/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./data/Salinas/Salinas_gt.mat')['salinas_gt']
    else:
        print('Please add your dataset.')
    return data, labels

def train(model,classes,config,device,sampler):
    epochs =  config['epoch']
    lr = config['lr']
    wd = config['weight_decay']
    dataset = config['dataset']
    gamma = config['gamma']
    model_name = config['model']
    model = model.to(device)
    criterion = FocalLoss(class_num=classes,gamma=gamma,alpha=sampler,use_alpha=True) 
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0.1*epochs*len(data_loader['train']), num_training_steps = epochs*len(data_loader['train']))
    best_acc = 0
    best_model = None
    best_epoch = 0
    train_losses = []  # 记录每个 epoch 的训练损失
    val_losses = []    # 记录每个 epoch 的验证损失
    
    for epoch in range(epochs):
        for phase in ['train','val']:
            running_loss = 0
            running_correct = 0
            for i, (data, target) in enumerate(data_loader[phase]):
                data, target = data.to(device), target.to(device)
                if phase == 'train':
                    model.train()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(data)
                        loss = criterion(outputs, target)
                running_loss += loss.item()*data.shape[0]
                _,outputs = torch.max(outputs,1)
                running_correct += torch.sum(outputs == target.data)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_correct.double() / len(data_loader[phase].dataset)
            
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                
            print('[Epoch: %d] [%s] [loss: %.4f] [acc: %.4f]' % (epoch + 1,phase,epoch_loss,epoch_acc))   
            
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                
    # 保存训练过程中损失的变化图
    save_loss_plot(train_losses, val_losses, dataset, model_name)
                
    path = f'./weights/{dataset}_{model_name}_{best_epoch}.pth'
    torch.save(best_model.state_dict(),path)
    print('Finished Training')
    return best_epoch
    
def test(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test
   
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
import matplotlib.pyplot as plt

def save_loss_plot(train_losses, val_losses, dataset_name, model_name):
    epochs = len(train_losses)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.savefig(f'./result/{dataset_name}/{model_name}_loss_plot.png')
    plt.close()

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

class to_dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.len = x.shape[0]
        self.data = torch.FloatTensor(x)
        self.gt = torch.LongTensor(y)
    def __getitem__(self,index):
        return self.data[index],self.gt[index]
    def __len__(self):
        return self.len

dataset_name = config['dataset']
model_name = config['model']
train_ratio = float(config['train_ratio'])
patch = config['patch_size']
data,gt = load_data(dataset_name)
batch_size=config['batch_size']


all_data,all_gt = create_patch(data,gt,patch_size = patch)
band = all_data.shape[-1]
classes = int(np.max(np.unique(all_gt)))
x_train,y_train,x_val,y_val,x_test,y_test,sampler = split_dataset(all_data,all_gt,train_ratio=train_ratio)
train_ds = to_dataset(x_train,y_train)
val_ds = to_dataset(x_val,y_val)
test_ds = to_dataset(x_test,y_test)
all_ds = to_dataset(all_data,all_gt)
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)
                                           
val_loader = torch.utils.data.DataLoader(val_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
data_loader = {'train':train_loader,
               'val':val_loader}

test_loader = torch.utils.data.DataLoader(test_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
all_loader = torch.utils.data.DataLoader(all_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)

model = DBCTNet(channels=16,
            patch=config['patch_size'],
            bands=band,
            num_class=classes,
            fc_dim=config['fc_dim'],
            heads=config['heads'],
            drop=config['drop'],
            )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_epoch = train(model,classes,config,device,sampler)

path = f'./weights/{dataset_name}_{model_name}_{best_epoch}.pth'
model.load_state_dict(torch.load(path))
model = model.to(device)
p_t,t_t=test(device,model,test_loader)
OA, AA_mean, Kappa, AA=output_metric(t_t, p_t)
print('Test \n','OA: ',OA,'AA_mean: ',AA_mean,' Kappa: ',Kappa,'AA: ',AA)

# 保存测试结果到文件
def save_test_results(OA, AA_mean, Kappa, AA, dataset_name, model_name):
    result_dir = os.path.join('./result', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{model_name}_test_results_1.txt'), 'w') as f:
        f.write(f'Test Results for {model_name} on {dataset_name} dataset:\n')
        f.write(f'Overall Accuracy (OA): {OA}\n')
        f.write(f'Average Accuracy (AA_mean): {AA_mean}\n')
        f.write(f'Kappa: {Kappa}\n')
        f.write(f'Class-wise Accuracy (AA):\n')
        for i, acc in enumerate(AA):
            f.write(f'Class {i + 1}: {acc}\n')

save_test_results(OA, AA_mean, Kappa, AA, dataset_name, model_name)
