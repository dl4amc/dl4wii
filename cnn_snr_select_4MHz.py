
# coding: utf-8

# In[1]:


import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

torch.cuda.empty_cache() 

class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.next()
        self.last_duration = (time.time() - start)
        return n

    next = __next__

def shuffle_in_unison(x_data, y_data, seed=195735):
    '''
    This method shuffles the data of the Data Set axis of the test and training data.
    It is inspired by http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    '''
    np.random.seed(seed)
    shuffled_x = np.empty(x_data.shape, dtype=x_data.dtype)
    shuffled_y = np.empty(y_data.shape, dtype=y_data.dtype)
    permutation = np.random.permutation(x_data.shape[2])
    for old_index, new_index in enumerate(permutation):
        shuffled_x[::,::,new_index,::,::] = x_data[::,::,old_index,::,::]
        shuffled_y[::,::,new_index] = y_data[::,::,old_index]
    return shuffled_x, shuffled_y

def split_data(x_data, y_data, validation_fraction=0.2):
    """
    Splits the data into training and validation data
    according to the fraction that was specified. The samples are shuffled and then selected.
    The data is equally splitted along classes and signal to noise ratios.
    The new data array, validation array and the according label arrays are returned.
    """
    # Shuffle data
    x_data, y_data = shuffle_in_unison(x_data, y_data)
    # Split data
    nb_sets = x_data.shape[2]
    nb_cutted = int(np.floor(nb_sets * validation_fraction))
    x_test = x_data[::,::,-1:(-nb_cutted-1):-1,::,::]
    y_test = y_data[::,::,-1:(-nb_cutted-1):-1]
    x_data = np.delete(x_data, np.s_[-1:(-nb_cutted-1):-1], axis=2)
    y_data = np.delete(y_data, np.s_[-1:(-nb_cutted-1):-1], axis=2)
    return x_data, y_data, x_test, y_test

def load_data(storage_folder, which_kind):
    """
    Unpickle the data stored in the folder. The name of the measurement data
    inside the folder is considered as 'data_arrays_$which_kind-data-' + $standards and the
    label data is considered as 'labels-data-' + $standards.
    The measurement data and the labels for each standard 
    are returned in a list.
    Standard must be a list of Strings. The data is loaded in the order of the
    standards.
    """
    x_data = None
    y_data = None
    data_file = None
    # decide which data to load
    if which_kind == 'iq':
        data_file = 'data_iq.p'
    elif which_kind == 'fft':
        data_file = 'data_fft.p'
    elif which_kind == 'amplitude phase':
        data_file = 'data_amplitude_phase.p'
    elif which_kind == 'fft amplitude phase':
        data_file = 'data_fft_amplitude_phase.p'
    elif which_kind == 'subset fft':
        data_file = 'data_subset_fft.p'
    elif which_kind == 'subset fft amplitude phase':
        data_file = 'data_subset_fft_amplitude_phase.p'
    elif which_kind == 'fft 2422-2424 2429-2431':
        data_file = 'data_fft_2422_2424_2429_2431.p'
    elif which_kind == 'fft amplitude/phase 2422-2424 2429-2431':
        data_file = 'data_fft_amplitude_phase_2422_2424_2429_2431.p'
    else:
        raise ValueError('Parameter which_kind must be "iq" for IQ-data or "fft" for FFT-data.')
    # load input data (x)
    data_path = os.path.join(storage_folder, data_file)
    with open(data_path, mode='rb') as storage: 
        x_data = pickle.load(storage)
    # load output data/labels (y)
    label_file = 'labels.p'
    label_path = os.path.join(storage_folder, label_file)
    with open(label_path, mode='rb') as storage:
        y_data = pickle.load(storage, encoding='latin1')
    return x_data, y_data

def normalize_data(x_train, x_test):
    """
    $x_train and $x_test are numpy arrays which should be normalized.
    Normalizes the training data to have a train_mean of 0 and a standard deviation of 1. 
    The test data is normalized with the parameters of the training data
    Returns the normalized data in the same format as given.
    """
    train_mean_1 = np.mean(x_train[:,:,:,:,0])
    train_mean_2 = np.mean(x_train[:,:,:,:,1])

    train_std_dev_1 = np.std(x_train[:,:,:,:,0])
    train_std_dev_2 = np.std(x_train[:,:,:,:,1])

    x_train[:,:,:,:,0] = (x_train[:,:,:,:,0] - train_mean_1) / train_std_dev_1 # element-wise operations
    x_train[:,:,:,:,1] = (x_train[:,:,:,:,1] - train_mean_2) / train_std_dev_2 # element-wise operations
    
    x_test[:,:,:,:,0] = (x_test[:,:,:,:,0] - train_mean_1) / train_std_dev_1 # element-wise operations
    x_test[:,:,:,:,1] = (x_test[:,:,:,:,1] - train_mean_2) / train_std_dev_2 # element-wise operations

    return x_train, x_test

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize=(14,11))
    plt.title(title)
    sn.heatmap(df_cm, center=0, cmap=plt.cm.Blues, annot=True)


# In[2]:


# define the neural network structure

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 2))

        self.linear1 = nn.Linear(256*48, 1024)
        self.linear2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):

        output = self.conv1(input)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = output.view(-1, 256*48)
        
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.linear2(output)
        output = self.softmax(output)
        
        return output
    


# In[3]:


# prepare the data for cnn

# load iq data, for fft data use 'fft' instead of 'iq'
folder = '../data'
x_data, y_data = load_data(folder, 'fft 2422-2424 2429-2431')
y_data = y_data[:10, :, :]
# split data in training and test set
x_train, y_train, x_test, y_test = split_data(x_data, y_data, validation_fraction=0.33)
x_train, x_test = normalize_data(x_train, x_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train.strides)
print(y_train.strides)
print(x_test.strides)
print(y_test.strides)

if x_test.strides[2] < 0:
    x_test = np.flip(x_test, axis=2)
if y_test.strides[2] < 0:
    y_test = np.flip(y_test, axis=2)

x_train = torch.from_numpy(x_train).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
x_test = torch.from_numpy(x_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.long)

#########################################################################################
#SNR selection
#define SNR selection function
def snr_select(x_train, y_train, snr):
    x_train = x_train.contiguous().view(10,21,480,1,52,2)
    y_train = y_train.contiguous().view(10,21,480)
    i = snr//2 + 10
    x_train_1 = x_train[:,i,:,:,:,:]
    y_train_1 = y_train[:,i,:]
    print(x_train_1.shape)
    print(y_train_1.shape)
    return x_train_1, y_train_1

#Specify which snr value to select by directly changing the third
#argument in the following line
x_train_snr1, y_train_snr1 = snr_select(x_train,y_train,-2)
x_train = x_train_snr1
y_train = y_train_snr1

# for training

x_train_training = x_train.contiguous().view(10*1*480, 1, 52, 2)
y_train_training = y_train.contiguous().view(10*1*480)
x_test_training = x_test.contiguous().view(10*21*235, 1, 52, 2)
y_test_training = y_test.contiguous().view(10*21*235)

# for plot

x_test_plot = x_test.contiguous().view(10, 21, 235, 1, 52, 2)
y_test_plot = y_test

# for confusion matrix

x_test_confusion = x_test.contiguous().view(10, 21*235, 1, 52, 2)

# confusion matrix for each SNR

x_test_confusion_each_SNR = x_test.contiguous().view(10,21,235,1,52,2).permute(1,0,2,3,4,5)


# In[4]:


# train and test ResNet for recognizing WiFi signal

NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 4096
LR = 1e-4

best_val_accuracy = 0
best_val_loss = 100
number_epoch_until_best = 1
training_time = 0
training_time_until_best = 0
average_time_per_epoch = 0

train_dataloader = data.DataLoader(
    dataset=data.TensorDataset(x_train_training, y_train_training), 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

test_dataloader = data.DataLoader(
    data.TensorDataset(x_test_training, y_test_training), 
    batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

model = CNN()
model.cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch_idx in range(1, NUM_EPOCHS+1):
    progress_training_epoch = tqdm(
        IteratorTimer(train_dataloader), 
        desc=f'Epoch {epoch_idx}/{NUM_EPOCHS}, Training',
        miniters=1, ncols=88, position=0,
        leave=True, total=len(train_dataloader), smoothing=.9)
    progress_validation_epoch = tqdm(
        IteratorTimer(test_dataloader), 
        desc=f'Epoch {epoch_idx}/{NUM_EPOCHS}, Validation',
        miniters=1, ncols=88, position=0, 
        leave=True, total=len(test_dataloader), smoothing=.9)
    
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, (input, target) in enumerate(progress_training_epoch):
        batch_size = input.size()[0]
        input = input.cuda()
        target = target.cuda()
        target_onehot = torch.zeros((batch_size, 10), dtype=torch.float).cuda().scatter_(
            dim=1, index=target.view(batch_size, 1), value=1.0)
        output = model(input)
        batch_loss = criterion(output, target_onehot)
        batch_loss.backward()
        optimizer.step()
        model.zero_grad()
        train_loss += batch_size * batch_loss
    training_time += time.time() - start_time
       
    model.eval()
    val_loss = 0
    test_total_num_correct = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(progress_validation_epoch):
            batch_size = input.size()[0]
            input = input.cuda()
            target = target.cuda()
            target_onehot = torch.zeros((batch_size, 10), dtype=torch.float).cuda().scatter_(
                dim=1, index=target.view(batch_size, 1), value=1.0)
            output = model(input)
            batch_loss = criterion(output, target_onehot)  
            val_loss += batch_size * batch_loss
            test_total_num_correct += torch.eq(output.argmax(dim=1), target).sum()  

    val_accuracy = test_total_num_correct.item()/(10*21*235)
    val_loss = val_loss/(10*21*235)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        number_epoch_until_best_accuracy = epoch_idx
        training_time_until_best = training_time
        torch.save(model.state_dict(), './model/cnn-15classes.pth')
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        number_epoch_until_best_loss = epoch_idx
        
    print(f'epoch:{epoch_idx}, '
          f'training loss:{(train_loss/(10*21*480)): .5f}, '
          f'validation loss:{val_loss: .5f}, '
          f'accuracy: {val_accuracy: .4f}, '
          f'best accuracy: {best_val_accuracy: .4f}')

    if epoch_idx > number_epoch_until_best_accuracy+4 and epoch_idx > number_epoch_until_best_loss+4:
        break

model.load_state_dict(torch.load('./model/cnn-15classes.pth'))
model.cuda()
model.eval()

print(f'total training time: {training_time_until_best}')
print(f'number of epochs: {number_epoch_until_best_accuracy}')
print(f'time per epoch: {(training_time_until_best/number_epoch_until_best_accuracy): .2f}')

accuracy_each_class_each_SNR = np.zeros((10, 21))
model.eval()
with torch.no_grad():
    for i in range(10):
        for j in range(21):
            target = y_test_plot[i, j]
            input = x_test_plot[i, j].cuda()
            output = model(input).argmax(dim=1).cpu()
            num_correct = torch.eq(output, target).sum().item()
            accuracy_each_class_each_SNR[i, j] = num_correct / 235
acc_each_snr = list()
for j in range(21):
    total_acc = 0
    for i in range(10):
        total_acc += accuracy_each_class_each_SNR[i, j]
    total_acc /= 10
    acc_each_snr.append(total_acc)
print('Overall accuracy for each SNR:', acc_each_snr)

# In[5]:
"""
accuracy_each_class = np.zeros(10)
model.eval()
with torch.no_grad():
    for i in range(10):
        target = y_test_plot[i].contiguous().view(21*235)
        input = x_test_plot[i].cuda().contiguous().view(21*235, 1, 52, 2)
        output = model(input).argmax(dim=1).cpu()
        num_correct = torch.eq(output, target).sum().item()
        accuracy_each_class[i] = num_correct / (21 * 235)

print(accuracy_each_class[:6].mean())
print(accuracy_each_class[6:9].mean())
print(accuracy_each_class[9])
"""

# In[6]:

# plot accuracy vs SNR for each class
accuracy_each_class_each_SNR = np.zeros((10, 21))

plt.figure(figsize=(8, 8))
plt.xlim([-20, 20])
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 11))
plt.xlabel('SNR')
plt.ylabel('accuracy')
plt.grid(linestyle='--')

model.eval()
with torch.no_grad():
    for i in range(10):
        for j in range(21):
            target = y_test_plot[i, j]
            input = x_test_plot[i, j].cuda()
            output = model(input).argmax(dim=1).cpu()
            num_correct = torch.eq(output, target).sum().item()
            accuracy_each_class_each_SNR[i, j] = num_correct / 235

plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[0], 
    label='1', marker='o')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[1], 
    label='2', marker='v')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[2], 
    label='3', marker='^')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[3], 
    label='4', marker='<')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[4], 
    label='5', marker='>')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[5], 
    label='6', marker='1')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[6], 
    label='7', marker='2')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[7], 
    label='8', marker='3')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[8], 
    label='9', marker='4')
plt.plot(
    range(-20, 21, 2), accuracy_each_class_each_SNR[9], 
    label='10', marker='s')

plt.legend(loc='lower right')
plt.savefig("cnn-4MHz-line-each-SNR")
plt.show()

# In[7]:


# plot confusion matrix for all SNR
classes = ['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = np.zeros((10, 10))
confusion_matrix_norm = np.zeros((10, 10))

model.eval()
with torch.no_grad():
    for i in range(10):
        input = x_test_confusion[i].cuda()
        output= model(input).argmax(dim=1).cpu()
        confusion_matrix[i] = output.bincount(minlength=10)
    
confusion_matrix_norm = confusion_matrix / (21*235)
confusion_matrix_norm = (confusion_matrix_norm > 0.01) * confusion_matrix_norm
# plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_matrix_norm, labels=classes)
plt.savefig("cnn-4MHz-confusion-all")


# In[8]:


classes = ['1','2','3','4','5','6','7','8','9','10']
confusion_matrix = np.zeros((10, 10))
confusion_matrix_norm = np.zeros((10, 10))

model.eval()
with torch.no_grad():
    for j in range(21):
        confusion_matrix = np.zeros((10, 10))
        for i in range(10):
            input = x_test_confusion_each_SNR[j, i].cuda()
            output= model(input).argmax(dim=1).cpu()
            confusion_matrix[i] = output.bincount(minlength=10)
        confusion_matrix_norm = confusion_matrix / 235
        plot_confusion_matrix(confusion_matrix_norm, 
                              title=f'confusion matrix SNR={-20 + j*2}', 
                              labels=classes)
        plt.savefig(f"CNN-4MHz-confusion-SNR{-20 + j*2}")

