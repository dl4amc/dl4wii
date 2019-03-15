#!/usr/bin/env python
# coding: utf-8

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
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
torch.cuda.empty_cache()


dataset_path = '../data'
dataset_type = 'fft amplitude phase'
experiment_type = 
final_dimensions = 


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
    else:
        raise ValueError('Parameter which_kind must be "iq" for IQ-data or "fft" for FFT-data.')
    # load input data (x)
    data_path = os.path.join(storage_folder, data_file)
    with open(data_path, mode='rb') as storage: 
        x_data = pickle.load(storage, encoding='latin1')
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

    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 2))
        self.linear1 = nn.Linear(256*60, 1024)
        self.linear2 = nn.Linear(1024, 15)
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
        output = output.view(-1, 256*60)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output

def run_experiment(exp_type, exp_dims, x_train, x_test):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if exp_type == 'pcs':
        # PCS (Principal Component Subsampling) Setup
        from sklearn.decomposition import PCA
        pca_rate = exp_dims   # Number of samples after PCA
        pca = PCA(n_components=pca_rate*2)

        # PCS
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        x_train_new = x_train_new.transpose((2, 0, 1))
        x_train_new = np.append(x_train_new[0], x_train_new[1], axis=1)
        pca_apply = pca.fit(x_train_new)
        total_num_samples = (pca_apply.components_.shape[1])//2
        component_list = list()
        for component in pca_apply.components_:
            linear_coeff_list = list()
            for idx in range(total_num_samples):
                linear_coeff_list.append(component[idx]**2 + component[idx+total_num_samples]**2)
            component_list.append(linear_coeff_list)
        sample_list = list()
        for samp_idx in range(total_num_samples):
            total_mag = 0
            for comp in component_list:
                total_mag = comp[samp_idx]
            sample_list.append((samp_idx, total_mag))
        sample_list.sort(key=lambda x: x[1], reverse=True)
        sample_list = sample_list[:pca_rate]
        sample_list.sort(key=lambda x: x[0])
        sample_list = [sample_val[0] for sample_val in sample_list]
        x_train_new = x_train_new.transpose()
        x_train_new = np.array(np.split(x_train_new, 2))
        x_train_new = x_train_new.transpose((1, 0, 2))
        x_train_new = x_train_new[sample_list]
        x_train_new = x_train_new.transpose((2, 0, 1))
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        x_test_new = x_test_new.transpose((2, 0, 1))
        x_test_new = np.append(x_test_new[0], x_test_new[1], axis=1)
        x_test_new = x_test_new.transpose()
        x_test_new = np.array(np.split(x_test_new, 2))
        x_test_new = x_test_new.transpose((1, 0, 2))
        x_test_new = x_test_new[sample_list]
        x_test_new = x_test_new.transpose((2, 0, 1))
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif exp_type == 'autoencoder':
        # Autoencoder Setup
        ae_rate = exp_dims
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        x_train_new = x_train_new.transpose((2, 0, 1))
        x_train_new = np.append(x_train_new[0], x_train_new[1], axis=1)

        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        x_test_new = x_test_new.transpose((2, 0, 1))
        x_test_new = np.append(x_test_new[0], x_test_new[1], axis=1)

        input_dim = Input(shape = (256, ))
        encoding_dim = 2 * ae_rate
        # Encoder
        encoded1 = Dense(192, activation = 'relu')(input_dim)
        encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
        # Decoder
        decoded1 = Dense(192, activation = 'relu')(encoded2)
        decoded2 = Dense(256, activation = 'sigmoid')(decoded1)

        autoencoder = Model(input = input_dim, output = decoded2)
        autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
        autoencoder.fit(x_train_new, x_train_new, nb_epoch = 100, batch_size = 100, shuffle = True, validation_data = (x_test_new, x_test_new),
                        callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
        encoder = Model(input = input_dim, output = encoded2)
        encoded_input = Input(shape = (encoding_dim, ))

        x_train_new = encoder.predict(x_train_new)
        x_train_new = x_train_new.transpose()
        x_train_new = np.array(np.split(x_train_new, 2))
        x_train_new = x_train_new.transpose()
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        x_test_new = encoder.predict(x_test_new)
        x_test_new = x_test_new.transpose()
        x_test_new = np.array(np.split(x_test_new, 2))
        x_test_new = x_test_new.transpose()
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif exp_type == 'pca'
        # PCA Setup
        from sklearn.decomposition import PCA
        pca_rate = exp_dims
        pca = PCA(n_components=pca_rate*2)

        # PCA
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        x_train_new = x_train_new.transpose((2, 0, 1))
        x_train_new = np.append(x_train_new[0], x_train_new[1], axis=1)
        pca_apply = pca.fit(x_train_new)
        x_train_new = pca_apply.transform(x_train_new)
        x_train_new = x_train_new.transpose()
        x_train_new = np.array(np.split(x_train_new, 2))
        x_train_new = x_train_new.transpose()
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        x_test_new = x_test_new.transpose((2, 0, 1))
        x_test_new = np.append(x_test_new[0], x_test_new[1], axis=1)
        x_test_new = pca_apply.transform(x_test_new)
        x_test_new = x_test_new.transpose()
        x_test_new = np.array(np.split(x_test_new, 2))
        x_test_new = x_test_new.transpose()
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif exp_type == 'high_mag_subsampling'
        # Heuristic Sub Sampling
        n_samples = exp_dims
        new_X_train = list()
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        for wave_idx, wave in enumerate(x_train_new):
            amp_list = [(iq_idx, ((iq_val[0] ** 2) + (iq_val[1] ** 2) ** 0.5)) for iq_idx, iq_val in enumerate(wave)]
            amp_list.sort(key=lambda x: x[1], reverse=True)
            amp_list = amp_list[:n_samples]
            amp_list.sort(key=lambda x: x[0])
            amp_list = [amp_val[0] for amp_val in amp_list]
            wave = wave[amp_list]
            new_X_train.append(wave)
        x_train_new = np.stack(new_X_train)
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        new_X_test = list()
        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        for wave_idx, wave in enumerate(x_test_new):
            amp_list = [(iq_idx, ((iq_val[0] ** 2) + (iq_val[1] ** 2) ** 0.5)) for iq_idx, iq_val in enumerate(wave)]
            amp_list.sort(key=lambda x: x[1], reverse=True)
            amp_list = amp_list[:n_samples]
            amp_list.sort(key=lambda x: x[0])
            amp_list = [amp_val[0] for amp_val in amp_list]
            wave = wave[amp_list]
            new_X_test.append(wave)
        x_test_new = np.stack(new_X_test)
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif exp_type == 'uniform_subsampling':
        # Uniform Sub Sampling
        n_samples = exp_dims
        sample_idx = [num for num in range(0, 128, 128//n_samples)]
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        x_train_new = x_train_new.transpose((1, 0, 2))
        x_train_new = x_train_new[sample_idx]
        x_train_new = x_train_new.transpose((1, 0, 2))
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        x_test_new = x_test_new.transpose((1, 0, 2))
        x_test_new = x_test_new[sample_idx]
        x_test_new = x_test_new.transpose((1, 0, 2))
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif exp_type == 'random_subsampling':
        # Random Sub Sampling
        n_samples = exp_dims
        sample_idx = np.random.choice(range(0,128), size=n_samples, replace=False)
        x_train_new = np.concatenate(x_train, axis=0)
        x_train_new = np.concatenate(x_train_new, axis=0)
        x_train_new = x_train_new.transpose((1, 0, 2))
        x_train_new = x_train_new[sample_idx]
        x_train_new = x_train_new.transpose((1, 0, 2))
        x_train_new = np.array(np.split(x_train_new, 315))
        x_train_new = np.array(np.split(x_train_new, 15))

        x_test_new = np.concatenate(x_test, axis=0)
        x_test_new = np.concatenate(x_test_new, axis=0)
        x_test_new = x_test_new.transpose((1, 0, 2))
        x_test_new = x_test_new[sample_idx]
        x_test_new = x_test_new.transpose((1, 0, 2))
        x_test_new = np.array(np.split(x_test_new, 315))
        x_test_new = np.array(np.split(x_test_new, 15))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\nAfter application of PCA or subsampling:')
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print(x_train.strides)
        print(y_train.strides)
        print(x_test.strides)
        print(y_test.strides)
        return x_train_new, x_test_new
        

# load iq data, for fft data use 'fft' instead of 'iq'
folder = dataset_path
x_data, y_data = load_data(folder, dataset_type)
x_train1, y_train, x_test1, y_test = split_data(x_data, y_data, validation_fraction=0.33)
x_train, x_test = normalize_data(x_train1, x_test1)

print('Before application of PCA or subsampling:')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train.strides)
print(y_train.strides)
print(x_test.strides)
print(y_test.strides)

x_train, x_test = run_experiment(experiment_type, final_dimensions, x_train, x_test)

if x_test.strides[2] < 0:
	x_test = np.flip(x_test, axis=2)
if y_test.strides[2] < 0:
	y_test = np.flip(y_test, axis=2)

x_train = torch.from_numpy(x_train).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
x_test = torch.from_numpy(x_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.long)

#########################################################################################
# for training
x_train_training = x_train.contiguous().view(15*21*480, 1, final_dimensions, 2)
y_train_training = y_train.contiguous().view(15*21*480)
x_test_training = x_test.contiguous().view(15*21*235, 1, final_dimensions, 2)
y_test_training = y_test.contiguous().view(15*21*235)

# for plot
x_test_plot = x_test.contiguous().view(15, 21, 235, 1, final_dimensions, 2)
y_test_plot = y_test

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
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

test_dataloader = data.DataLoader(
    data.TensorDataset(x_test_training, y_test_training), 
    batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

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
        target_onehot = torch.zeros((batch_size, 15), dtype=torch.float).cuda().scatter_(
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
            target_onehot = torch.zeros((batch_size, 15), dtype=torch.float).cuda().scatter_(
                dim=1, index=target.view(batch_size, 1), value=1.0)
            output = model(input)
            batch_loss = criterion(output, target_onehot)  
            val_loss += batch_size * batch_loss
            test_total_num_correct += torch.eq(output.argmax(dim=1), target).sum()  

    val_accuracy = test_total_num_correct.item()/(15*21*235)
    val_loss = val_loss/(15*21*235)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        number_epoch_until_best_accuracy = epoch_idx
        training_time_until_best = training_time
#        torch.save(model.state_dict(), './model/cnn-15classes.pth')
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        number_epoch_until_best_loss = epoch_idx
        
    print(f'epoch:{epoch_idx}, '
          f'training loss:{(train_loss/(15*21*480)): .5f}, '
          f'validation loss:{val_loss: .5f}, '
          f'accuracy: {val_accuracy: .4f}, '
          f'best accuracy: {best_val_accuracy: .4f}')

    if epoch_idx > number_epoch_until_best_accuracy+4 and epoch_idx > number_epoch_until_best_loss+4:
        break

#model.load_state_dict(torch.load('./model/cnn-15classes.pth'))
model.cuda()
model.eval()

print(f'total training time: {training_time_until_best}')
print(f'number of epochs: {number_epoch_until_best_accuracy}')
print(f'time per epoch: {(training_time_until_best/number_epoch_until_best_accuracy): .2f}')

accuracy_each_class_each_SNR = np.zeros((15, 21))
model.eval()
with torch.no_grad():
    for i in range(15):
        for j in range(21):
            target = y_test_plot[i, j]
            input = x_test_plot[i, j].cuda()
            output = model(input).argmax(dim=1).cpu()
            num_correct = torch.eq(output, target).sum().item()
            accuracy_each_class_each_SNR[i, j] = num_correct / 235
acc_each_snr = list()
for j in range(21):
    total_acc = 0
    for i in range(15):
        total_acc += accuracy_each_class_each_SNR[i, j]
    total_acc /= 15
    total_acc *= 100
    acc_each_snr.append(total_acc)
print('Overall accuracy for each SNR:', acc_each_snr)

