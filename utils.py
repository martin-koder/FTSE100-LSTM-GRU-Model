#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from sklearn.model_selection  import train_test_split
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import seaborn as sns

#formatting settings common to all plots
plt.rcParams["font.family"] = "serif"

def create_returns(X,y, log_return = True):#create returns and take the log to normalise for unbounded data 
    X_returns = []
    y_returns = []
    for i in range(len(X)-1):
        X_return = X[i+1] - X[i]
        if log_return:
            X_return = np.log(X[i+1])- np.log(X[i])
        X_returns.append(X_return)
        y_return = y[i+1] - y[i]
        if log_return:
            y_return = np.log(y[i+1])- np.log(y[i])
        y_returns.append(y_return)
    X_returns = np.array(X_returns)
    y_returns = np.array(y_returns)
    
    return X_returns, y_returns
    
def create_sequences(X,y,seq_length): # function to create sequences
    new_X=[]
    for i in range(len(X)-seq_length):
        new_X.append(X[i:i+seq_length])
    new_y=y[seq_length:]
    new_X=np.array(new_X)
    assert new_X.shape[0]==new_y.shape[0]
    return new_X, new_y

#no need for a complicated dataset class
class MyDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y   
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
    def __len__ (self):
        return len(self.X)

from torch.nn import Sequential, Linear, Conv1d, MaxPool1d, LSTM, Dropout

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class LSTM(nn.Module):
    def __init__(self, device, input_size=1, hidden_layer_size=100, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1, batch_first=True) #MK batch_first

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq) # self.hidden_cell is unecessary
        lstm_out=lstm_out.to(device)
        lstm_out=lstm_out[:,-1,:]  #take the last one
        predictions = self.linear(lstm_out.reshape(len(input_seq), -1)) 

        return predictions

class GRU(nn.Module):
    def __init__(self, device, input_dim=1, hidden_dim=100, num_layers=1, output_dim=1):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0=h0.to(device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

#made generic train function, takes net as arg (can use for either LSTM or GRU)
#fixed error printing so it conforms to saved history values (i.e mean of train/valid loss))
def train(net, train_loader, valid_loader, n_epochs, device, optimizer, loss_function):

    loss_history=[]
    valid_history=[]
    
    for epoch in range(n_epochs):
        loss_epoch=0

        for X_batch,y_batch in train_loader:
            if device == torch.device('cuda'):
                X_batch,y_batch  = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output=net.forward(X_batch)#
            loss = loss_function(output, y_batch)
            loss.backward()#(retain_graph=True) 
            optimizer.step()

            loss_epoch+=loss.detach().cpu().numpy()
        loss_history.append(loss_epoch/len(train_loader))#take the mean so we can compare with valid_loss
        # Validation phase

        with torch.no_grad():
            val_loss_epoch=0
            for X_val,y_val in valid_loader:
                X_val,y_val  = X_batch.to(device), y_batch.to(device)
                output = net.forward(X_val)
                val_loss = loss_function(output, y_val)
                val_loss_epoch+=val_loss.cpu().numpy()#don't need detach() because already no_graded.
            valid_history.append(val_loss_epoch/len(valid_loader))
            print('Epoch=',epoch, 'Mean Squared Error = ', loss_epoch/len(train_loader), 'Valid Mean Squared Error = ',                 val_loss_epoch/len(valid_loader))

    return loss_history, valid_history

#loss plotting function
def plot_loss(loss_history, valid_history):
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.plot(loss_history, label='Training')
    plt.plot(valid_history, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Mean loss')
    plt.legend()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Training & validation losses come down steadily')
    plt.tight_layout()

def plot_loss_compare(lstm_loss, lstm_loss_3, gru_loss, gru_loss_3):
    plt.plot(lstm_loss, label='LSTM')
    plt.plot(lstm_loss_3, label='GRU')
    plt.plot(gru_loss, label='LSTM_3')
    plt.plot(gru_loss_3, label='GRU_3')
    plt.xlabel('Epochs')
    plt.ylabel('Mean loss')
    plt.legend()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('\n Comparison of validation loss over epochs')
    plt.tight_layout()
    
    
def calc_test_loss(net, test_loader, device):
    all_predictions = torch.tensor([]).to(device)
    net.eval()
    loss_function = nn.MSELoss() 
    for X_batch,y_batch in test_loader:
            X_batch,y_batch  = X_batch.to(device), y_batch.to(device)
            output=net.forward(X_batch)#
            loss = loss_function(output, y_batch)
            loss=loss.detach().cpu().numpy()
            all_predictions = torch.cat((all_predictions, output), dim=0)
    #net.train()
    print('Mean Squared Error = ', loss)
    return loss
#calc_test_loss(lstm_net, test_loader)

def test_confusion_matrix(net, test_loader, device):
    all_predictions = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            X_batch,y_batch  = X_batch.to(device), y_batch.to(device)
            output=net.forward(X_batch)
            all_predictions = torch.cat((all_predictions, output), dim=0)
            labels = torch.cat((labels, y_batch), dim=0)
        
        predictions_iod = torch.tensor([1 if x>0 else 0 for x in all_predictions])
        y_test_iod = torch.tensor([1 if x>0 else 0 for x in labels])

        stacked = torch.stack((predictions_iod,y_test_iod), dim=1)
        confusion_matrix = torch.zeros([2,2])
        for [pred, label] in stacked:
            confusion_matrix[pred][label]+= int(1)
        return confusion_matrix
    
def test_confusion_matrix_baseline(test_loader,device):
    all_predictions = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    retrn_distrb = torch.tensor(df['Close*'].pct_change().dropna().values)

    with torch.no_grad():
        for _,y_batch in test_loader:
            _,y_batch  = _, y_batch.to(device)
            output=torch.tensor(np.random.choice(retrn_distrb, b_size)).to(device).view(b_size,1)
            all_predictions = torch.cat((all_predictions, output), dim=0)
           
            labels = torch.cat((labels, y_batch), dim=0)
        all_predictions = torch.narrow(all_predictions, 0, 0, 1723).view(-1)#need to reduce size by 5 as last batch is cropped
        predictions_iod = torch.tensor([1 if x>0 else 0 for x in all_predictions])
        y_test_iod = torch.tensor([1 if x>0 else 0 for x in labels])

        stacked = torch.stack((predictions_iod,y_test_iod), dim=1)
        confusion_matrix = torch.zeros([2,2])
        for [pred, label] in stacked:
            confusion_matrix[pred][label]+= int(1)
        return confusion_matrix
    
def plot_conf_matrix(cm, y_test):
    #calculate accuracy
    accuracy = (sum(cm.diag())/len(y_test)).detach().item()
    accuracy = round(accuracy*100,1)
    print('Overall accuracy = ', accuracy,'%')

    # sns.heatmap(cm, vmin = 0, annot = True)#, fmt = "d")
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize = (3,3))
    ax = sns.heatmap(cm, annot=True, annot_kws={"size": 20}, fmt='.4g', cmap='gray', linewidths=1, 
                     linecolor='black', xticklabels=['Predicted \n Up', 'Predicted \n Down'],
                     yticklabels=['True\nUp ', ' True\nDown'],
                     cbar=False)#reduce the decimal digits displayed for greater visibility
    ax.tick_params(left=False, bottom=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_title('Overall accuracy\n= '+str(accuracy)+'%\n')

    #per class accuracy
    pc_acc = (cm.diag()/cm.sum(1)).detach()
    print('Predicted up accuracy =', round(pc_acc[0].item()*100,1),'%')
    print('Predicted down accuracy =', round(pc_acc[1].item()*100, 1),'%') 
    
    
# autoregression function allows for forward predictions
def autoregression(net, X, n_preds, device):
    y_preds=[]
    for i in range(n_preds):
        y_pred=net(X.to(device)).detach().cpu().unsqueeze(1)# give extra dim to y to match X
        y_preds.append(y_pred.squeeze().numpy())
        X = X[:,1:] #chop the last X of the sequence, then put the y_pred in its place for the next loop
        X = torch.cat([X, y_pred], dim=1) # dim1 concat on seq_len 
    return np.array(y_preds).T

# https://stackoverflow.com/questions/55892584/convert-log-returns-to-actual-price-of-a-time-series-forecast-using-r
def convert_returns_to_prices(returns):
    prices  = np.exp(np.cumsum(returns, 1))
    return prices


def create_example_returns_prediction(model): # specify LSTM or GRU
    #extract some data to compare autogression pred with true values
    auto_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),shuffle=True) # using the testset data
    X,y_true =next(iter(auto_loader)) # X dims is batch, seq len, num feats  # get first item # or list the iter then idx
    y_preds = autoregression(model, X, n_preds, device) # choosing n_preds worth of predictions
    #concatenate pred and true data with X sequence used as input
    preds_data = np.concatenate([X[0].numpy().squeeze(),y_preds[0]])# take the first example
    true_data = np.concatenate([X[0].numpy().squeeze(),y_true[:n_preds].squeeze()]) # irst X corresponds to first 10 y
    
    return preds_data, true_data

def plot_returns(preds_data, true_data, rolling_av_rebased):
    #plot predicted and true returns
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.plot(preds_data, alpha=0.5, label='Model return preds')
    plt.plot(true_data,alpha =0.5, label='True returns') # depends if y is shuffled
    plt.plot(rolling_av_rebased,label='Rolling av returns')
    plt.title('Predicted and true RETURNS\n')
    plt.legend()
    plt.xlabel('\nTrading days\n \n(Sequence len=25, n_preds=20, returns rebased to 1)')

def create_price_predictions(preds_data, true_data):
    preds_data_reshaped = preds_data.reshape(1, number_sequences+n_preds)# because convert_returns_to_prices assumes a batch input
    true_data_reshaped = true_data.reshape(1, number_sequences+n_preds) 
    pred_prices = convert_returns_to_prices(preds_data_reshaped)
    true_prices = convert_returns_to_prices(true_data_reshaped)
    retrn = random_walk()
    retrn[:number_sequences] = preds_data[:number_sequences]
    random_wlk = retrn.reshape(1, number_sequences+n_preds)
    random_wlk = convert_returns_to_prices(random_wlk)
 
    return pred_prices, true_prices, random_wlk

def plot_prices(pred_prices, true_prices, random_wlk):
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    arr = np.empty(number_sequences-1)
    arr[:] = np.NaN
    random_wlk[0][:number_sequences-1] = arr
    pred_prices[0][:number_sequences-1] = arr
    plt.plot(true_prices[0], label='True prices')
    plt.plot(pred_prices[0], color='green', label='Model preds')
    plt.plot(random_wlk[0], label='Random walk')
    plt.legend()
    plt.title('PRICES: Model predictions compared to true values\n and random walk baseline\n')
    plt.xlabel('\nTrading days\n \n(Sequence len=25, n_preds=20, prices rebased to 1)')

def random_walk():
    retrn_distrb = df['Close*'].pct_change().dropna().to_numpy()
    retrn = np.random.choice(retrn_distrb, n_preds)
    arr = np.empty(number_sequences)
    arr[:] = np.NaN
    retrn = np.insert(retrn,0,arr)
    return retrn

def preds_versus_randwlk(model, n_low, n_high, n_samples):
    mse_preds=[]
    mse_random_wlks=[]
    for n in range(n_low,n_high):
        n_preds=n
        pred_prices_lst=[]
        true_prices_lst=[]
        random_wlk_lst=[]
        for i in range(n_samples):
            preds_data, true_data = create_example_returns_prediction(model)
            pred_prices, true_prices, random_wlk = create_price_predictions(preds_data, true_data)
            pred_prices_lst.append(pred_prices.reshape(-1)[-1])
            true_prices_lst.append(true_prices.reshape(-1)[-1])
            random_wlk_lst.append(random_wlk.reshape(-1)[-1])
        mse_pred = mean_squared_error(true_prices_lst, pred_prices_lst)
        mse_random_wlk = mean_squared_error(true_prices_lst, random_wlk_lst)
        mse_preds.append(mse_pred)
        mse_random_wlks.append(mse_random_wlk)
    return mse_preds, mse_random_wlks

def plot_preds_versus_randwlk(mse_preds,mse_random_wlks):
    plt.plot(mse_preds, color='green', label='Model preds')
    plt.plot(mse_random_wlks, label='Random walk preds')
    plt.xlabel('n_preds (days)\n \n(Averaged MSE over n specified example sequences of length 25)')
    plt.ylabel('MSE vs true prices')
    plt.legend()
    plt.title('Model improves its margin over random walk\n as the prediction horizon increases\n')