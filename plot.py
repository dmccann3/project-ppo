import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


def avg_std(data):
    print(len(data))
    avg = np.zeros(len(data))
    std = np.zeros(len(data))
    sums = np.zeros(len(data))

    for i in range(len(data)):
        avg[i] = np.mean(data[i])
        std[i] = np.std(data[i])
        sums[i] = np.sum(data[i])

    ubounds = avg+std
    lbounds = avg-std
    
    return avg, lbounds, ubounds, sums

def write_to_csv(training_returns, training_losses_A, training_losses_C, ret_filename, loss_filename_A, loss_filename_C):
    return_data, ret_lbounds, ret_ubounds, sums = avg_std(training_returns)
    print(return_data)
    print(ret_lbounds)
    print(ret_ubounds)

    returns_df = pd.DataFrame({'avg':return_data, 'lbound':ret_lbounds, 'ubound':ret_ubounds, 'sums':sums})
    losses_df_A = pd.DataFrame({'losses':training_losses_A})
    losses_df_C = pd.DataFrame({'losses':training_losses_C})

    returns_df.to_csv(ret_filename)
    losses_df_A.to_csv(loss_filename_A)
    losses_df_C.to_csv(loss_filename_C)

def plot_testing(return_data, filename):
    print(return_data)
    episodes = np.linspace(0, len(return_data), len(return_data))

    fig, (ax) = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(episodes, return_data, color='red', label='R')
    ax.set_xlabel('Episodes', size=10)
    ax.set_ylabel('Return', size=10)
    ax.set_title('Total Returns over 100 Evaluations', size=12)
    ax.set_xbound(0.0)
    ax.grid()
    ax.legend()
    fig.savefig(filename)


def plot_returns(data_file, figure_file):
    ret_df = pd.read_csv(data_file)

    avgs = ret_df.loc[:,'avg']
    ubounds = ret_df.loc[:,'lbound']
    lbounds = ret_df.loc[:,'ubound']
    episodes = np.linspace(7524, 7524+len(avgs), len(avgs))

    
    fig, (ax) = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(episodes, avgs, color='blue', label='avg.')
    ax.set_xlabel('Episodes', size=10)
    ax.set_ylabel('Return', size=10)
    ax.set_title('Average Return', size=12)
    ax.grid()
    ax.set_ybound(0.0)
    ax.legend()
    fig.savefig(figure_file)

def plot_loss(data_file, figure_file, title, color):
    loss_data = pd.read_csv(data_file)

    loss = loss_data.loc[:,'losses']
    episodes = np.linspace(0, len(loss), len(loss))


    fig, (ax) = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(episodes, loss, color=color, label='L')
    ax.set_xlabel('Episodes', size=10)
    ax.set_ylabel('Loss', size=10)
    ax.set_title(title, size=12)
    ax.grid()
    ax.legend()
    fig.savefig(figure_file)








