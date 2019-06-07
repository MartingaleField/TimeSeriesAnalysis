from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scikit_wrappers import CausalCNNEncoderClassifier
import json
import os
import torch
from utils import Dataset
from tqdm import trange
from losses import triplet_loss
from networks import causal_cnn
from pandas.plotting import register_matplotlib_converters


# from tensorboard_logger import configure, log_value


# import wandb


def fit_encoder(X, params):
    varying = bool(np.isnan(np.sum(X)))

    train = torch.from_numpy(X)
    if params['cuda']:
        train = train.cuda(params['gpu'])

    train_torch_dataset = Dataset(X)
    train_generator = torch.utils.data.DataLoader(train_torch_dataset, batch_size=params['batch_size'], shuffle=True)
    encoder = causal_cnn.CausalCNNEncoder(params['in_channels'], params['channels'], params['depth'],
                                          params['reduced_size'], params['out_channels'], params['kernel_size'])
    if params['cuda']:
        encoder.cuda(params['gpu'])
    encoder.double()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])

    # configure("BasisAnalysis/run-epoch20", flush_secs=2)
    # wandb.init(project="BasisAnalysis")

    losses = {'training': []}
    for i in trange(params['epochs']):
        for batch in train_generator:
            if params['cuda']:
                batch = batch.cuda(params['gpu'])
            optimizer.zero_grad()
            if not varying:
                _loss = triplet_loss.TripletLoss(params['compared_length'], params['nb_random_samples'],
                                                 params['negative_penalty'])
            else:
                _loss = triplet_loss.TripletLossVaryingLength(params['compared_length'], params['nb_random_samples'],
                                                              params['negative_penalty'])
            loss = _loss(batch, encoder, train)
            loss.backward()
            optimizer.step()
        # log_value('loss', loss, i)
        losses['training'].append(loss)
        print(loss)
        # print(torch.cuda.get_device_name(0))
        # wandb.log(losses)
    return encoder, losses


def encode(X, encoder, params):
    varying = bool(np.isnan(np.sum(X)))

    test = Dataset(X)
    test_generator = torch.utils.data.DataLoader(test, batch_size=1)
    features = np.zeros((X.shape[0], params['out_channels']))
    encoder = encoder.eval()

    count = 0
    with torch.no_grad():
        if not varying:
            for batch in test_generator:
                if params['cuda']:
                    batch = batch.cuda(params['gpu'])
                features[count:(count + 1)] = encoder(batch).cpu()
                count += 1
        else:
            for batch in test_generator:
                if params['cuda']:
                    batch = batch.cuda(params['gpu'])
                length = batch.size(2) - torch.sum(torch.isnan(batch[0, 0])).data.cpu().numpy()
                features[count:(count + 1)] = encoder(batch[:, :, :length]).cpu()
                count += 1
    return features


def get_time_series(tickers, start, end, path):
    splitted = np.array_split(tickers, 50)

    ts = []
    for i, spt in enumerate(splitted):
        # print(spt)
        if not os.path.exists(os.path.join(path, str(i) + '.csv')):
            df = data.DataReader(spt, 'yahoo', start, end)['Adj Close']
            df.to_csv(path_or_buf=os.path.join(path, str(i) + '.csv'))
        else:
            df = pd.read_csv(filepath_or_buffer=os.path.join(path, str(i) + '.csv'))
        ts.append(df[:])
    ts = pd.concat(ts, axis=1)
    ts = ts.loc[:, ~ts.columns.duplicated()]
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts = ts.set_index(ts['Date'])
    ts = ts.drop(['Date'], axis=1)
    return ts


if __name__ == '__main__':
    register_matplotlib_converters()

    hyper = r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\default_hyperparameters.json'
    save_path = r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\params'

    tickers = pd.read_pickle('sp500tickers.pickle')
    tickers = list(map(lambda s: s.replace('.', '-'), tickers))
    tickers = [x for x in tickers if x != 'DOW']
    # tickers = np.array(['GS', 'MSFT', 'MRK', 'MS', 'JPM', 'AMZN', 'NVDA', 'GOOG'])
    start_date = '2016-01-01'
    end_date = '2018-12-31'

    # for tkr in ['ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR']:
    #     print(tkr)
    #     df = data.DataReader(tkr, 'yahoo', start_date, end_date)['Adj Close']

    df = get_time_series(tickers, start_date, end_date,
                         r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\Data\time_series')

    df = df.dropna(how='all')
    df = np.log(1 + df.pct_change().dropna(how='all'))

    # df['GS 2'] = df['GS']
    # idx = []
    # for i in range(6):
    #     idx += list(range(100 * i, 100 * i + 10))
    # df.iloc[idx, 1] = np.nan

    # df['GS 3'] = df['GS'].shift(20)

    # df = (df - df.min()) / (df.max() - df.min())

    df = (df - df.mean()) / df.std()

    train = df.dropna(how='any').values.T
    train = train.reshape(train.shape[0], 1, train.shape[1])

    with open(hyper, 'r') as hf:
        params = json.load(hf)

    params['in_channels'] = np.shape(train)[1]
    params['cuda'] = True
    params['gpu'] = 0
    params['early_stopping'] = None
    params['out_channels'] = 80
    params['reduced_size'] = 40
    params['epochs'] = 500
    params['depth'] = 10
    params['batch_size'] = 16
    params['lr'] = 0.0005

    print(params)

    encoder, losses = fit_encoder(train, params)

    features = np.zeros((train.shape[0], params['out_channels']))
    count = 0
    df1 = df.copy()
    for col in df1:
        ts = df1[col][:].dropna()
        ts = ts.values.T.reshape(1, 1, ts.size)
        features[count, :] = encode(ts, encoder, params)
        count += 1
        if count == features.shape[0]:
            break

    with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(losses['training'])), losses['training'])
        plt.show()

    subset = ['GS', 'MRK']
    with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        df2 = df[subset].rolling(window=10).mean()
        ax[0].plot(df2, lw=2)
        ax[0].legend(df2.columns, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3,
                     borderaxespad=0, frameon=False)
        # ax[0].legend(df2.columns)
        ax[1].plot(features[np.where(np.isin(tickers, subset))].T, ls='--', marker='o', lw=2)
        # ax[1].legend(df2.columns)
        plt.show()
