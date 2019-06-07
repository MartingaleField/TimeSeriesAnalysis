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
from pprint import pprint
import bs4 as bs
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


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
    for i in range(params['epochs']):
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
            print('[LOSS] Epoch {} : {}'.format(i + 1, loss))
        # log_value('loss', loss, i)
        losses['training'].append(loss)

        # print(torch.cuda.get_device_name(0))
        # wandb.log(losses)
    return encoder, losses


def move_nan_to_end(arr):
    arr2 = np.concatenate((arr[~np.isnan(arr)], np.full((len(arr) - np.sum(~np.isnan(arr)),), np.nan)))
    return arr2


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


def save_encoder(encoder, prefix_file):
    torch.save(encoder.state_dict(), prefix_file + '_encoder.pth')


def load_encoder(prefix_file, params):
    """
    Loads an encoder.

    @param prefix_file Path and prefix of the file where the model should
           be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
    """
    encoder = causal_cnn.CausalCNNEncoder(params['in_channels'], params['channels'], params['depth'],
                                          params['reduced_size'], params['out_channels'], params['kernel_size'])
    if params['cuda']:
        encoder.load_state_dict(torch.load(
            prefix_file + '_encoder.pth',
            map_location=lambda storage, loc: storage.cuda(params['gpu'])
        ))
    else:
        encoder.load_state_dict(torch.load(
            prefix_file + '_encoder.pth',
            map_location=lambda storage, loc: storage
        ))
    if params['cuda']:
        encoder.cuda(params['gpu'])
    encoder.double()
    return encoder


def calculate_ES(df, percentile=0.975):
    df -= df.mean()
    percentile = 1 - percentile + 1e-10
    ES = {}
    for col in df:
        hs = df[col].values
        shs = np.concatenate((hs, -hs))
        sorted_shs = np.sort(shs)
        EScutoff = int(np.floor(percentile * len(sorted_shs)))
        tails = sorted_shs[:EScutoff, ]
        ES[col] = np.average(tails)
    return pd.DataFrame.from_dict(ES, orient='index', columns=['ES'])


if __name__ == '__main__':
    hyper = r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\default_hyperparameters.json'
    save_path = r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\params'

    # tickers = ['GS', 'MSFT', 'MRK', 'MS', 'JPM', 'AMZN', 'NVDA', 'GOOG', 'JD']
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    tickers = list(map(lambda s: s.replace('\n', '').replace('.', '-'), tickers))

    start_date = '2016-01-01'
    end_date = '2018-12-31'

    # for tkr in tickers:
    #     file = r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\Data\{}.csv'.format(tkr)
    #     if os.path.exists(file) or tkr == 'DOW':
    #         continue
    #     else:
    #         print(tkr)
    #     tmp = data.DataReader(tkr, 'yahoo', start_date, end_date)['Adj Close']
    #     tmp.to_csv(file, index=True)

    tickers.remove('DOW')
    # df = data.DataReader(tickers, 'yahoo', start_date, end_date)['Adj Close']
    # df.to_csv(r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\Data\sp500.csv', index=True)
    df = pd.read_csv(r'D:\Dropbox\UNC\PYTHON\UnsupervisedScalableRepresentationLearningTimeSeries\Data\sp500.csv')
    df = df.set_index(pd.to_datetime(df['Date']))
    df = df.drop(['Date'], axis=1)
    df = df.dropna(how='all')
    df = np.log(1 + df.pct_change().dropna(how='all'))
    df = (df - df.mean()) / df.std()
    ES = calculate_ES(df)

    train = df.dropna(how='all').values.T
    for i in range(train.shape[0]):
        train[i, :] = move_nan_to_end(train[i, :])
    train = train.reshape(train.shape[0], 1, train.shape[1])

    with open(hyper, 'r') as hf:
        params = json.load(hf)
    params['batch_size'] = 16
    params['in_channels'] = np.shape(train)[1]
    params['cuda'] = True
    params['gpu'] = 0
    params['early_stopping'] = None
    params['out_channels'] = 160
    params['reduced_size'] = 80
    params['epochs'] = 50
    params['nb_random_samples'] = 20
    pprint(params)

    # encoder, losses = fit_encoder(train, params)
    # save_encoder(encoder, 'model_' + str(params['out_channels']))

    encoder = load_encoder('model_160', params)
    features = encode(train, encoder, params)

    # features = np.zeros((train.shape[0], params['out_channels']))
    # count = 0
    # df1 = df.copy()
    # for col in df1:
    #     ts = df1[col][:].dropna()
    #     ts = ts.values.T.reshape(1, 1, ts.size)
    #     features[count, :] = encode(ts, encoder, params)
    #     count += 1
    #     if count == features.shape[0]:
    #         break

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(range(len(losses['training'])), losses['training'])
    # plt.show()
    #
    # ids = [0, 1]
    # with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
    #     fig, ax = plt.subplots(2, 1, figsize=(25, 10))
    #     df2 = df.rolling(window=10).mean()
    #     ax[0].plot(df2.iloc[:, ids], lw=1)
    #     ax[1].plot(features[ids, :].T, marker='o', lw=1, ls='--')
    #     ax[0].legend(df.columns[ids], loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=len(ids), borderaxespad=0,
    #                  frameon=False)
    #     plt.show()
