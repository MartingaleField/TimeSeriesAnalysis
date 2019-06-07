import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, model_selection, metrics
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
import json
from pprint import pprint
from sp500 import calculate_ES, move_nan_to_end, load_encoder, encode
import torch
from utils import Dataset
from termcolor import colored


def modelfit(alg, X_tr, y_tr, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(data=X_tr, label=y_tr)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(X_tr, y_tr, eval_metric='rmse')

    dtrain_predictions = alg.predict(X_tr)

    print("\nModel Report")
    print("Accuracy : {:.4g}".format(np.sqrt(metrics.mean_squared_error(y_tr, dtrain_predictions))))
    pprint(alg.get_params())

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Feature Importance Score')


def check_tuning_boundary(param_grid, best_param):
    conditions = []
    for key in param_grid.keys():
        conditions += [
            best_param[key] == param_grid[key][0] and best_param[key] != 0,
            best_param[key] == param_grid[key][-1]
        ]
    if any(conditions):
        raise Exception("Reached boundaries. Needs further tuning.")
    else:
        print("Boundary checking cleared.")


def prediction_performance(data):
    ES_df = pd.DataFrame(data)
    ES_df['Lower (10%)'] = ES_df['True ES'] * 1.1
    ES_df['Upper (10%)'] = ES_df['True ES'] * 0.9
    ES_df['Lower (5%)'] = ES_df['True ES'] * 1.05
    ES_df['Upper (5%)'] = ES_df['True ES'] * 0.95
    ES_df['Accept (10%)'] = (ES_df['Lower (10%)'] <= ES_df['Pred ES']) & (ES_df['Pred ES'] <= ES_df['Upper (10%)'])
    ES_df['Accept (5%)'] = (ES_df['Lower (5%)'] <= ES_df['Pred ES']) & (ES_df['Pred ES'] <= ES_df['Upper (5%)'])
    ES_df = ES_df.sort_values(by=['True ES'])
    ES_df = ES_df.reset_index()
    ES_df = ES_df.drop(['index'], axis=1)
    return ES_df


if __name__ == '__main__':
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

    encoder_model = 'CausalCNNEncoderl160'
    if encoder_model == 'CausalCNNEncoderl160':
        params_file = 'hyper_CausalCNNEncoderl160.json'
    else:
        params_file = 'hyper_CausalCNNEncoderl80.json'
    params = json.load(open(params_file, 'r'))
    pprint(params)

    encoder = load_encoder(encoder_model, params)

    features = encode(train, encoder, params)

    X = features
    Y = ES.values

    # keep = np.where(Y < -2.4)[0]
    # X = X[keep, :]
    # Y = Y[keep]

    '''
    Train-test split with stratification
    '''
    # bins = np.linspace(0, len(Y), 50)
    # y_binned = np.digitize(np.linspace(0, len(Y) - 1, len(Y)), bins)
    # X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, Y, test_size=0.1, stratify=y_binned, random_state=42)

    X_tr, X_te, y_tr, y_te = X, X, Y, Y

    '''
    Tunning
    '''
    xgb_params = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'seed': 27,
        'n_jobs': -1
    }
    xgb_reg = xgb.XGBRegressor(**xgb_params)
    modelfit(xgb_reg, X_tr, y_tr)

    ####################################################################################################################

    params_test1 = {
        'max_depth': range(1, 4, 1),
        'min_child_weight': range(8, 11, 1)
    }
    gsearch = model_selection.GridSearchCV(
        estimator=xgb_reg,
        param_grid=params_test1,
        scoring='neg_mean_squared_error',
        iid=False,
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    gsearch.fit(X_tr, y_tr)
    pprint(gsearch.best_params_)
    check_tuning_boundary(params_test1, gsearch.best_params_)

    ####################################################################################################################

    params_test2 = {
        'gamma': [i / 10 for i in range(5)]
    }
    gsearch.set_params(estimator=gsearch.best_estimator_, param_grid=params_test2)
    gsearch.fit(X_tr, y_tr)
    pprint(gsearch.best_params_)
    check_tuning_boundary(params_test2, gsearch.best_params_)

    ####################################################################################################################

    xgb_reg = gsearch.best_estimator_
    xgb_reg.set_params(n_estimators=1000)
    modelfit(xgb_reg, X_tr, y_tr)

    ####################################################################################################################

    params_test3 = {
        'subsample': [i / 10 for i in range(1, 11)],
        'colsample_bytree': [i / 10 for i in range(1, 11)]
    }
    gsearch.set_params(estimator=xgb_reg, param_grid=params_test3)
    gsearch.fit(X_tr, y_tr)
    pprint(gsearch.best_params_)
    check_tuning_boundary(params_test3, gsearch.best_params_)

    ####################################################################################################################

    params_test4 = {
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1],
        # 'reg_lambda': [0.1, 1, 10, 100]
    }
    gsearch.set_params(estimator=gsearch.best_estimator_, param_grid=params_test4)
    gsearch.fit(X_tr, y_tr)
    pprint(gsearch.best_params_)
    check_tuning_boundary(params_test4, gsearch.best_params_)

    ####################################################################################################################

    xgb_reg = gsearch.best_estimator_
    xgb_reg.set_params(learning_rate=0.01, n_estimators=10000)
    modelfit(xgb_reg, X_tr, y_tr)

    ####################################################################################################################

    # xgb_reg.fit(X_tr, y_tr)
    if encoder_model == 'CausalCNNEncoderl160':
        xgboost_model = 'xgboost_vec160_sp500.pickle.dat'
    else:
        xgboost_model = 'xgboost_vec80_sp500.pickle.dat'
    pickle.dump(xgb_reg, open(xgboost_model, 'wb'))

    xgb_reg = pickle.load(open(xgboost_model, 'rb'))
    fitted = xgb_reg.predict(X_tr)
    rmse = np.sqrt(metrics.mean_squared_error(y_tr, fitted))
    pprint('RMSE: {}'.format(rmse))

    # results = xgb_reg.evals_result()
    # epochs = len(results['validation_0']['rmse'])
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.plot(range(0, epochs), results['validation_0']['rmse'], label='Train')
    # ax.plot(range(0, epochs), results['validation_1']['rmse'], label='Test')
    # ax.legend()
    # fig.show()

    lr = linear_model.LinearRegression()
    lr.fit(y_tr, fitted)
    y_pred = lr.predict(y_tr)
    print('R2: {}'.format(lr.score(y_tr, fitted)))
    with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.scatter(y_tr, fitted, alpha=0.5)
        ax.plot(y_tr, y_pred, 'black')
        ax.set_xlabel('True ES')
        limit = [min(np.min(y_tr), np.min(y_pred)) - 0.1, max(np.max(y_tr), np.max(y_pred)) + 0.1]
        ax.set_xlim(limit)
        ax.set_ylabel('Pred ES')
        ax.set_ylim(limit)
        ax.plot(np.linspace(limit[0], limit[1], 10), np.linspace(limit[0], limit[1], 10), 'red', ls='--', lw=1)
        ax.set_title('R2: {:.4f}'.format(lr.score(y_tr, fitted)))
        fig.show()

    data = {'True ES': y_tr.squeeze(1), 'Pred ES': fitted}
    ES_df = prediction_performance(data)
    print("Accuracy with 10% tolerance: {:.2f}%".format(ES_df['Accept (10%)'].sum() / ES_df.shape[0] * 100))
    print("Accuracy with 5% tolerance: {:.2f}%".format(ES_df['Accept (5%)'].sum() / ES_df.shape[0] * 100))

    with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.fill_between(ES_df.index, ES_df['Lower (10%)'], ES_df['Upper (10%)'], alpha=0.2, color='grey')
        ax.fill_between(ES_df.index, ES_df['Lower (5%)'], ES_df['Upper (5%)'], alpha=0.4, color='grey')
        # ax.plot(ES_df[['True ES', 'Pred ES']], marker='o', markersize=7, alpha=0.7)
        ax.scatter(ES_df['True ES'].index, ES_df['True ES'].values, s=20, alpha=0.6, color='red')
        ax.scatter(ES_df['Pred ES'].index, ES_df['Pred ES'].values, s=20, alpha=0.6, color='navy')
        ax.legend(['10% Error Band', '5% Error Band', 'True ES', 'Pred ES'], loc='upper left')
        ax.set_title('Accuracy with 10% tolerance: {:.2f} %\nAccuracy with 5% tolerance: {:.2f} %'
                     .format(ES_df['Accept (10%)'].sum() / ES_df.shape[0] * 100,
                             ES_df['Accept (5%)'].sum() / ES_df.shape[0] * 100))
        fig.show()

    ####################################################################################################################

    preds = xgb_reg.predict(X_te)
    rmse = np.sqrt(metrics.mean_squared_error(y_te, preds))
    pprint('RMSE: {}'.format(rmse))

    lr = linear_model.LinearRegression()
    lr.fit(y_te, preds)
    y_pred = lr.predict(y_te)
    print('R2: {}'.format(lr.score(y_te, preds)))
    plt.scatter(y_te, preds, color='black')
    plt.plot(y_te, y_pred, color='blue', lw=3)
    plt.show()

    data = {'True ES': y_te.squeeze(1), 'Pred ES': preds}
    ES_df = prediction_performance(data)
    print("Accuracy with 10% tolerance: {:.2f}%".format(ES_df['Accept (10%)'].sum() / ES_df.shape[0] * 100))
    print("Accuracy with 5% tolerance: {:.2f}%".format(ES_df['Accept (5%)'].sum() / ES_df.shape[0] * 100))

    with plt.style.context(('seaborn-ticks', 'seaborn-talk')):
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.fill_between(ES_df.index, ES_df['Lower (10%)'], ES_df['Upper (10%)'], alpha=0.2)
        ax.fill_between(ES_df.index, ES_df['Lower (5%)'], ES_df['Upper (5%)'], alpha=0.2)
        ax.plot(ES_df[['True ES', 'Pred ES']], marker='o', markersize=7, alpha=0.7)
        ax.legend(['True ES', 'Pred ES', '10% Error Band', '5% Error Band'], loc='upper left')
        fig.show()
