'''
Multiprocessing doesn't work well on Windows in IPython console.
You can run your code in an external terminal(cmd) to have the results
you want.
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.linear_model import ElasticNet
from sklearn.metrics import confusion_matrix


def classify_positive(rating):

    if np.round(rating) >= 4.0:
        rating = 1
    else:
        rating = 0

    return rating


def cal_MSE(X, y, model):

    y_pred = model.predict(X)
    mse = np.mean((y.values - y_pred) ** 2)

    return mse


def cal_misclassification(X, y_sentiment, model):

    y_pred = pd.Series(model.predict(X))
    y_pred = y_pred.map(lambda x: classify_positive(x))
    misclassification = sum(np.abs(y_sentiment.values-y_pred))/len(y_sentiment)

    return misclassification


def elastic_train_cv(X, y, Kfold, log_alpha, l1_ratio, fit_intercept=True):

    y = pd.Series(y)
    y_sentiment = y.map(lambda x: classify_positive(x))

    misclassification_train_cv = []
    misclassification_valid_cv = []
    mse_train_cv = []
    mse_valid_cv = []

    for train_idx, valid_idx in Kfold.split(X):

        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        enet = ElasticNet(alpha=np.exp(log_alpha),
                          l1_ratio=l1_ratio,
                          fit_intercept=fit_intercept,
                          max_iter=1e+6,
                          normalize=False)

        elastic = enet.fit(X_train, y_train)

        mis_train = cal_misclassification(X=X_train,
                                          y_sentiment=y_sentiment[train_idx],
                                          model=elastic)

        mis_valid = cal_misclassification(X=X_valid,
                                          y_sentiment=y_sentiment[valid_idx],
                                          model=elastic)

        mse_train = cal_MSE(X=X_train, y=y_train, model=elastic)
        mse_valid = cal_MSE(X=X_valid, y=y_valid, model=elastic)

        misclassification_train_cv.append(mis_train)
        misclassification_valid_cv.append(mis_valid)
        mse_train_cv.append(mse_train)
        mse_valid_cv.append(mse_valid)

    result = {}

    result['log_alpha'], result['l1_ratio'] = log_alpha, l1_ratio
    result['misclassification_train'] = np.mean(misclassification_train_cv)
    result['misclassification_valid'] = np.mean(misclassification_valid_cv)
    result['MSE_train'] = np.mean(mse_train_cv)
    result['MSE_valid'] = np.mean(mse_valid_cv)
    result['num_of_coef_'] = len(elastic.coef_[elastic.coef_ != 0])

    print(f'CPU: {mp.current_process()}, '
          f'alpha: {result["log_alpha"]}, '
          f'l1_ratio: {result["l1_ratio"]}')
    print(f'misclassification_train: {result["misclassification_train"]:.10f}'
          f'  |  MSE_train: {result["MSE_train"]:.10f}')
    print(f'misclassification_train: {result["misclassification_valid"]:.10f}'
          f'  |  MSE_train: {result["MSE_valid"]:.10f}')
    print(f'Number of Not 0 coefficient: {result["num_of_coef_"]}')
    print(f'{"-"*94}\n')

    return result


def get_best_model(X, y, elastic_results, fit_intercept=True):

    measure_list = [result['MSE_valid'] for result in elastic_results]

    idx = np.argmin(measure_list)
    log_alpha, l1_ratio = \
        elastic_results[idx]['log_alpha'], elastic_results[idx]['l1_ratio']

    enet = ElasticNet(alpha=np.exp(log_alpha),
                      l1_ratio=l1_ratio,
                      fit_intercept=fit_intercept,
                      max_iter=1e+6,
                      normalize=False)

    elastic = enet.fit(X, y)

    return elastic


def get_sentiment_dictionary(dictionary, elastic):

    senti_dict = {}
    senti_dict['word'] = dictionary
    senti_dict['coef'] = elastic.coef_
    senti_dict = pd.DataFrame(senti_dict, columns=['word', 'coef'])

    return senti_dict


def sort_sentiment_dictionary(sentiment_dictionary):

    sentiment_dictionary_sorted = pd.concat([
        sentiment_dictionary.sort_values(
            by=['coef'], ascending=True).reset_index(drop=True),
        sentiment_dictionary.sort_values(
            by=['coef'], ascending=False).reset_index(drop=True)],
        axis=1,
        names=['negative', 'negative_coef', 'positive', 'positive_coef'])

    return sentiment_dictionary_sorted


def results_to_df(results):

    log_alpha = []
    l1_ratio = []
    misclassification_train = []
    misclassification_valid = []
    mse_train = []
    mse_valid = []
    num_of_coef_ = []

    for result in results:

        log_alpha.append(result['log_alpha'])
        l1_ratio.append(result['l1_ratio'])
        misclassification_train.append(result['misclassification_train'])
        misclassification_valid.append(result['misclassification_valid'])
        MSE_train.append(result['MSE_train'])
        MSE_valid.append(result['MSE_valid'])
        num_of_coef_.append(result['num_of_coef_'])

    df = {}
    df['log_alpha'] = log_alpha
    df['l1_ratio'] = l1_ratio
    df['misclassification_train'] = misclassification_train
    df['misclassification_valid'] = misclassification_valid
    df['MSE_train'] = MSE_train
    df['MSE_valid'] = MSE_valid
    df['num_of_coef_'] = num_of_coef_

    df = pd.DataFrame(
        df, columns=[
            'log_alpha', 'l1_ratio', 'misclassification_train',
            'misclassification_valid', 'MSE_train', 'MSE_valid',
            'num_of_coef_'])

    return df


def dictionary_evaluation(y_true, y_pred):

    c_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c_matrix.ravel()

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return c_matrix, accuracy, recall, precision, f1
