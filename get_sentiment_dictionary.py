import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import pickle
import argparse
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold

from sentiment_dictionary import elasticnet as en
from preprocessing.preprocessing import classify_positive
from visualization import visualization as vs


def check_model_in_list(x):

    model_list = ['md', 'sm']
    if x in model_list:
        return 'en_core_web_' + x
    else:
        raise ValueError('wrong model type')


def filter_word_list(review, word_list):
    temp = []
    for word in review.split():
        if word in word_list:
            temp.append(word)
    return ' '.join(temp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_cpu', type=int, default=-1,
        help='The number of CPU for multiprocessing')
    parser.add_argument(
        '--model', type=check_model_in_list, default='md',
        help='POS-Tagging Model list')

    args = parser.parse_args()

    if args.n_cpu == -1:
        n_cpu = mp.cpu_count()
    else:
        n_cpu = args.n_cpu

    filepath = '../results'
    list_dir = os.listdir(os.path.join(filepath, 'Preprocess'))
    files = [x for x in list_dir if '.csv' in x]

    df = pd.DataFrame()
    for file in files:
        temp = pd.read_csv(os.path.join(filepath, 'Preprocess', file))
        df = pd.concat([df, temp], axis=0)

    df = df.dropna()
    df = df.reset_index(drop=True)

    reviews = df[args.model]
    reviews = reviews.str.lower()
    reviews.is_copy = False

    vect = CountVectorizer()
    doc_term_matrix = vect.fit_transform(reviews)
    dictionary = vect.get_feature_names()

    doc_count = (doc_term_matrix != 0).sum(axis=0).tolist()[0]
    doc_count = dict(zip(dictionary, doc_count))
    word_count = doc_term_matrix.sum(axis=0).tolist()[0]
    word_count = dict(zip(dictionary, word_count))

    new_word_list = [word for word in dictionary if doc_count[word] >= 5]

    reviews = reviews.map(lambda x: filter_word_list(x, new_word_list))

    df['extracted'] = reviews
    df = df[df['extracted'].map(lambda x: len(str(x).split())) >= 10]
    df = df.reset_index(drop=True)
    df.to_csv(f'../results/{args.model}.csv', index=False)

    reviews = df['extracted']
    ratings = df['rating']

    vect = CountVectorizer()
    doc_term_matrix = vect.fit_transform(reviews)
    dictionary = vect.get_feature_names()
    Tf = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False)
    doc_term_matrix_tf = Tf.transform(doc_term_matrix)
    Kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    log_alpha_list = np.arange(-1, -16, -1)
    l1_ratio_list = [.01, .05, .1, .3, .5, .7, .9, .95, .99]
    hyper_parameter_list = list(product(*[log_alpha_list, l1_ratio_list]))

    mp.freeze_support()

    with mp.Pool(processes=n_cpu) as pool:
        results = pool.starmap(en.elastic_train_cv, hyper_parameter_list)

    with open(f'../results/{args.model}_elasticnet_result.pkl', 'wb') as f:
        pickle.dump(results, f)

    # with open(f'../results/{args.model}_elasticnet_result.pkl', 'rb') as f:
    #     results = pickle.load(f)

    elastic = en.get_best_model(doc_term_matrix_tf, ratings, results)

    y_sentiment = ratings.map(lambda x: classify_positive(x))

    mis = en.cal_misclassification(doc_term_matrix_tf, y_sentiment, elastic)
    mse = en.cal_MSE(doc_term_matrix_tf, ratings, elastic)
    print('-'*50)
    print('misclassificationen of best model: %.10f' % mis)
    print('MSE of best model: %.10f' % mse)

    y_sentiment_pred = pd.Series(elastic.predict(doc_term_matrix_tf))
    y_sentiment_pred = y_sentiment_pred.map(lambda x: classify_positive(x))
    c_matrix, accuracy, recall, precision, f1 = \
        en.dictionary_evaluation(y_sentiment, y_sentiment_pred)

    print('c_matrix: ', c_matrix)
    print('accuracy: %.10f' % accuracy)
    print('recall: %.10f' % recall)
    print('precision: %.10f' % precision)
    print('f1: %.10f' % f1)

    sentiment_dictionary = en.get_sentiment_dictionary(dictionary, elastic)
    sentiment_dictionary['word_count'] = \
        sentiment_dictionary['word'].map(lambda x: word_count[x])
    sentiment_dictionary['doc_count'] = \
        sentiment_dictionary['word'].map(lambda x: doc_count[x])

    sentiment_dictionary.to_csv(
        f'../results/{args.model}_sentiment_dictionary.csv', index=False)

    with open(f'../results/{args.model}_selected_elasticnet.pkl', 'wb') as f:
        pickle.dump(elastic, f)

    results_df = en.results_to_df(results)

    vs.elasticnet_performance(results_df,
                              False,
                              f'{args.model}_elasticnet_loss.png')
