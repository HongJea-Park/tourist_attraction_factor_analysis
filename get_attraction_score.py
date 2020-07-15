import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from visualization.visualization import visualize_factor_score


parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='../results/doc_term_mat.txt',
                    help='term document matrix file')
parser.add_argument('--vocab_file', default='../results/vocab.txt',
                    help='vocab file')
parser.add_argument('--results_folder', default='../results/seanmf_results',
                    help='seanmf results folder')
parser.add_argument('--n_topics', type=int, default=15,
                    help='number of topics')
parser.add_argument('--alpha', type=float, default=0.33,
                    help='alpha')
parser.add_argument('--beta', type=float, default=2.0,
                    help='beta')
args = parser.parse_args()

tmp_folder = \
    f'{str(args.results_folder)}_{args.n_topics}_{args.alpha}_{args.beta}'

H = np.loadtxt(f'../results/{tmp_folder}/H.txt')
H = H / H.sum(axis=1).reshape(-1, 1)

sentence_idx_list = []
with open('../results/sentence_idx.txt', 'r', -1, 'utf-8') as f:
    while True:
        line = f.readline()
        line = line.replace('\n', '')
        if not line:
            break
        sentence_idx_list.append(int(line))

H = H[sentence_idx_list, :]

with open('../results/en_core_web_md_selected_elasticnet.pkl', 'rb') as f:
    elastic = pickle.load(f)

sentence_df = pd.read_csv(
    '../results/sentence_df_selected.csv')
sentiment_df = pd.read_csv(
    '../results/en_core_web_md_sentiment_dictionary.csv')
sentiment_dictionary = list(sentiment_df['word'].values)
sentence_df['sentence'] = \
    sentence_df['sentence'].map(lambda x: ' '.join(
        [word for word in x.split() if word in sentiment_dictionary]))

reviews = sentence_df['sentence']

vect = CountVectorizer(vocabulary=sentiment_dictionary)
doc_term_matrix = vect.fit_transform(reviews, )
dictionary = vect.get_feature_names()
Tf = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False)
doc_term_matrix_tf = Tf.transform(doc_term_matrix)

sentiment_score = elastic.predict(doc_term_matrix_tf) - elastic.intercept_
topic_score = H*sentiment_score.reshape(-1, 1)

factor = {
    '경험': [14],
    '입장/대기시간': [4],
    '박물관/전시': [10, 12],
    '도시경관': [3],
    '전통건축': [13],
    '교통/이동수단': [2, 5],
    '안내서비스': [6],
    '자연경관/산책로': [1],
    '쇼핑': [11],
    '먹거리': [7],
    '쇼핑/먹거리': [0, 8, 9]}

factor_score = np.zeros((topic_score.shape[0], 11))
for i, value in enumerate(factor.values()):
    factor_score[:, i] = topic_score[:, value].sum(axis=1)
factor_score[:, 8] += factor_score[:, 10]/2
factor_score[:, 9] += factor_score[:, 10]/2
factor_score = factor_score[:, :10]
factor_score = pd.DataFrame().from_records(factor_score)
factor_score.columns = ['경험', '입장/대기시간', '박물관/전시', '도시경관',
                        '전통건축', '교통/이동수단', '안내서비스', '자연경관/산책로',
                        '쇼핑', '먹거리']
df = pd.concat([sentence_df[['spot', 'date']], factor_score], axis=1)
df['date'] = pd.to_datetime(df['date'].astype(str))
datelist = pd.date_range(
    start='20170101', end='20190701', freq=pd.offsets.MonthEnd())
monthly_df = pd.DataFrame()

for date in datelist:
    mean = df[df['date'] <= date].groupby(by='spot').mean()
    mean = mean.reset_index(drop=False)
    time_list = pd.Series([date for i in range(len(mean))], name='date')
    monthly_df = pd.concat(
        [monthly_df, pd.concat([time_list, mean], axis=1)], axis=0)
    monthly_df = monthly_df.reset_index(drop=True)

monthly_df['date'] = monthly_df['date'].dt.strftime('%Y%m')
monthly_df = monthly_df.set_index(['spot', 'date'])
monthly_df = monthly_df.reset_index()
attraction_sorted = df['spot'].value_counts().index.values

visualize_factor_score(monthly_df,
                       attraction_sorted[1],
                       drop_option=False)

visualize_factor_score(monthly_df,
                       attraction_sorted[2],
                       drop_option=True,
                       drop_values=(6, 4),
                       save_option=False,
                       filename=attraction_sorted[1])
