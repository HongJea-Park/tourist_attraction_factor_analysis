'''
Visualize Topics
refer : https://github.com/tshi04/SeaNMF
'''
import pandas as pd
import argparse
import numpy as np

from SeaNMF.utils import read_docs, read_vocab, calculate_PMI

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='../results/doc_term_mat.txt',
                    help='term document matrix file')
parser.add_argument('--vocab_file', default='../results/vocab.txt',
                    help='vocab file')
parser.add_argument('--results_folder', default='../results/seanmf_results',
                    help='seanmf results folder')
parser.add_argument('--n_topics', type=int, default=15,
                    help='number of topics')
parser.add_argument('--n_top', type=int, default=20,
                    help='number of top keywords')
parser.add_argument('--alpha', type=float, default=0.33,
                    help='alpha')
parser.add_argument('--beta', type=float, default=2.0,
                    help='beta')
args = parser.parse_args()

docs = read_docs(args.corpus_file)
vocab = read_vocab(args.vocab_file)
n_docs = len(docs)
n_terms = len(vocab)
print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

dt_mat = np.zeros([n_terms, n_terms])
for itm in docs:
    for kk in itm:
        for jj in itm:
            if kk != jj:
                dt_mat[int(kk), int(jj)] += 1.0
print('co-occur done')

tmp_folder = \
    f'{str(args.results_folder)}_{args.n_topics}_{args.alpha}_{args.beta}'
par_file = tmp_folder + '/W.txt'
W = np.loadtxt(par_file, dtype=float)
n_topic = W.shape[1]
print('n_topic={}'.format(n_topic))

PMI_arr = []
n_topKeyword = args.n_top
for k in range(n_topic):
    topKeywordsIndex = W[:, k].argsort()[::-1][:n_topKeyword]
    PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
print('Average PMI={}'.format(np.average(np.array(PMI_arr))))

index = np.argsort(PMI_arr)

keyword_list, topic_list, PMI_list = [], [], []

for k in index:
    topic = 'Topic ' + str(k+1)
    topic_list.append(topic)
    print(topic + ': ', end=' ')
    PMI_list.append(PMI_arr[k])
    print(PMI_arr[k], end=' ')
    keyword = ''
    for w in np.argsort(W[:, k])[::-1][:n_topKeyword]:
        keyword += vocab[w] + ' '
    keyword_list.append(keyword)
    print(keyword, end=' ')
    print()

factor_df = {}
factor_df['topic'] = topic_list
factor_df['PMI'] = PMI_list
factor_df['keyowrd'] = keyword_list

factor = {
    14: '경험',
    4: '입장/대기시간',
    12: '박물관/전시',
    3: '도시경관',
    0: '쇼핑/먹거리',
    13: '전통건축',
    8: '쇼핑/먹거리',
    5: '교통/이동수단',
    9: '쇼핑/먹거리',
    2: '교통/이동수단',
    6: '안내서비스',
    1: '자연경관/산책로',
    10: '박물관/전시',
    11: '쇼핑',
    7: '먹거리'}
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

factor_df = pd.DataFrame(factor_df)

par_file = tmp_folder + '/H.txt'
H = np.loadtxt(par_file, dtype=float)
