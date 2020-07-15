import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
import spacy
from collections import Counter
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

from factor import extraction
from preprocessing import preprocessing as pre
from SeaNMF.model import SeaNMFL1


nlp = spacy.load("en_core_web_md")


def get_sentence_df(df):

    sentence_df = {}
    sentence_df['review_id'] = []
    sentence_df['sentence_id'] = []
    sentence_df['sentence'] = []
    sentence_df['spot'] = []
    sentence_df['date'] = []
    sentence_df['rating'] = []

    corp = [nlp(text) for text in df['sent'].values]

    for review_id, doc in enumerate(corp):

        sent_list = [nlp(sent.string.strip()) for sent in doc.sents]

        for sentence_id, sent in enumerate(sent_list):

            sentence_df['review_id'].append(
                str(review_id).zfill(5))
            sentence_df['sentence_id'].append(
                f'{str(review_id).zfill(5)}_{str(sentence_id).zfill(3)}')
            sentence_df['spot'].append(
                df['spot'][review_id])
            sentence_df['date'].append(
                pd.to_datetime(df['date'][review_id]).strftime('%Y%m%d'))
            sentence_df['rating'].append(
                df['rating'][review_id])

            sentence = []

            for token in sent:

                if (token.lemma_ in nlp.vocab and
                    token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB'] and
                    token.is_stop is False and
                    token.is_alpha and
                        token not in STOP_WORDS):

                    sentence.append(token.lemma_.lower())

            sentence = ' '.join([word for word in sentence])
            sentence_df['sentence'].append(sentence)

    sentence_df = pd.DataFrame(sentence_df)

    return sentence_df


if __name__ == '__main__':

    df = pd.read_csv('../results/en_core_web_md.csv')
    sentiment_df = pd.read_csv(
        '../results/en_core_web_md_sentiment_dictionary.csv')
    sentiment_dictionary = sentiment_df['word'].values

    sentence_df = get_sentence_df(df)

    word_counter = Counter(
        [w for s in sentence_df['sentence'] for w in s.split()])
    word_dictionary = [
        w for w, c in word_counter.items() if c >= 10 and len(w) > 1]

    sentence_df['extracted'] = sentence_df['sentence'].map(
        lambda x: ' '.join([
            word for word in x.split() if word in word_dictionary]))

    stopwords = [
        'korea', 'myeongdong', 'tower', 'namsan', 'seoul', 'busan', 'insadong',
        'thing', 'great', 'worth', 'fun', 'nice', 'tour', 'love', 'enjoy',
        'like', 'good', 'interesting', 'lovely', 'sure', 'park', 'recommend',
        'day', 'go', 'come', 'get', 'want', 'take', 'need']

    with open('../results/sentence.txt', 'w', -1, 'utf-8') as f:
        for s in sentence_df['extracted']:
            s = ' '.join([w for w in s.split() if w not in stopwords])
            if len(s.split()) >= 3:
                f.write(f'{s}\n')

    sentence_list = []
    with open('../results/sentence.txt', 'r', -1, 'utf-8') as f:
        while True:
            line = f.readline()
            line = line.replace('\n', '').split()
            if not line:
                break
            sentence_list.append(line)
    sentence_df['list'] = sentence_df['sentence'].map(lambda x: x.split())

    df_idx_list = []
    sentence_idx_list = []
    idx = 0
    s, e = 0, 200
    for i, sentence in enumerate(sentence_list):
        sub_df = sentence_df['list'].loc[s:e]
        sub_df = sub_df.map(lambda x: np.in1d(sentence, x).sum())
        if sub_df.max() == len(sentence):
            if (sub_df == sub_df.max()).sum() == 1:
                idx = sub_df.idxmax()
                df_idx_list.append(idx)
            else:
                l_array = sentence_df['list'].loc[s:e][sub_df == sub_df.max()]
                l_array = l_array.map(lambda x: len(x))
                idx = len_array.idxmin()
                df_idx_list.append(idx)
            sentence_idx_list.append(i)
            if e-idx <= 50:
                e += 100
                s += 100

    with open('../results/sentence_idx.txt', 'w', -1, 'utf-8') as f:
        for idx in sentence_idx_list:
            f.write(f'{idx}\n')

    with open('../results/sentence_df_idx.txt', 'w', -1, 'utf-8') as f:
        for idx in df_idx_list:
            f.write(f'{idx}\n')

    H = np.loadtxt('../results/')
