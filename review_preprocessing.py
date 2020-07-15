import pandas as pd
import os
import spacy
from spacy_langdetect import LanguageDetector
from spacy.lang.en.stop_words import STOP_WORDS

datapath = r'../reviews'
resultpath = r'../results'

# %% extract english from each reviews

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

files = [x for x in os.listdir(os.path.join(datapath)) if '.csv' in x]

for f in files:
    print(f)
    review = pd.read_csv(os.path.join(datapath, f))
    data = review['title']+'. '+review['comment']
    # data=data.str.lower()

    doc = [nlp(text) for text in data.values]
    doc_sents = [x.sents for x in doc]

    review['sent'] = ''  # pd.DataFrame(columns=['text'])
    # en_data['rating'] = review['rating']

    for i, sents in enumerate(doc_sents):
        temp = []
        for sent in sents:
            if sent._.language['language'] == 'en':
                temp.append(sent.text)
        review.loc[i, 'sent'] = ' '.join(temp)
    review.to_csv(os.path.join(resultpath, 'Preprocess', f), index=False)


# %% remove stop words, number etc. and execute pos tagging

list_dir = os.listdir(os.path.join(resultpath, 'Preprocess'))
files = [x for x in list_dir if '.csv' in x]

model_list = ['en_core_web_md', 'en_core_web_sm']

for model in model_list:
    nlp = spacy.load(model)
    for f in files:
        print(f)
        review = pd.read_csv(os.path.join(resultpath, 'Preprocess', f))
        review = review.dropna()
        word_list = pd.Series(index=review.index, dtype=str)
        corp = [nlp(text) for text in review['sent'].values]
        for i, doc in enumerate(corp):
            temp = []
            for token in doc:
                if (token.lemma_ in nlp.vocab and
                    token.pos_ in ['ADJ', 'ADV', 'VERB', 'NOUN'] and
                    token.is_stop is False
                    and token.is_alpha
                        and token not in STOP_WORDS):
                    if token.lemma_ in [
                            'disappointed',
                            'disappointment',
                            'disappointing']:
                        temp.append('disappoint')
                    elif token.lemma_ in ['poorly', 'poor']:
                        temp.append('poor')
                    else:
                        temp.append(token.lemma_)
            word_list[i] = ' '.join(temp).lower()
        review[model] = word_list
        if len(review):
            review.to_csv(
                os.path.join(resultpath, 'Preprocess', f), index=False)
