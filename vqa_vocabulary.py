## converts words to indexes for questions and answer indexes to words
## tokenizer model is also present here.
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json

NEG_CONTRACTIONS = [
    (r'aren\'t', 'are not'),
    (r'can\'t', 'can not'),
    (r'couldn\'t', 'could not'),
    (r'daren\'t', 'dare not'),
    (r'didn\'t', 'did not'),
    (r'doesn\'t', 'does not'),
    (r'don\'t', 'do not'),
    (r'isn\'t', 'is not'),
    (r'hasn\'t', 'has not'),
    (r'haven\'t', 'have not'),
    (r'hadn\'t', 'had not'),
    (r'mayn\'t', 'may not'),
    (r'mightn\'t', 'might not'),
    (r'mustn\'t', 'must not'),
    (r'needn\'t', 'need not'),
    (r'oughtn\'t', 'ought not'),
    (r'shan\'t', 'shall not'),
    (r'shouldn\'t', 'should not'),
    (r'wasn\'t', 'was not'),
    (r'weren\'t', 'were not'),
    (r'won\'t', 'will not'),
    (r'wouldn\'t', 'would not'),
    (r'ain\'t', 'am not') # not only but stopword anyway
]
BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
OTHER_CONTRACTIONS = {
    "'m": 'am',
    "'ll": 'will',
    "'s": 'has', # or 'is' but both are stopwords
    "'d": 'had'  # or 'would' but both are stopwords
}
# The input statement is expected a string.
def tokenizing_sentence(line):
    ## Transform negative contractions
    for neg in NEG_CONTRACTIONS:
        line = re.sub(neg[0], neg[1], line)
    ## Tokenising the words
    tokens = word_tokenize(line)

    # transform other contractions (e.g 'll --> will)
    tokens = [OTHER_CONTRACTIONS[token] if OTHER_CONTRACTIONS.get(token)
              else token for token in tokens]

    return tokens


class Vocabulary(object):
    def __init__(self):
        print("Created Vocabulary Object")
        self.missingWords = 0
        nltk.download('stopwords')
        nltk.download('punkt')

    def build(self,questions_json_file):
        print("Building the indexes")
        self.words = []
        self.word2idx = {}
        word_counts = {}
        train_ques = json.load(open(questions_json_file, 'r'))
        train_size = len(train_ques['questions'])

        for i in tqdm(list(range(train_size)), desc='sentences'):
            question = train_ques['questions'][i]['question']
            for w in tokenizing_sentence(question.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1.0

        word_counts = sorted(list(word_counts.items()),
                             key=lambda x: x[1],
                             reverse=True)

        self.num_words = len(word_counts)

        for idx in range(self.num_words):
            word, word_count = word_counts[idx]
            self.word2idx[word] = idx
            self.words.append(word)


        print("Total number of different words in question {}".format(len(word_counts)))


    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = tokenizing_sentence(sentence.lower())
        try:

            word_idxs = [int(self.word2idx[w]) for w in words]
        except:
            self.missingWords = self.missingWords + 1
            word_idxs = []
        return word_idxs

    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.words[i] for i in idxs]
        if words[-1] != '.':
            words.append('.')
        length = np.argmax(np.array(words)=='.') + 1
        words = words[:length]
        sentence = "".join([" "+w if not w.startswith("'") \
                            and w not in string.punctuation \
                            else w for w in words]).strip()
        return sentence


