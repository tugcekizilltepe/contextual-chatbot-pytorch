import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(sentence, vocabulary):
    tokenized_stemmed_sentence = [stem(w) for w in sentence]
    vectorized = np.zeros(len(vocabulary) + 1, dtype=np.float64)
    for word in tokenized_stemmed_sentence:
        if word in vocabulary:
            vectorized[vocabulary.index(word)] += 1
        else:
            vectorized[vectorized.shape[0] - 1] += 1

    return vectorized
