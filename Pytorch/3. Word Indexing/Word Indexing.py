import re
import string
import torch
import torch.nn as nn
import torch.optim as optim

paragraph = '''
Machine learning is fascinating.
It allows computers to learn from data.
The more data, the better the learning.
Deep learning is a subset of machine learning.
Neural networks are at the core of deep learning.
Artificial intelligence is evolving rapidly.
Data science combines domain expertise with programming skills.
Big data plays a  CRUCIAL role in  MODERN analytics.
Natural language processing is a key part of AI.
Predictive modeling helps in forecasting future trends.'''

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

tokens = tokenize(paragraph)
lower_tokens = [word.lower() for word in tokens]
print("TOKENS: ", tokens)
print("\nLOWER TOKENS: ", lower_tokens)

def lemmatize(token):
    lemmas = {
        'learning': 'learn',
        'computers': 'computer',
        'data': 'datum',
        'networks': 'network'
    }

    return lemmas.get(token, token)

lemmatized_tokens = [lemmatize(token) for token in lower_tokens]
print("LEMMATIZED TOKENS: ", lemmatized_tokens)

def stem(token):
    suffixes = ['ing', 'ed', 's']

    for suffix in suffixes:
        if token.endswith(suffix):
            return token[:-len(suffix)]

    return token

stemmed_tokens = [stem(token) for token in lemmatized_tokens]
print("STEMMED TOKENS: ", stemmed_tokens)

stop_words = {'is', 'to', 'the', 'from', 'and', 'are', 'at', 'of', 'a'}
filtered_tokens = [token for token in stemmed_tokens if token not in stop_words]
print("FILTERED TOKENS: ", filtered_tokens)

def create_word_vectors(tokens, vector_size=50):
    vocab = set(tokens)
    word_vectors = {word: torch.rand(vector_size) for word in vocab}
    return word_vectors

word_vectors = create_word_vectors(filtered_tokens)
word_to_index = {word: idx for idx, word in enumerate(filtered_tokens)}
print("WORD TO INDEX: ", word_to_index)