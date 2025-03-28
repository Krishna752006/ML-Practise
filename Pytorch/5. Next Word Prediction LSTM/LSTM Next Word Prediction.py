import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

data = {
    "text": [
        "The movie was fantastic and very engaging",
        "I hated the acting and the storyline",
        "It was boring and lacked depth",
        "Amazing performance by the actors and great direction",
        "Not worth watching at all",
        "One of the best movies I have ever seen"
    ]
}

df = pd.DataFrame(data)

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

df['tokens'] = df['text'].apply(tokenize)

word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=50, window=3, min_count=1)

def prepare_sequences(tokens, model, context_size=3):
    X, y = [], []
    for i in range(len(tokens) - context_size):
        context = tokens[i:i + context_size]
        target = tokens[i + context_size]
        X.append([model.wv[word] for word in context])
        y.append(model.wv.key_to_index[target])
    return np.array(X), np.array(y)

context_size = 3
X, y = [], []

for tokens in df['tokens']:
    X_seq, y_seq = prepare_sequences(tokens, word2vec_model, context_size)
    X.extend(X_seq)
    y.extend(y_seq)

X, y = np.array(X), np.array(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

class WordPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(WordPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

input_size = 50  # Size of the word vector
hidden_size = 64  # Number of hidden units in LSTM
vocab_size = len(word2vec_model.wv)  # Vocabulary size

model = WordPredictionLSTM(input_size, hidden_size, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def predict_next_word(context, model, word2vec_model, context_size=3):
    tokens = tokenize(context)
    if len(tokens) < context_size:
        raise ValueError(f"Context must have at least {context_size} words")
    tokens = tokens[-context_size:]

    # Convert to NumPy array first to avoid tensor warning
    vectors = np.array([word2vec_model.wv[word] for word in tokens])
    input_tensor = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    predicted_word = word2vec_model.wv.index_to_key[predicted_index]
    return predicted_word

def interactive_predict(model, word2vec_model, context_size=3):
    print("\nInteractive Word Prediction")
    print("Enter a context sentence to predict the next word.")
    print("Type 'exit' to quit.\n")

    while True:
        context = input("Enter context: ")
        if context.lower() == 'exit':
            print("Exiting interactive testing. Goodbye!")
            break

        try:
            next_word = predict_next_word(context, model, word2vec_model, context_size)
            print(f"Predicted next word: \"{next_word}\"")
        except ValueError as e:
            print(f"Error: {e}. Ensure the context has at least {context_size} words.")

interactive_predict(model, word2vec_model)