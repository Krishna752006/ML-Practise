import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim

data = {
    "review": [
        # Positive reviews
        "The movie was absolutely amazing! A masterpiece.",
        "Fantastic performances by the entire cast.",
        "Loved the story and the cinematography. A must-watch!",
        "The action scenes were thrilling and well-executed.",
        "An emotional and heartwarming experience.",
        "Brilliant direction and an outstanding soundtrack.",
        "A gripping storyline that kept me on the edge of my seat.",
        "One of the best movies I've ever seen!",
        "The characters were relatable and well-developed.",
        "A perfect blend of humor and drama.",
        
        # Negative reviews
        "The movie was a complete waste of time.",
        "Terrible acting and a poorly written script.",
        "The story was boring and predictable.",
        "I couldn’t relate to any of the characters.",
        "The pacing was slow and the plot lacked depth.",
        "Way too many plot holes, very disappointing.",
        "The humor felt forced and awkward.",
        "It was overhyped and did not live up to expectations.",
        "The ending was rushed and unsatisfying.",
        "Poor direction and lackluster performances."
    ],
    "sentiment": [
        # Sentiments for positive reviews
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Sentiments for negative reviews
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

df['tokens'] = df['review'].apply(tokenize)

word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=50, window=3, min_count=1)

def sentence_to_vectors(tokens, model, max_len=5):
    vectors = []
    for word in tokens:
        if word in model.wv:  # Check if the word is in the model
            vectors.append(model.wv[word])  # Append the word's vector
    # Pad or truncate to max_len
    vectors = vectors[:max_len] + [[0] * model.vector_size] * (max_len - len(vectors))
    return np.array(vectors)

# Convert reviews to vectors
max_len = 5
X = np.array([sentence_to_vectors(tokens, word2vec_model, max_len) for tokens in df['tokens']])
y = np.array(df['sentiment'])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x) 
        out = self.fc(hidden[-1])
        return out

input_size = 50  # Size of the word vector
hidden_size = 32
output_size = 2  # Binary classification (positive/negative)

model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)  
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def predict_sentiment(review, model, word2vec_model, max_len=5):
    tokens = tokenize(review)
    vectors = sentence_to_vectors(tokens, word2vec_model, max_len)

    input_tensor = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1)

    return predicted.item()

sample_reviews = [
    "This movie was great and very entertaining!",
    "waste",
    "it was boring.",
    "one of the best movies."
]

for review in sample_reviews:
    sentiment = predict_sentiment(review, model, word2vec_model)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: \"{review}\" => Sentiment: {sentiment_label}")