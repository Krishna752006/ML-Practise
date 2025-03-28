import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('Google_Stock_Price.csv')
prices = data['Open'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(prices_scaled)):
    X.append(prices_scaled[i-sequence_length:i, 0])
    y.append(prices_scaled[i, 0])
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

X_train = X_train.view(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.view(X_test.shape[0], X_test.shape[1], 1)

class StockPriceRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockPriceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = StockPriceRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test).detach().numpy()

predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction")
plt.show()

# Predict on a new sample
sample_sequence = prices[-60:]  # Use the last 60 days as an example
sample_sequence_scaled = scaler.transform(sample_sequence)

# Convert to tensor and reshape for the model
sample_tensor = torch.tensor(sample_sequence_scaled, dtype=torch.float32).view(1, sequence_length, 1)

# Make prediction
model.eval()
with torch.no_grad():
    predicted_scaled = model(sample_tensor).item()

# Inverse transform to get the actual predicted price
predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

# The actual value (next day's price)
actual_price = prices[-1]  # Replace this with the actual value if available

print(f"Actual Stock Price: {actual_price[0]:.2f}")
print(f"Predicted Stock Price: {predicted_price:.2f}")