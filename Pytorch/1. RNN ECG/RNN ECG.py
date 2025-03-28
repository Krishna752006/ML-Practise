import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

time_steps = np.linspace(0, 100, 1000)
ecg_signal = np.sin(time_steps) + np.random.normal(scale = 0.1,size = len(time_steps))

df = pd.DataFrame({'ECG':ecg_signal})

sequence_lengh = 20

def create_sequence(data, sl):
    sequences = []
    targets = []

    for i in range(len(data)-sl):
        seq = data[i:i+sl]
        target = data[i+sl]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences),np.array(targets)

sequences,targets = create_sequence(df['ECG'].values,sequence_lengh)

X = torch.tensor(sequences,dtype = torch.float32).unsqueeze(-1)
y = torch.tensor(targets,dtype = torch.float32).unsqueeze(-1)

class RNNModel(nn.Module):
    def __init__(self, input_size = 1,hidden_size = 64,num_layers = 2):
        super(RNNModel,self).__init__()
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        rnn_out,_ = self.rnn(x)
        output = self.fc(rnn_out[:,-1,:])
        return output

model = RNNModel() 

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs,y)
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')

model.eval()
with torch.no_grad():
    test_seq = X[-1].unsqueeze(0)
    predictions = []

    print("\nGenerated Predictions:")
    for i in range(50):
        pred = model(test_seq)
        predictions.append(pred.item()) 

        print(f"Step {i+1}, Predicted Value: {pred.item():.5f}")  

        pred = pred.unsqueeze(-1)
        test_seq = torch.cat((test_seq[:, 1:, :], pred.view(1, 1, 1)), dim=1)

plt.plot(df['ECG'].values,label = "Original ECG")
plt.plot(range(len(df)-50,len(df)),predictions,label = "Predicted ECG",linestyle = "dashed")
plt.legend()
plt.show()