import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Load the dataset
df = pd.read_csv('ett.csv')

# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Target variable 'OT' - Oil Temperature
ot_data = df[['OT']].values

# Normalize the data
scaler = StandardScaler()
ot_data_scaled = scaler.fit_transform(ot_data)

# Function to create sequences of data for time series prediction
def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + pred_length)])
    return np.array(X), np.array(y)

# Create sequences for input and output (prediction)
seq_length = 96  # 4 days of hourly data as input
pred_length = 24  # Predict next day (24 hours)
X, y = create_sequences(ot_data_scaled, seq_length, pred_length)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
batch_size = 32
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Transformer-based Informer model definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class InformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers):
        super(InformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class InformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_decoder_layers):
        super(InformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output

class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(Informer, self).__init__()
        self.encoder = InformerEncoder(input_dim, d_model, nhead, num_encoder_layers)
        self.decoder = InformerDecoder(output_dim, d_model, nhead, num_decoder_layers)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

# Initialize model
input_dim = 1
output_dim = 1
d_model = 64
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3

model = Informer(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers)

# Define training parameters
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, batch_y)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, batch_y)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Evaluation function
def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X, batch_y)
            predictions.extend(outputs.cpu().numpy().reshape(-1))  # Reshape and extend
            actuals.extend(batch_y.cpu().numpy().reshape(-1))  # Reshape and extend

    # Convert to numpy arrays
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared Score: {r2}')

    return predictions, actuals, len(predictions)

# Plot for Hypothesis 1: Actual vs Predicted (Short-term prediction)
def plot_hypothesis_1_results(test_dates, actuals, predictions, pred_len):
    # Adjust to visualize the first prediction window
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates[:pred_len], actuals[:pred_len], label='Actual', color='blue')
    plt.plot(test_dates[:pred_len], predictions[:pred_len], label='Predicted', color='red', alpha=0.7)
    plt.title('Hypothesis 1: Actual vs Predicted (Short-term)')
    plt.xlabel('Date')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot for Hypothesis 2: Long-term Prediction Improvements
def plot_hypothesis_2_results(test_dates, actuals, predictions, long_term_window):
    # Adjust to visualize a longer prediction window (long-term predictions)
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates[:long_term_window], actuals[:long_term_window], label='Actual', color='blue')
    plt.plot(test_dates[:long_term_window], predictions[:long_term_window], label='Predicted', color='red', alpha=0.7)
    plt.title('Hypothesis 2: Long-term Prediction Improvements')
    plt.xlabel('Date')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run Training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# Run Evaluation
predictions, actuals, n_predictions = evaluate_model(model, test_loader, scaler, device)

# Select test dates
test_dates = df.index[-n_predictions:]

# Plot Hypothesis 1 results (short-term)
plot_hypothesis_1_results(test_dates, actuals, predictions, pred_length)

# Plot Hypothesis 2 results (long-term)
plot_hypothesis_2_results(test_dates, actuals, predictions, 7 * 24)  # 7 days
