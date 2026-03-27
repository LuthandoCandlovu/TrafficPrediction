import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]   # take last timestep
        return self.fc(last_out)

def create_sequences(df, seq_len=6, target_col='speed'):
    """
    Create sequences for LSTM: X shape (samples, seq_len, n_features)
    y shape (samples,)
    """
    df = df.sort_values('timestamp').copy()
    features = ['hour', 'day_of_week', 'weather_temp', 'weather_precip',
                'volume', 'speed']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(df[target_col].iloc[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def main():
    df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'])
    seq_len = 6
    X, y, scaler = create_sequences(df, seq_len=seq_len)

    # Split time-based
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Model
    input_dim = X.shape[2]   # number of features
    model = TrafficLSTM(input_dim=input_dim, hidden_dim=64,
                        num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val_t).numpy().flatten()
    mae = mean_absolute_error(y_val, preds_val)
    print(f"LSTM - MAE: {mae:.2f}")

    # Save model and scaler
    torch.save(model.state_dict(), 'lstm_model.pth')
    joblib.dump(scaler, 'lstm_scaler.pkl')
    print("LSTM model saved to lstm_model.pth")

if __name__ == '__main__':
    main()
