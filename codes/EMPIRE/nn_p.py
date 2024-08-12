import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import ast
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

class NNP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze()

class NNPDataset(Dataset):
    def __init__(self, fsd_data, scenario_data, labels):
        self.fsd_data = fsd_data
        self.scenario_data = scenario_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fsd = torch.tensor(self.fsd_data[idx], dtype=torch.float32)
        scenario = torch.tensor(self.scenario_data[idx], dtype=torch.float32)
        return torch.cat([fsd, scenario]), self.labels[idx]

def train_nnp(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

    return model

def evaluate_model(model, test_loader, device, scaler_labels):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    predictions = scaler_labels.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_labels.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    return r2, mae


def load_your_data(filename):
    df = pd.read_csv(filename)
    df['fsd_data'] = df['fsd_data'].apply(ast.literal_eval)
    df['scenario_data'] = df['scenario_data'].apply(ast.literal_eval)
    
    fsd_data = df['fsd_data'].tolist()
    scenario_data = df['scenario_data'].tolist()
    labels = df['label'].tolist()

    # Normalize the data
    scaler_fsd = StandardScaler()
    scaler_scenario = StandardScaler()
    scaler_labels = StandardScaler()

    fsd_data = scaler_fsd.fit_transform(fsd_data)
    scenario_data = scaler_scenario.fit_transform(scenario_data)
    labels = scaler_labels.fit_transform(np.array(labels).reshape(-1, 1)).flatten()

    labels = torch.tensor(labels, dtype=torch.float32)
    
    return fsd_data, scenario_data, labels, scaler_labels

if __name__ == "__main__":
    filename = "nnp_dataset_20240807011101.csv"
    fsd_data, scenario_data, labels, scaler_labels = load_your_data(filename)
    
    dataset = NNPDataset(fsd_data, scenario_data, labels)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_dim = len(dataset[0][0])
    hidden_dim = 256
    epochs = 5000
    lr = 0.0001
    
    model = NNP(input_dim, hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_nnp(model, train_loader, val_loader, epochs, lr, device)
    
    print("Training completed.")
    
    torch.save(trained_model.state_dict(), 'trained_nnp_model.pth')
    
    r2, mae = evaluate_model(trained_model, test_loader, device, scaler_labels)
    print(f"Test R^2 Score: {r2:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    with torch.no_grad():
        prediction = trained_model(sample_input)
    prediction = scaler_labels.inverse_transform(prediction.cpu().numpy().reshape(-1, 1))
    print("Sample prediction:", prediction[0][0])