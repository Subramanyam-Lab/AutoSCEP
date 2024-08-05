# nn_p.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NNP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

class TwoStageStochasticDataset(Dataset):
    def __init__(self, first_stage, scenarios, labels):
        self.first_stage = first_stage
        self.scenarios = scenarios
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.cat([self.first_stage[idx], self.scenarios[idx]], dim=0), self.labels[idx]

def train_nnp(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

    return model

# Usage example
if __name__ == "__main__":
    # Hyperparameters
    first_stage_dim = 10
    scenario_dim = 20
    hidden_dim = 64
    batch_size = 32
    epochs = 100
    lr = 0.001

    # Create dummy data
    num_samples = 1000
    first_stage = torch.randn(num_samples, first_stage_dim)
    scenarios = torch.randn(num_samples, scenario_dim)
    labels = torch.randn(num_samples)

    # Create dataset and dataloaders
    dataset = TwoStageStochasticDataset(first_stage, scenarios, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize and train the model
    model = NNP(first_stage_dim + scenario_dim, hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_nnp(model, train_loader, val_loader, epochs, lr, device)

    print("Training completed.")