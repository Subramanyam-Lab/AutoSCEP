# nn_e.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ScenarioEmbedding(nn.Module):
    def __init__(self, scenario_dim, embed_dim1, embed_dim2):
        super().__init__()
        self.fc1 = nn.Linear(scenario_dim, embed_dim1)
        self.fc2 = nn.Linear(embed_dim1, embed_dim2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class NNE(nn.Module):
    def __init__(self, first_stage_dim, scenario_dim, embed_dim1, embed_dim2, embed_dim3, hidden_dim):
        super().__init__()
        self.scenario_embedding = ScenarioEmbedding(scenario_dim, embed_dim1, embed_dim2)
        self.fc1 = nn.Linear(first_stage_dim + embed_dim2, embed_dim3)
        self.fc2 = nn.Linear(embed_dim3, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, first_stage, scenarios):
        scenario_embed = self.scenario_embedding(scenarios)
        scenario_embed = torch.mean(scenario_embed, dim=1)
        x = torch.cat([first_stage, scenario_embed], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

class TwoStageStochasticDataset(Dataset):
    def __init__(self, first_stage, scenarios, labels):
        self.first_stage = first_stage
        self.scenarios = scenarios
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.first_stage[idx], self.scenarios[idx], self.labels[idx]

def train_nne(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for first_stage, scenarios, labels in train_loader:
            first_stage, scenarios, labels = first_stage.to(device), scenarios.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(first_stage, scenarios)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for first_stage, scenarios, labels in val_loader:
                first_stage, scenarios, labels = first_stage.to(device), scenarios.to(device), labels.to(device)
                outputs = model(first_stage, scenarios)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

    return model

# Usage example
if __name__ == "__main__":
    # Hyperparameters
    first_stage_dim = 10
    scenario_dim = 20
    embed_dim1 = 64
    embed_dim2 = 32
    embed_dim3 = 16
    hidden_dim = 128
    batch_size = 32
    epochs = 100
    lr = 0.001

    # Create dummy data
    num_samples = 1000
    first_stage = torch.randn(num_samples, first_stage_dim)
    scenarios = torch.randn(num_samples, 100, scenario_dim)  # Assuming 100 scenarios per sample
    labels = torch.randn(num_samples)

    # Create dataset and dataloaders
    dataset = TwoStageStochasticDataset(first_stage, scenarios, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize and train the model
    model = NNE(first_stage_dim, scenario_dim, embed_dim1, embed_dim2, embed_dim3, hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_nne(model, train_loader, val_loader, epochs, lr, device)

    print("Training completed.")