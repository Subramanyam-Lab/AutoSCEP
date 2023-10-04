import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

# Load data
data = pd.read_csv('optimization_results.csv')
data['first stage decision'] = data['first stage decision'].apply(eval)
data['scenario'] = data['scenario'].apply(eval)

# # Separate lists into individual columns
# fsd_expanded = pd.DataFrame(data['first stage decision'].tolist(), columns=[f'fsd_{i}' for i in range(3)])
# scenario_expanded = pd.DataFrame(data['scenario'].tolist(), columns=[f'scenario_{i}' for i in range(3)])

# # Concatenate expanded columns and drop original columns
# data = pd.concat([data, fsd_expanded, scenario_expanded], axis=1).drop(['first stage decision', 'scenario'], axis=1)

# # Separate input and output
# X = data.drop('expected second stage value', axis=1)
# y = data['expected second stage value']

# Prepare data
X = pd.concat([data['first stage decision'].apply(pd.Series), 
               data['scenario'].apply(pd.Series)], axis=1)
y = data['expected second stage value']

# Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

# Define model
class Neural2SP(nn.Module):
    def __init__(self, input_size):
        super(Neural2SP, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.layer4(x)
        return x

# Initialize model, criterion and optimizer
model = Neural2SP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Adjust the learning rate


# Train model
EPOCHS = 100000
losses = []
min_val_loss = float('inf')
patience = 3000  # for early stopping
# stagnant_epochs = 0

for epoch in range(EPOCHS):
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Record loss for visualization
    losses.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    
    # Check for early stopping
    with torch.no_grad():
        val_loss = criterion(model(X_test_tensor), y_test_tensor)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1}, validation loss did not improve from {min_val_loss:.4f}')
                break
    
    # Step the learning rate scheduler
    # scheduler.step()


# Plotting loss after training
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('training_loss_plot.png')
plt.show()

# Evaluate model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    loss = criterion(predictions, y_test_tensor)
print(f'Test Loss: {loss.item():.4f}')
