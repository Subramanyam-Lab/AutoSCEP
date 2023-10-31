import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
import torch.onnx
import os
import torch
import re

# Check for GPU availability and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory and file pattern
directory_path = "results2/CPLP_50_50"
file_pattern = "results_50_50_{}.csv"

for i in range(30):
    file_path = os.path.join(directory_path, file_pattern.format(i))
    data = pd.read_csv(file_path)

    pattern = re.compile(r"results_(\d+)_(\d+)_(\d+).csv")
    match = pattern.search(file_path)
    if not match:
        print(f"Unexpected file name pattern: {file_path}")
        continue

    size1, size2, file_num = match.groups()

    model_directory = os.path.join("Models", f"CPLP_{size1}_{size2}")
    model_filename = f"CPLP_{size1}_{size2}_{file_num}.onnx"
    plot_filename = f"CPLP_{size1}_{size2}_{file_num}_loss_plot.png"

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    data['first stage decision'] = data['first stage decision'].apply(lambda x: list(eval(x)))
    X = data['first stage decision'].apply(pd.Series)
    y = data['expected second stage value']

    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors and move to the device
    # X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    # X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)

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

    # Initialize model, criterion and optimizer, and move the model to the device
    model = Neural2SP(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # Train model
    EPOCHS = 10000
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    stagnant_epochs = 0

    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Average training loss for the epoch
        train_losses.append(epoch_train_loss/len(train_loader))
        
        # Calculate validation loss for the epoch
        with torch.no_grad():
            model.eval()  # Set to evaluation mode
            val_loss = criterion(model(X_test_tensor), y_test_tensor)
            val_losses.append(val_loss.item())
            model.train()  # Reset to train mode for the next epoch
            
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
        
        # Check for early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= 3000:
                print(f'Early stopping at epoch {epoch+1}, validation loss did not improve from {min_val_loss:.4f}')
                break
        
        
    # Save model to ONNX format
    dummy_input = torch.randn(1, X_train.shape[1]).to(device)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그림 저장
    plot_save_path = os.path.join(model_directory, plot_filename)
    plt.savefig(plot_save_path)
    plt.close()

    # 모델 저장
    model_save_path = os.path.join(model_directory, model_filename)
    torch.onnx.export(model, dummy_input, model_save_path, verbose=True)