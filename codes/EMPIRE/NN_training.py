import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import onnx
from onnx_tf.backend import prepare
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data['X'] = data['X'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
    X = data['X'].tolist()
    y = data['ExpectedSecondStageValue'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y, X_train_tensor.shape[1]

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, criterion, scaler_y):
    model.eval()
    total_loss = 0.0
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            outputs_list.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    outputs = scaler_y.inverse_transform(outputs_list)
    targets = scaler_y.inverse_transform(targets_list)
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
    return outputs, targets

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def save_scalers(scaler_X, scaler_y, scaler_dir):
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(scaler_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(scaler_dir, 'scaler_y.pkl'))

def convert_to_onnx(model, input_dim, onnx_model_path):
    dummy_input = torch.randn(1, input_dim).to(device)
    torch.onnx.export(model, dummy_input, onnx_model_path)

def convert_to_tf(onnx_model_path, tf_model_path):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

if __name__ == "__main__":
    csv_file_path = 'nn_training_data.csv'
    model_dir = 'models'
    scaler_dir = 'scalers'
    onnx_model_path = os.path.join(model_dir, 'fully_connected_nn.onnx')
    tf_model_path = os.path.join(model_dir, 'fully_connected_nn_tf')
    
    train_loader, test_loader, scaler_X, scaler_y, input_dim = load_and_preprocess_data(csv_file_path)
    
    model = FullyConnectedNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    outputs, targets = evaluate_model(model, test_loader, criterion, scaler_y)
    
    save_model(model, os.path.join(model_dir, 'fully_connected_nn.pth'))
    save_scalers(scaler_X, scaler_y, scaler_dir)
    convert_to_onnx(model, input_dim, onnx_model_path)
    convert_to_tf(onnx_model_path, tf_model_path)