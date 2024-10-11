import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os

# Check for GPU availability and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_directory_path = "results_V2"
problem_sizes = [(25, 25)]
num_data_size = 30 

for size in problem_sizes:
    size1, size2 = size
    directory_path = os.path.join(base_directory_path, f"CPLP_{size1}_{size2}")
    file_pattern = f"results_{size1}_{size2}_{{}}.csv"

    for i in range(num_data_size):
        print(f"Processing file number {i} for problem size {size}")
        file_path = os.path.join(directory_path, file_pattern.format(i))
        model_filename = f"CPLP_{size1}_{size2}_{i}.onnx"
        plot_filename = f"CPLP_{size1}_{size2}_{i}_loss_plot.png"
        
        if base_directory_path == "results_V2": 
            model_directory = os.path.join("Models_V2", f"CPLP_{size1}_{size2}")
            data = pd.read_csv(file_path)
            data['first stage decision'] = data['first stage decision'].apply(lambda x: list(map(int, x.strip("[]").split())))
        else :
            model_directory = os.path.join("Models_V1", f"CPLP_{size1}_{size2}")
            data = pd.read_csv(file_path)
            data['first stage decision'] = data['first stage decision'].apply(lambda x: list(eval(x))) 
    
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Assuming the feasibility column exists in the data
        data['feasibility'] = data['feasibility'].astype(int)
        X = data['first stage decision'].apply(pd.Series)
        y = data['expected second stage value']
        feasibility = data['feasibility']

        # Data Splitting
        X_train, X_test, y_train, y_test, feasibility_train, feasibility_test = train_test_split(X, y, feasibility, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors and move to the device
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
        feasibility_train_tensor = torch.LongTensor(feasibility_train.values).view(-1).to(device)

        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)
        feasibility_test_tensor = torch.LongTensor(feasibility_test.values).view(-1).to(device)

        train_data = TensorDataset(X_train_tensor, feasibility_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)


        # Neural Network with Feasibility Output
        class Neural2SP(nn.Module):
            def __init__(self, input_size, dropout):
                super(Neural2SP, self).__init__()
                self.layer1 = nn.Linear(input_size, 5)
                self.layer2 = nn.Linear(5, 2)  # 2 outputs: one for value, one for feasibility
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.dropout(x)
                x = self.relu(x)
                x = self.layer2(x)
                return x[:, 0].view(-1, 1), self.softmax(x[:, 1].unsqueeze(1)).view(-1, 1)


        # Custom Loss Function
        def custom_loss(outputs, labels, feasibility, lambda_param):
            mse_loss = nn.MSELoss()(outputs[0], labels)
            bce_loss = nn.BCELoss()(outputs[1], feasibility.float().view(-1, 1))
            
            scale_factor = feasibility.float().mean()
            scaled_mse_loss = mse_loss * scale_factor

            total_loss = scaled_mse_loss + lambda_param * bce_loss
            return total_loss

        # Define the objective function for hyperopt
        def objective(params):
            lr = params['lr']
            weight_decay = params['weight_decay']
            dropout = params['dropout']
            lambda_param = params['lambda'] 

            model = Neural2SP(X_train.shape[1], dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Train model
            EPOCHS = 1000
            min_val_loss = float('inf')

            for epoch in range(EPOCHS):
                epoch_train_loss = 0
                for inputs, feasibility, labels in train_loader:
                    inputs, feasibility, labels = inputs.to(device), feasibility.to(device), labels.to(device)
                    value_output, feasibility_output = model(inputs)
                    loss = custom_loss((value_output, feasibility_output), labels, feasibility, lambda_param)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()

                # Calculate validation loss for the epoch
                with torch.no_grad():
                    model.eval()  # Set to evaluation mode
                    value_output, feasibility_output = model(X_test_tensor)
                    val_loss = custom_loss((value_output, feasibility_output), y_test_tensor, feasibility_test_tensor, lambda_param)
                    model.train()  # Reset to train mode for the next epoch
                    
                    # Check for early stopping
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1
                    if stagnant_epochs >= 30:
                        print(f'Early stopping at epoch {epoch+1}, validation loss did not improve from {min_val_loss:.4f}')
                        break
                    
            return {'loss': val_loss, 'status': STATUS_OK}
        
        space = {
            'lr': hp.loguniform('lr', -10, -2),
            'weight_decay': hp.loguniform('weight_decay', -10, -2),
            'dropout': hp.uniform('dropout', 0, 0.5),
            'lambda': hp.uniform('lambda', 1, 20)  # Range for lambda
        }

        # Run the optimizer
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
        
        
        print(f"Best parameters for file number {i}: ", best)

        # Now, use the best parameters to train the final model for this file
        best_dropout = best['dropout']
        best_lr = best['lr']
        best_weight_decay = best['weight_decay']

        model = Neural2SP(X_train.shape[1], best_dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)

        # Train model
        EPOCHS = 1000
        train_losses = []
        val_losses = []
        min_val_loss = float('inf')
        stagnant_epochs = 0
        lambda_param = 0.5

        for epoch in range(EPOCHS):
            epoch_train_loss = 0
            for inputs, feasibility, labels in train_loader:
                inputs, feasibility, labels = inputs.to(device), feasibility.to(device), labels.to(device)
                value_output, feasibility_output = model(inputs)
                loss = custom_loss((value_output, feasibility_output), labels, feasibility, lambda_param)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()

            # Average training loss for the epoch
            train_losses.append(epoch_train_loss/len(train_loader))
            
            # Calculate validation loss for the epoch
            with torch.no_grad():
                model.eval()  # Set to evaluation mode
                value_output, feasibility_output = model(X_test_tensor)
                val_loss = custom_loss((value_output, feasibility_output), y_test_tensor, feasibility_test_tensor, lambda_param)
                val_losses.append(val_loss.item())
                model.train()  # Reset to train mode for the next epoch
                
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
            
            # Check for early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= 50:
                    print(f'Early stopping at epoch {epoch+1}, validation loss did not improve from {min_val_loss:.4f}')
                    break
            
        
        # After the training loop
        # Make sure the model is in evaluation mode
        model.eval()

        # Generate predictions on the test dataset
        with torch.no_grad():
            predicted_tensor, _ = model(X_test_tensor)
            predicted = predicted_tensor.cpu().numpy()
            actual = y_test_tensor.cpu().numpy()

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5, label='Predicted vs Actual')

        # Add y=x line (perfect predictions line)
        max_val = max(actual.max(), predicted.max())
        min_val = min(actual.min(), predicted.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x Line')

        # Calculate and plot the trend line
        z = np.polyfit(actual.flatten(), predicted.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(actual, p(actual), "r--", label='Trend Line')

        # Label the plot
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        scatter_plot_save_path = os.path.join(model_directory, f"CPLP_{size1}_{size2}_{i}_accuracy.png")
        plt.savefig(scatter_plot_save_path)
        plt.close()
        # Save model to ONNX format
        dummy_input = torch.randn(1, X_train.shape[1]).to(device)
        
        # Plotting configuration
        tick_font_size = 25
        line_width = 4
        axis_line_width = 2
        dpi = 300

        # Create a new figure with configured DPI
        plt.figure(figsize=(10, 6), dpi=dpi)

        # Plot training and validation losses
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss', linewidth=line_width)
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss', linestyle='--', linewidth=line_width)

        # Set label names and title
        plt.xlabel('Epoch', fontsize=tick_font_size)
        plt.ylabel('Loss', fontsize=tick_font_size)
        plt.title('Loss over Epochs', fontsize=tick_font_size)

        # Set the tick parameters for both axes
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)

        # Set legend with larger font size
        plt.legend(fontsize=tick_font_size)

        # Set axis line width for the plot spines
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(axis_line_width)

        # Enable and configure grid
        plt.grid(True, linewidth=1)  # Set the grid line width to a value of your choice

        # Adjust layout to ensure everything fits without overlapping
        plt.tight_layout()
        
        plot_save_path = os.path.join(model_directory, plot_filename)
        plt.savefig(plot_save_path)
        plt.close()
        
        model_save_path = os.path.join(model_directory, model_filename)
        torch.onnx.export(model, dummy_input, model_save_path, verbose=False)


