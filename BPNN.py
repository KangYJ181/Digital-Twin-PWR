import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# hyperparameters
lr = 0.0001
patience = 200
max_epochs = 20000
hidden_layer_counts = [2, 3]
l2_lambdas = [1e-3, 1e-4, 1e-5]
hidden_neurons = np.arange(20, 52, 2)


# pseudo random, guaranteed repeatability
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# structure of BPNN
class BPNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(BPNN, self).__init__()
        layers = []
        prev_dim = input_dim

        # construct each hidden layer
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # construct the output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# training function
def train_model(model, X_train, y_train, X_val, y_val, l2_lambda, return_history=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # save the training loss
        train_loss_history.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            # save the validation loss
            val_loss_history.append(val_loss.item())

        # early stop
        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    # save the best model
    model.load_state_dict(best_model_state)

    if return_history:
        return train_loss_history, val_loss_history
    else:
        return best_val_loss

# leave-one-out cross-validation
def loocv(X, y, hidden_layers, l2_lambda):
    n_samples = X.shape[0]
    val_losses = []

    for i in range(n_samples):
        # reserve the i-th sample as the validation set.
        X_train = torch.cat([X[:i], X[i+1:]], dim=0)
        y_train = torch.cat([y[:i], y[i+1:]], dim=0)
        X_val = X[i].unsqueeze(0)
        y_val = y[i].unsqueeze(0)

        set_seed()
        model = BPNN(input_dim=1, output_dim=1, hidden_layers=hidden_layers)
        val_loss = train_model(model, X_train, y_train, X_val, y_val, l2_lambda)
        val_losses.append(val_loss)

    return np.mean(val_losses)

# grid search function
def grid_search(X, y):
    best_mse = float('inf')
    best_params = None

    for num_layers in hidden_layer_counts:
        for n_neurons in hidden_neurons:
            neurons = [n_neurons] * num_layers
            for l2 in l2_lambdas:
                print(f"layers: {num_layers}, neurons: {neurons}, L2: {l2}")
                mse = loocv(X, y, hidden_layers=neurons, l2_lambda=l2)
                print(f"Mean LOOCV MSE: {mse:.6f}")
                if mse < best_mse:
                    best_mse = mse
                    best_params = (tuple(neurons), l2)

    return best_params, best_mse

# plot function
def plot_predictions(model, X, y_true, train_loss, val_loss, X_scaler, y_scaler):
    model.eval()
    with torch.no_grad():
        X_curve = np.linspace(15, 33, 200).reshape(-1, 1).astype(np.float32)
        X_curve_scaled = X_scaler.transform(X_curve)
        X_curve_tensor = torch.tensor(X_curve_scaled, dtype=torch.float32)
        y_curve_scaled = model(X_curve_tensor).numpy()
        y_curve = y_scaler.inverse_transform(y_curve_scaled)

        y_pred_scaled = model(X).numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

    X_denorm = X_scaler.inverse_transform(X.numpy())
    y_true_denorm = y_scaler.inverse_transform(y_true.numpy())

    plt.figure(figsize=(12, 7))
    plt.plot(X_curve, y_curve, label='Predicted Curve', linewidth=1.5, color='orange')
    plt.scatter(X_denorm, y_true_denorm, label='True Values',
                marker='o', s=60, edgecolor='black', zorder=10)
    plt.scatter(X_denorm, y_pred, label='Predicted Points',
                marker='x', s=80, color='red', linewidth=2, zorder=9)
    plt.xlabel('ΔT', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.title('High-Resolution Prediction Results (0.01 interval)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Process Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# POD coef. data
X_raw = np.linspace(15, 33, 19).reshape(-1, 1).astype(np.float32)
y_raw = np.array([-653232.3004,
-653268.4491,
-653322.2126,
-653361.2528,
-653399.9444,
-653426.6864,
-653468.7705,
-653529.3569,
-653563.4319,
-653564.2548,
-653607.6149,
-653678.7427,
-653704.9272,
-653725.3747,
-653783.1491,
-653857.9658,
-653878.5449,
-653906.1636,
-653918.0219
]).reshape(-1, 1).astype(np.float32)

# create scalers
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# grid search
best_params, best_mse = grid_search(X_tensor, y_tensor)
best_hidden_layers, best_l2 = best_params
print(f"Best structure: {best_hidden_layers} Best L2: {best_l2}")

# training
set_seed()
final_model = BPNN(input_dim=1, output_dim=1, hidden_layers=list(best_hidden_layers))

train_losses, val_losses = train_model(final_model, X_tensor, y_tensor, X_tensor, y_tensor, l2_lambda=best_l2, return_history=True)

# plots
plot_predictions(final_model, X_tensor, y_tensor, train_losses, val_losses, X_scaler, y_scaler)



# prediction and save
base_points = np.arange(15, 34, 1).astype(np.float32)  # 15-33 inclusive, step 1
extra_points = np.array([15.5, 20.5, 24.5, 28.5, 32.5], dtype=np.float32)
all_points = np.concatenate([base_points.reshape(-1, 1),
                            extra_points.reshape(-1, 1)])
X_pred_scaled = X_scaler.transform(all_points)
with torch.no_grad():
    X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32)
    y_pred_scaled = final_model(X_pred_tensor).numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled)
results = pd.DataFrame({
    'X (Original Scale)': all_points.squeeze(),
    'Predicted Value': y_pred.squeeze()
})
results.to_csv('predictions_MEAN_1.csv', index=False, header=True)