import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import copy
import random
import time

# hyperparameters
lr = 0.001
patience = 200
max_epochs = 20000
l2_lambdas = [1e-3, 1e-4, 1e-5]
num_centers_list = np.arange(2, 20, 1)  # number of RBFs

start_time = time.time()


# device selection function
def select_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# pseudo random, guaranteed repeatability
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# structure of RBFNN
class RBFNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers):
        super(RBFNN, self).__init__()

        # initialize the center position of the RBFs
        initial_centers = torch.linspace(0, 1, num_centers).unsqueeze(1)
        self.centers = nn.Parameter(initial_centers, requires_grad=True)

        # initialize sigma of the RBFs
        sigma_init = 0.1
        sigma_tensor = torch.full((num_centers,), sigma_init, dtype=torch.float32)
        self.log_sigma = nn.Parameter(torch.log(sigma_tensor), requires_grad=True)

        self.linear = nn.Linear(num_centers, output_dim)

    def kernel_function(self, x):
        size = (x.size(0), self.centers.size(0), x.size(1))
        x_expanded = x.unsqueeze(1).expand(size)
        c_expanded = self.centers.unsqueeze(0).expand(size)

        sigma = torch.exp(self.log_sigma)
        beta = 1.0 / (2 * sigma ** 2)
        beta_expanded = beta.view(1, -1).expand(x.size(0), -1)

        dist = torch.sum((x_expanded - c_expanded) ** 2, dim=2)
        return torch.exp(- beta_expanded * dist)

    def forward(self, x):
        phi = self.kernel_function(x)
        return self.linear(phi)


# training function
def train_model(model, X_train, y_train, X_val, y_val, l2_lambda, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_history.append(val_loss.item())

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return best_val_loss, train_loss_history, val_loss_history


# leave-one-out cross-validation
def loocv(X, y, num_centers, l2_lambda, device):
    n_samples = X.shape[0]
    val_losses = []

    for i in range(n_samples):
        X_train = torch.cat([X[:i], X[i + 1:]], dim=0)
        y_train = torch.cat([y[:i], y[i + 1:]], dim=0)
        X_val = X[i].unsqueeze(0)
        y_val = y[i].unsqueeze(0)

        set_seed()
        model = RBFNN(input_dim=1, output_dim=1, num_centers=num_centers)
        val_loss, _, _ = train_model(model, X_train, y_train, X_val, y_val, l2_lambda, device)
        val_losses.append(val_loss)

    return np.mean(val_losses)


# grid search function
def grid_search(X, y, device):
    best_mse = float('inf')
    best_params = None

    for num_centers in num_centers_list:
        for wd in l2_lambdas:
            print(f"Num of centers: {num_centers}, L2: {wd}")
            mse = loocv(X, y, num_centers, wd, device)
            print(f"Mean LOOCV MSE: {mse:.6f}")
            if mse < best_mse:
                best_mse = mse
                best_params = (num_centers, wd)

    return best_params, best_mse


# plot function
def plot_predictions(model, X, y_true, train_loss, X_scaler, y_scaler, device):
    model.eval()
    with torch.no_grad():
        # prediction results
        X_curve = np.linspace(15, 33, 200).reshape(-1, 1).astype(np.float32)
        X_curve_scaled = X_scaler.transform(X_curve)
        X_curve_tensor = torch.tensor(X_curve_scaled, dtype=torch.float32).to(device)
        y_curve_scaled = model(X_curve_tensor).cpu().numpy()
        y_curve = y_scaler.inverse_transform(y_curve_scaled)

        # real data point
        X = X.to(device)
        y_pred_scaled = model(X).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        # inverse normalization
        X_denorm = X_scaler.inverse_transform(X.cpu().numpy())
        y_true_denorm = y_scaler.inverse_transform(y_true.cpu().numpy())

        centers_scaled = model.centers.detach().cpu().numpy()
        centers_original = X_scaler.inverse_transform(centers_scaled)

    # prediction curve, point and real data point
    plt.figure(figsize=(12, 7))
    plt.plot(X_curve, y_curve, label='Predicted Curve', linewidth=1.5, color='green')
    plt.scatter(X_denorm, y_true_denorm, label='True Values',
                marker='o', s=60, edgecolor='black', zorder=10)
    plt.scatter(X_denorm, y_pred, label='Predicted Points',
                marker='x', s=80, color='red', linewidth=2, zorder=9)

    # centers of RBFs
    plt.scatter(centers_original,
                np.full_like(centers_original, y_true_denorm.mean()),  # Y值仅用于可视化
                label='RBF Centers',
                marker='^', color='blue', s=100, edgecolor='black', zorder=8)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.title('High-Resolution Prediction Results with RBF Centers', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# POD coef. data
x_raw = np.arange(15, 34, 1).reshape(-1, 1).astype(np.float32)
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
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(x_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# convert to tensors
X_tensor = torch.tensor(x_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# select device
use_gpu = False
device = select_device(use_gpu)

# grid search
best_params, best_mse = grid_search(X_tensor, y_tensor, device)
best_centers, best_l2 = best_params
print(f"Best num of centers: {best_centers}, Best L2: {best_l2}")

set_seed()
final_model = RBFNN(input_dim=1, output_dim=1, num_centers=best_centers)
_, train_losses, _ = train_model(final_model, X_tensor, y_tensor, X_tensor, y_tensor, best_l2, device)

end_time = time.time()
print("Time consuming: {:.2f} seconds".format(end_time - start_time))

# get centers and sigmas
centers_scaled = final_model.centers.detach().cpu().numpy()
centers_original = x_scaler.inverse_transform(centers_scaled)
learned_sigma = torch.exp(final_model.log_sigma).detach().cpu().numpy()

# create DataFrame for centers and sigmas
data = {
    'Best_L2': [best_l2] * best_centers,
    'Best_Num_Centers': [best_centers] * best_centers,
    'Center': centers_original.flatten(),
    'Sigma': learned_sigma,
}
df = pd.DataFrame(data)
df.to_csv('Params_MEAN_1.csv', index=False)

num_epochs = len(train_losses)
df_loss = pd.DataFrame({
    'Epoch': list(range(1, len(train_losses) + 1)),
    'Training_Loss': train_losses
})
df_loss.to_csv('Loss_MEAN_1.csv', index=False)

# plots
plot_predictions(final_model, X_tensor, y_tensor, train_losses, x_scaler, y_scaler, device)

# prediction and save
prediction_points = np.arange(15, 34, 1).astype(np.float32).reshape(-1, 1)
prediction_points_extra = np.array([15.5, 20.5, 24.5, 28.5, 32.5], dtype=np.float32).reshape(-1, 1)
all_points = np.concatenate([prediction_points, prediction_points_extra], axis=0)
X_pred_scaled = x_scaler.transform(all_points)
X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_scaled = final_model(X_pred_tensor).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

results = pd.DataFrame({
    'X (Original Scale)': all_points.squeeze(),
    'Predicted Value': y_pred.squeeze()
})
results.to_csv('Pred_MEAN_1.csv', index=False, header=True)