import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import pandas as pd


# pseudo random, guaranteed repeatability
Seed = 42
np.random.seed(Seed)

# set the hyperparameter range
param_bounds = {
    'C': (1, 100),
    'epsilon': (0.001, 1),
    'gamma': (1, 1000)
}

# POD coef. data
x = np.arange(15, 34).reshape(-1, 1)
y = np.array([-589.9985988,
-592.5060567,
-600.4624246,
-628.1062431,
-576.9196995,
-597.8399283,
-598.7800246,
-596.0347764,
-567.8403039,
-571.04924,
-579.792372,
-601.1838636,
-579.821415,
-577.7517749,
-618.5478679,
-575.4082145,
-572.1465832,
-612.6711194,
-567.60076
]).reshape(-1, 1)
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

# define the SVM
def svr_cv(C, epsilon, gamma):
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    mse = []
    for train_idx, val_idx in LeaveOneOut().split(x_scaled):
        model.fit(x_scaled[train_idx], y_scaled[train_idx])
        pred = model.predict(x_scaled[val_idx])
        mse.append((pred[0] - y_scaled[val_idx][0])**2)
    return -np.mean(mse)

# perform Bayesian optimization
optimizer = BayesianOptimization(
    f=svr_cv,
    pbounds=param_bounds,
    random_state=Seed,
    verbose=2
)
optimizer.maximize(init_points=50, n_iter=150)

# save the optimization results
history = optimizer.res
iterations = range(1, len(history)+1)
current_mse = [-h['target'] for h in history]
best_mse = [min(current_mse[:i]) for i in range(1, len(current_mse)+1)]
best_so_far = []
best_target = float('inf')
best_params = None

for i, res in enumerate(history):
    if -res['target'] < best_target:
        best_target = -res['target']
        best_params = res['params']
    best_so_far.append({
        'iteration': i + 1,
        'C': best_params['C'],
        'epsilon': best_params['epsilon'],
        'gamma': best_params['gamma'],
        'MSE': best_target
    })

pd.DataFrame(best_so_far).to_csv('progress_MEAN_1.csv', index=False)

# plot the optimization iteration curve
plt.figure(figsize=(10, 6))
plt.plot(iterations, current_mse, 'bo-', label='Current MSE')
plt.plot(iterations, best_mse, 'r--', label='Best MSE')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('MSE (log scale)', fontsize=12)
plt.yscale('log')
plt.title(f'optimization progress\nFinal Best MSE: {best_mse[-1]:.4f}', fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# train the final model
best_params = optimizer.max['params']
final_model = SVR(
    C=best_params['C'],
    epsilon=best_params['epsilon'],
    gamma=best_params['gamma']
)
final_model.fit(x_scaled, y_scaled)

# calculate the prediction results
x_main = np.arange(15, 34).reshape(-1, 1)
x_extra = np.array([15.5, 20.5, 24.5, 28.5, 32.5]).reshape(-1, 1)
x_all = np.vstack([x_main, x_extra])
x_all_scaled = x_scaler.transform(x_all)
y_pred = y_scaler.inverse_transform(
    final_model.predict(x_all_scaled).reshape(-1, 1)
).ravel()

# save the prediction results
pd.DataFrame({
    'x': x_all.flatten(),
    'prediction': y_pred
}).to_csv('predictions_MEAN_1.csv', index=False)

# plot the prediction curve
x_plot = np.linspace(15, 33, 300).reshape(-1, 1)
x_plot_scaled = x_scaler.transform(x_plot)
y_plot = y_scaler.inverse_transform(final_model.predict(x_plot_scaled).reshape(-1, 1)).ravel()

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', lw=2, label='Prediction Curve')
plt.scatter(x, y, c='r', s=50, ec='k', label='True Values')
plt.scatter(x, y_scaler.inverse_transform(final_model.predict(x_scaled).reshape(-1, 1)),facecolors='none', edgecolors='g', s=80, label='Predictions')
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output (y)', fontsize=12)
plt.title(f'Final SVR Model (C={best_params["C"]:.2f}, ε={best_params["epsilon"]:.3f}, γ={best_params["gamma"]:.1f})',fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()