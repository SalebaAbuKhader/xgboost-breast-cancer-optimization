from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score 
import optuna

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train baseline XGBoost model 
print("\n=== Baseline XGBoost Model ===")
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
baseline_model.fit(X_train, y_train)

# Evaluate the model 
baseline_preds = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)

print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

# Define objective function for Optuna 
def objective(trial): 
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0)
    }
    # Train XGBoost model
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    model.fit(X_train, y_train)
    # Evaluate model
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# Create study and optimize
print("\n=== Optuna Optimization ===")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")

# Define parameter grid for grid search 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
}

# Train XGBoost model using grid search
print("\n=== Grid Search Optimization ===")
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_:.4f}")

# Define parameter distribution for randomized search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Train XGBoost model using randomized search
print("\n=== Random Search Optimization ===")
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    verbose=1
)
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best accuracy: {random_search.best_score_:.4f}")

print("\n=== Summary ===")
print(f"Baseline XGBoost Accuracy: {baseline_accuracy:.4f}")
print(f"Optuna Best Accuracy: {study.best_value:.4f}")
print(f"Grid Search Best Accuracy: {grid_search.best_score_:.4f}")
print(f"Random Search Best Accuracy: {random_search.best_score_:.4f}")