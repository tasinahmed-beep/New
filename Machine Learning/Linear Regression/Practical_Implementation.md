# Linear Regression: Practical Implementation

This guide provides hands-on examples of implementing linear regression in Python, from scratch and using popular libraries.

## Prerequisites

- Python 3.6 or higher
- Basic understanding of Python programming
- Familiarity with NumPy arrays

## Setup

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## 1. Linear Regression from Scratch

### 1.1 Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

### 1.2 Implement the Linear Regression Class

```python
class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the linear regression model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            The step size for gradient descent updates
        n_iterations : int, default=1000
            Number of iterations for gradient descent
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Make predictions
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and store loss
            loss = np.mean((y_predicted - y)**2)
            self.loss_history.append(loss)
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self):
        """
        Get the learned parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias
        }
```

### 1.3 Example with Car Fuel Efficiency Data

```python
# Create sample data (similar to the examples in the theoretical documents)
# Car weight (in 1000s of pounds) vs. fuel efficiency (MPG)
car_data = {
    'weight': [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37],
    'mpg': [18, 15, 18, 16, 15, 14, 24]
}

df = pd.DataFrame(car_data)
X = df[['weight']].values
y = df['mpg'].values

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.xlabel('Weight (1000s of pounds)')
plt.ylabel('Fuel Efficiency (MPG)')
plt.title('Car Weight vs. Fuel Efficiency')
plt.grid(True)
plt.show()

# Split data (in this small example, we'll use all data for training)
# In practice, you should always split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_scratch.fit(X_train, y_train)

# Get the learned parameters
params = model_scratch.get_params()
print(f"Learned weight: {params['weights'][0]:.4f}")
print(f"Learned bias: {params['bias']:.4f}")
print(f"Model equation: MPG = {params['bias']:.2f} + {params['weights'][0]:.2f} * Weight")

# Make predictions
y_pred = model_scratch.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Data points')
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model_scratch.predict(x_line)
plt.plot(x_line, y_line, 'r-', label='Regression line')
plt.xlabel('Weight (1000s of pounds)')
plt.ylabel('Fuel Efficiency (MPG)')
plt.title('Linear Regression: Car Weight vs. Fuel Efficiency')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(model_scratch.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()
```

## 2. Using Scikit-Learn

### 2.1 Basic Implementation

```python
from sklearn.linear_model import LinearRegression

# Create and train the model
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

# Get parameters
print(f"Scikit-learn weight: {model_sklearn.coef_[0]:.4f}")
print(f"Scikit-learn bias: {model_sklearn.intercept_:.4f}")

# Make predictions
y_pred_sklearn = model_sklearn.predict(X_test)

# Evaluate
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
print(f"Scikit-learn MSE: {mse_sklearn:.4f}")
print(f"Scikit-learn R-squared: {r2_sklearn:.4f}")
```

### 2.2 Multiple Features Example

```python
# Create a more complex dataset with multiple features
np.random.seed(42)
n_samples = 100

# Generate synthetic data
X1 = np.random.normal(3.5, 0.5, n_samples)  # Weight
X2 = np.random.normal(150, 30, n_samples)   # Horsepower
X3 = np.random.normal(100, 20, n_samples)   # Engine displacement

# True relationship: MPG = 40 - 5*weight - 0.05*horsepower - 0.02*displacement + noise
y_multi = 40 - 5*X1 - 0.05*X2 - 0.02*X3 + np.random.normal(0, 2, n_samples)

# Create DataFrame
df_multi = pd.DataFrame({
    'weight': X1,
    'horsepower': X2,
    'displacement': X3,
    'mpg': y_multi
})

# Prepare data
X_multi = df_multi[['weight', 'horsepower', 'displacement']].values
y_multi = df_multi['mpg'].values

# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Print coefficients
print("Multiple Linear Regression Results:")
print(f"Intercept: {model_multi.intercept_:.4f}")
print("Coefficients:")
for i, feature in enumerate(['weight', 'horsepower', 'displacement']):
    print(f"  {feature}: {model_multi.coef_[i]:.4f}")

# Evaluate
y_pred_multi = model_multi.predict(X_test_multi)
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)
print(f"\nMultiple Regression MSE: {mse_multi:.4f}")
print(f"Multiple Regression R-squared: {r2_multi:.4f}")
```

## 3. Hyperparameter Tuning

### 3.1 Learning Rate Experiment

```python
def experiment_learning_rates(learning_rates, X, y):
    """
    Experiment with different learning rates to see their effect on convergence.
    """
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        model = LinearRegressionScratch(learning_rate=lr, n_iterations=500)
        model.fit(X, y)
        plt.plot(model.loss_history, label=f'LR = {lr}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
experiment_learning_rates(learning_rates, X_train, y_train)
```

### 3.2 Batch Size Experiment

```python
class MiniBatchGradientDescent:
    """
    Linear Regression with mini-batch gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for j in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]
                
                # Calculate gradients
                y_predicted = np.dot(X_batch, self.weights) + self.bias
                dw = (1/len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1/len(X_batch)) * np.sum(y_predicted - y_batch)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Calculate loss for this iteration
            y_full_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_full_pred - y)**2)
            self.loss_history.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

def experiment_batch_sizes(batch_sizes, X, y):
    """
    Experiment with different batch sizes.
    """
    plt.figure(figsize=(12, 8))
    
    for bs in batch_sizes:
        model = MiniBatchGradientDescent(learning_rate=0.01, n_iterations=500, batch_size=bs)
        model.fit(X, y)
        plt.plot(model.loss_history, label=f'Batch Size = {bs}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Effect of Batch Size on Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test different batch sizes
batch_sizes = [1, 4, 16, 32, len(X_train)]  # SGD, small batches, large batches, full batch
experiment_batch_sizes(batch_sizes, X_train, y_train)
```

## 4. Model Evaluation and Diagnostics

### 4.1 Comprehensive Evaluation

```python
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive model evaluation.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"{model_name} Evaluation:")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Check for overfitting
    if test_mse > train_mse * 1.5:
        print("⚠️  Potential overfitting detected!")
    
    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

# Evaluate both models
scratch_results = evaluate_model(model_scratch, X_train, X_test, y_train, y_test, "Scratch Model")
sklearn_results = evaluate_model(model_sklearn, X_train, X_test, y_train, y_test, "Scikit-learn Model")
```

### 4.2 Residual Analysis

```python
def plot_residuals(model, X, y, title="Residual Plot"):
    """
    Plot residuals to check model assumptions.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Residual vs Predicted plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)
    
    # Q-Q plot for normality
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot residuals for both models
plot_residuals(model_scratch, X_test, y_test, "Scratch Model Residuals")
plot_residuals(model_sklearn, X_test, y_test, "Scikit-learn Model Residuals")
```

## 5. Common Pitfalls and Solutions

### 5.1 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Create data with different scales
X_unscaled = np.column_stack([
    np.random.normal(3.5, 0.5, 100),  # Weight (small scale)
    np.random.normal(150, 30, 100)    # Horsepower (large scale)
])
y_unscaled = 40 - 5*X_unscaled[:, 0] - 0.05*X_unscaled[:, 1] + np.random.normal(0, 2, 100)

# Split data
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_unscaled, y_unscaled, test_size=0.2, random_state=42
)

# Model without scaling
model_unscaled = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_unscaled.fit(X_train_u, y_train_u)

# Model with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_u)
X_test_scaled = scaler.transform(X_test_u)

model_scaled = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_scaled.fit(X_train_scaled, y_train_u)

# Compare convergence
plt.figure(figsize=(12, 6))
plt.plot(model_unscaled.loss_history, label='Without Scaling')
plt.plot(model_scaled.loss_history, label='With Scaling')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Effect of Feature Scaling on Convergence')
plt.legend()
plt.grid(True)
plt.show()
```

### 5.2 Learning Rate Issues

```python
def demonstrate_learning_rate_issues(X, y):
    """
    Demonstrate problems with learning rate selection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Too small learning rate
    model1 = LinearRegressionScratch(learning_rate=0.001, n_iterations=1000)
    model1.fit(X, y)
    axes[0, 0].plot(model1.loss_history)
    axes[0, 0].set_title('Learning Rate Too Small (0.001)')
    axes[0, 0].set_ylabel('Loss')
    
    # Good learning rate
    model2 = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model2.fit(X, y)
    axes[0, 1].plot(model2.loss_history)
    axes[0, 1].set_title('Good Learning Rate (0.01)')
    
    # Too large learning rate
    model3 = LinearRegressionScratch(learning_rate=0.5, n_iterations=1000)
    model3.fit(X, y)
    axes[1, 0].plot(model3.loss_history)
    axes[1, 0].set_title('Learning Rate Too Large (0.5)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    
    # Very large learning rate (divergence)
    model4 = LinearRegressionScratch(learning_rate=1.0, n_iterations=100)
    try:
        model4.fit(X, y)
        axes[1, 1].plot(model4.loss_history)
    except:
        axes[1, 1].text(0.5, 0.5, 'Model Diverged!', 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        transform=axes[1, 1].transAxes,
                        fontsize=12)
    axes[1, 1].set_title('Learning Rate Very Large (1.0)')
    axes[1, 1].set_xlabel('Iteration')
    
    for ax in axes.flat:
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

demonstrate_learning_rate_issues(X_train, y_train)
```

## 6. Best Practices Summary

1. **Always split your data** into training and testing sets
2. **Scale features** when they have different ranges
3. **Experiment with learning rates** to find the optimal value
4. **Monitor loss curves** to detect convergence issues
5. **Evaluate using multiple metrics** (MSE, R², etc.)
6. **Check residuals** to validate model assumptions
7. **Start with simple models** before moving to complex ones
8. **Document your experiments** and results

## 7. Next Steps

After mastering these implementations, you can explore:

- Regularization techniques (Ridge, Lasso)
- Polynomial regression for non-linear relationships
- Cross-validation for robust model evaluation
- Feature engineering techniques
- More advanced optimization algorithms

---

**Related Documents**:
- [Linear_Regression.md](Linear_Regression.md) - Theoretical foundation
- [Gradient_Descent.md](Gradient_Descent.md) - Optimization theory
- [Hyperparameter.md](Hyperparameter.md) - Hyperparameter tuning concepts
- [Loss.md](Loss.md) - Loss function details

**Last Updated**: October 2025