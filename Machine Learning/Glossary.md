# Machine Learning Glossary

This glossary defines key terms used throughout the machine learning materials. Terms are organized alphabetically with references to where they are discussed in detail.

## A

**Absolute Error**
- The absolute difference between predicted and actual values
- Used in MAE (Mean Absolute Error) calculations
- See: [Loss.md](Linear%20Regression/Loss.md)

**Accuracy**
- The proportion of correct predictions out of total predictions
- Commonly used in classification problems
- Note: Not appropriate for regression problems

**Activation Function**
- A function that determines the output of a neural network node
- Not covered in current materials but relevant for neural networks

**Adam Optimizer**
- An adaptive learning rate optimization algorithm
- Not covered in current materials

## B

**Batch**
- A subset of the training dataset used in one iteration
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Batch Gradient Descent**
- Uses the entire dataset to compute gradients in each iteration
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Batch Size**
- The number of examples processed before updating model parameters
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Bias**
- The intercept term in a linear model (b in y = b + wx)
- Also called w₀ in some contexts
- A parameter learned during training
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

**Binary Classification**
- Classification with two possible outcomes
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

## C

**Classification**
- Predicting a category or class label
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

**Clustering**
- Grouping similar data points together
- An unsupervised learning technique
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

**Convex Function**
- A function where any line segment between two points lies above the function
- Linear regression loss functions are convex
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

**Convergence**
- When the model parameters stop changing significantly between iterations
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

**Cost Function**
- Synonym for loss function
- Measures how well the model is performing
- See: [Loss.md](Linear%20Regression/Loss.md)

**Cross-Validation**
- A technique for evaluating model performance
- Not covered in current materials

## D

**Dataset**
- A collection of examples used for training or evaluation
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Derivative**
- The rate of change of a function
- Used in gradient descent to find the direction of steepest ascent
- See: [Gradient Descent Calculation.md](Linear%20Regression/Gradient%20Descent%20Calculation.md)

**Diversity**
- The range of examples covered in a dataset
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

## E

**Epoch**
- One complete pass through the entire training dataset
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Evaluation**
- Assessing model performance on unseen data
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Example**
- A single instance in a dataset, containing features and possibly a label
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Feature**
- An input variable used to make predictions
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Feature Engineering**
- The process of creating new features from existing data
- Not covered in current materials

**Full Batch Gradient Descent**
- Uses the entire dataset for each parameter update
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

## G

**Generalization**
- The ability of a model to perform well on unseen data
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Generative AI**
- Models that create new content based on input
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

**Gradient**
- The vector of partial derivatives of a function
- Points in the direction of steepest increase
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

**Gradient Descent**
- An optimization algorithm that iteratively adjusts parameters to minimize loss
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

## H

**Hyperparameter**
- A configuration value set before training begins
- Examples: learning rate, batch size, epochs
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

## I

**Inference**
- Using a trained model to make predictions on new data
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Intercept**
- The value where the regression line crosses the y-axis
- Same as bias in linear regression
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

**Iteration**
- One update of model parameters
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

## L

**L1 Loss**
- The sum of absolute differences between predicted and actual values
- See: [Loss.md](Linear%20Regression/Loss.md)

**L2 Loss**
- The sum of squared differences between predicted and actual values
- See: [Loss.md](Linear%20Regression/Loss.md)

**Label**
- The target value we want to predict
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Labeled Example**
- An example that contains both features and a label
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Learning Rate**
- A hyperparameter that controls how much to change parameters in each iteration
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Linear Regression**
- A regression model that assumes a linear relationship between features and label
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

**Loss**
- A measure of how wrong the model's predictions are
- See: [Loss.md](Linear%20Regression/Loss.md)

**Loss Curve**
- A graph showing how loss changes over iterations
- See: [Gradient_Descent.md](Linear%20Regression/Gradient_Descent.md)

**Loss Function**
- A function that quantifies the difference between predicted and actual values
- See: [Loss.md](Linear%20Regression/Loss.md)

## M

**MAE (Mean Absolute Error)**
- The average of absolute differences between predicted and actual values
- See: [Loss.md](Linear%20Regression/Loss.md)

**Mean Absolute Error**
- See MAE

**Mean Squared Error**
- See MSE

**Mini-Batch**
- A subset of the training dataset larger than one example but smaller than the full dataset
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Mini-Batch Gradient Descent**
- Uses mini-batches for parameter updates
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Model**
- The mathematical representation learned from data
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**MSE (Mean Squared Error)**
- The average of squared differences between predicted and actual values
- See: [Loss.md](Linear%20Regression/Loss.md)

**Multiclass Classification**
- Classification with more than two possible outcomes
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

## N

**Neural Network**
- A model inspired by the human brain
- Not covered in current materials

**Noise**
- Random variations in data that can affect training
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

## O

**Outlier**
- A data point that differs significantly from other observations
- Can affect model performance differently depending on the loss function
- See: [Loss.md](Linear%20Regression/Loss.md)

**Overfitting**
- When a model performs well on training data but poorly on new data
- Not covered in current materials

## P

**Parameter**
- A value learned by the model during training (e.g., weights and bias)
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

**Partial Derivative**
- The derivative of a function with respect to one variable
- Used in gradient descent
- See: [Gradient Descent Calculation.md](Linear%20Regression/Gradient%20Descent%20Calculation.md)

**Prediction**
- The output of a model for a given input
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

## R

**Regression**
- Predicting a continuous numeric value
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

**Regularization**
- Techniques to prevent overfitting
- Not covered in current materials

**Reinforcement Learning**
- Learning through rewards and penalties
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

**RMSE (Root Mean Squared Error)**
- The square root of the mean squared error
- See: [Loss.md](Linear%20Regression/Loss.md)

**Root Mean Squared Error**
- See RMSE

## S

**Slope**
- The rate of change in a linear relationship
- Same as weight in linear regression
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

**Stochastic Gradient Descent (SGD)**
- Uses one example at a time for parameter updates
- See: [Hyperparameter.md](Linear%20Regression/Hyperparameter.md)

**Supervised Learning**
- Learning from labeled examples
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

## T

**Training**
- The process of adjusting model parameters to minimize loss
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

## U

**Underfitting**
- When a model is too simple to capture the underlying pattern
- Not covered in current materials

**Unlabeled Example**
- An example that contains features but no label
- See: [Supervised_Learning.md](Introduction%20to%20Machine%20learning/Supervised_Learning.md)

**Unsupervised Learning**
- Learning from unlabeled data by finding patterns
- See: [Machine_learning.md](Introduction%20to%20Machine%20learning/Machine_learning.md)

## V

**Validation**
- Evaluating model performance on data not used for training
- Not covered in current materials

## W

**Weight**
- The coefficient of a feature in a linear model
- Determines the importance of the feature
- See: [Linear_Regression.md](Linear%20Regression/Linear_Regression.md)

---

**Note**: This glossary covers terms used in the current materials. As you expand your content, you may need to add additional terms.

**Last Updated**: October 2025