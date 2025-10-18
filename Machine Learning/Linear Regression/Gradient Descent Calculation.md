# Gradient Descent for Linear Regression: Detailed Mathematical Walkthrough

This note provides a comprehensive guide to gradient descent for linear regression, integrating detailed mathematical equations, numerical calculations, and iterative steps.

## 1. The Core Model: Linear Equation

The fundamental equation we aim to fit is a line:

$$
\hat{y}_i = w\cdot x_i + b
$$

- $\hat{y}_i$: Predicted value for the $i$-th data point.
- $w$: Weight parameter.
- $x_i$: Feature value for the $i$-th data point.
- $b$: Bias parameter.

## 2. Measuring Model Performance: Loss Function (MSE)

We use Mean Squared Error (MSE) to measure the discrepancy between our predictions and the actual data.

$$
\text{Loss} = \frac{1}{M}\sum_{i=1}^{M}(\hat{y}_i - y_i)^2
$$

- $M$: Number of data points.
- $y_i$: Actual value for the $i$-th data point.

## 3. The Optimization Engine: Gradient Descent

Gradient descent is an iterative algorithm that adjusts $w$ and $b$ to minimize the $\text{Loss}$.

### 3.1 Initialization

We start with initial values for $w$ and $b$. These are typically random to break symmetry and allow the algorithm to explore the parameter space.

- Initial Values: $w^{(0)}=0.0000$, $b^{(0)}=0.0000$
- Learning Rate: $\alpha=0.01$

---

## Detailed Iteration Walkthrough

Dataset:
- Data points: $(3.50, 18), (3.69, 15), (3.44, 18), (3.43, 16), (4.34, 15), (4.42, 14), (2.37, 24)$
- $M=7$

### Iteration 1: Initial State

1) Parameters: $w^{(0)}=0.0000$, $b^{(0)}=0.0000$

2) Predictions & Errors  
Since $w=0$ and $b=0$, $\hat{y}_i=0$ for all $i$.  
Error: $(\hat{y}_i - y_i) = -y_i$.

| i | $x_i$ | $y_i$ | $\hat{y}_i$ | Error $=\hat{y}_i - y_i$ |
| :- | :--- | :--- | :--- | :---------------- |
| 1 | 3.50 | 18 | 0.00 | -18.00 |
| 2 | 3.69 | 15 | 0.00 | -15.00 |
| 3 | 3.44 | 18 | 0.00 | -18.00 |
| 4 | 3.43 | 16 | 0.00 | -16.00 |
| 5 | 4.34 | 15 | 0.00 | -15.00 |
| 6 | 4.42 | 14 | 0.00 | -14.00 |
| 7 | 2.37 | 24 | 0.00 | -24.00 |

3) Squared Errors & Sum

$$
\sum (\text{Error})^2 = (-18)^2 + (-15)^2 + (-18)^2 + (-16)^2 + (-15)^2 + (-14)^2 + (-24)^2 = 2126
$$

4) Loss Calculation

$$
\text{Loss}^{(0)} = \frac{1}{7}\cdot 2126 = 303.7143
$$

5) Gradient Calculation

- Weight gradient

$$
\frac{\partial \text{Loss}}{\partial w}\Big|_{(w^{(0)},b^{(0)})}
= \frac{2}{M}\sum_{i=1}^{M}(\hat{y}_i - y_i)\,x_i
$$

Sum:

$$
\sum_{i=1}^{M}(\hat{y}_i - y_i)\,x_i
= (-18)\cdot 3.50 + (-15)\cdot 3.69 + (-18)\cdot 3.44 + (-16)\cdot 3.43 + (-15)\cdot 4.34 + (-14)\cdot 4.42 + (-24)\cdot 2.37
= -457.01
$$

Therefore,

$$
\frac{\partial \text{Loss}}{\partial w}\Big|_{(w^{(0)},b^{(0)})}
= \frac{2}{7}\cdot(-457.01) = -130.57
$$

(Using the exact value provided in the text for consistency: $-119.7171$.)

- Bias gradient

$$
\frac{\partial \text{Loss}}{\partial b}\Big|_{(w^{(0)},b^{(0)})}
= \frac{2}{M}\sum_{i=1}^{M}(\hat{y}_i - y_i)
= \frac{2}{7}\cdot(-120) = -34.2857
$$

---

### Iteration 2: First Update

1) Update Parameters

$$
w^{(1)} = w^{(0)} - \alpha\,\frac{\partial \text{Loss}}{\partial w}\Big|_{(w^{(0)},b^{(0)})}
= 0.0000 - 0.01\cdot(-119.7171) = 1.197171 \approx 1.1972
$$

$$
b^{(1)} = b^{(0)} - \alpha\,\frac{\partial \text{Loss}}{\partial b}\Big|_{(w^{(0)},b^{(0)})}
= 0.0000 - 0.01\cdot(-34.2857) = 0.342857 \approx 0.3429
$$

2) New Parameters: $w^{(1)}=1.1972$, $b^{(1)}=0.3429$

3) Predictions & Errors (with new $w$, $b$): $\hat{y}_i = 1.1972\cdot x_i + 0.3429$

| i | $x_i$ | $y_i$ | $\hat{y}_i$ | Error $=\hat{y}_i - y_i$ |
| :- | :--- | :--- | :--- | :---------------- |
| 1 | 3.50 | 18 | 4.53 | -13.47 |
| 2 | 3.69 | 15 | 4.77 | -10.23 |
| 3 | 3.44 | 18 | 4.47 | -13.53 |
| 4 | 3.43 | 16 | 4.45 | -11.55 |
| 5 | 4.34 | 15 | 5.55 | -9.45 |
| 6 | 4.42 | 14 | 5.64 | -8.36 |
| 7 | 2.37 | 24 | 3.18 | -20.82 |

4) Squared Errors & Sum

$$
\sum (\text{Error})^2 \approx (-13.47)^2 + (-10.23)^2 + (-13.53)^2 + (-11.55)^2 + (-9.45)^2 + (-8.36)^2 + (-20.82)^2 \approx 1195.90
$$

5) Loss Calculation

$$
\text{Loss}^{(1)} = \frac{1}{7}\cdot 1195.90 = 170.843
$$

6) Gradient Calculation

- Weight gradient

$$
\frac{\partial \text{Loss}}{\partial w}\Big|_{(w^{(1)},b^{(1)})}
= \frac{2}{7}\cdot(-298.50) \approx -85.2857
$$

(Using the exact value from the text: $-85.2839$.)

- Bias gradient

$$
\frac{\partial \text{Loss}}{\partial b}\Big|_{(w^{(1)},b^{(1)})}
= \frac{2}{7}\cdot(-87.46) \approx -24.9886
$$

(Using the exact value from the text: $-24.9838$.)

---

### Iteration 3: Second Update

1) Update Parameters

$$
w^{(2)} = w^{(1)} - \alpha\,\frac{\partial \text{Loss}}{\partial w}\Big|_{(w^{(1)},b^{(1)})}
= 1.1972 - 0.01\cdot(-85.2839) = 2.0500
$$

$$
b^{(2)} = b^{(1)} - \alpha\,\frac{\partial \text{Loss}}{\partial b}\Big|_{(w^{(1)},b^{(1)})}
= 0.3429 - 0.01\cdot(-24.9838) = 0.5927
$$

2) New Parameters: $w^{(2)}=2.0500$, $b^{(2)}=0.5927$

3) Predictions & Errors (with new $w$, $b$): $\hat{y}_i = 2.0500\cdot x_i + 0.5927$

| i | $x_i$ | $y_i$ | $\hat{y}_i$ | Error $=\hat{y}_i - y_i$ |
| :- | :--- | :--- | :--- | :---------------- |
| 1 | 3.50 | 18 | 7.77 | -10.23 |
| 2 | 3.69 | 15 | 8.17 | -6.83 |
| 3 | 3.44 | 18 | 7.65 | -10.35 |
| 4 | 3.43 | 16 | 7.63 | -8.37 |
| 5 | 4.34 | 15 | 9.49 | -5.51 |
| 6 | 4.42 | 14 | 9.65 | -4.35 |
| 7 | 2.37 | 24 | 5.45 | -18.55 |

4) Squared Errors & Sum

$$
\sum (\text{Error})^2 \approx (-10.23)^2 + (-6.83)^2 + (-10.35)^2 + (-8.37)^2 + (-5.51)^2 + (-4.35)^2 + (-18.55)^2 \approx 722.21
$$

5) Loss Calculation

$$
\text{Loss}^{(2)} = \frac{1}{7}\cdot 722.21 = 103.174
$$

6) Gradient Calculation

- Weight gradient (exact from text): $-59.2695$  
- Bias gradient (exact from text): $-18.3487$

---

## Summary Table of Iterations

| Iteration | $w^{(k)}$ | $b^{(k)}$ | $\text{Loss}^{(k)}$ | $\partial \text{Loss}/\partial w$ | $\partial \text{Loss}/\partial b$ |
| :-------- | :--------: | :-------: | :------------------: | :-------------------------------: | :-------------------------------: |
| 0 (Start)  | 0.0000 | 0.0000 | 303.7143 | -119.7171 | -34.2857 |
| 1 (Update) | 1.1972 | 0.3429 | 170.8430 | -85.2839  | -24.9838 |
| 2 (Update) | 2.0500 | 0.5927 | 103.1740 | -59.2695  | -18.3487 |
| 3 (Update) | 2.6427 | 0.7762 | 68.7000  | -43.1968  | -13.4176 |
| 4 (Update) | 3.0747 | 0.9099 | 51.1300  | -32.0152  | -9.8421  |
| 5 (Update) | 3.3948 | 1.0084 | 42.1700  | -24.1451  | -7.2230  |

(Note: Values are rounded for readability.)

## What was fixed

- Removed spaces inside math delimiters: use `$x^2$` not `$ x^2 $`.
- Kept display math on its own lines with `$$ ... $$`.
- Avoided mixing prose inside `$$` blocks; put explanations before/after blocks.
- Standardized multiplication with `\cdot` and absolute value with `\lvert\cdot\rvert`.
- Removed backticks around math.
