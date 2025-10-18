## Linear regression: Loss

Loss is a numerical metric that describes how wrong a model's predictions are. Loss measures the distance between the model's predictions and the actual labels. The goal of training a model is to minimize the loss, reducing it to its lowest possible value.

> **Related Topics:**
> * [Linear Regression](Linear_Regression.md) - The model whose predictions we're evaluating
> * [Gradient Descent](Gradient_Descent.md) - How we minimize loss
> * [Practical Implementation](Practical_Implementation.md) - Code examples for calculating loss

### AI-generated Key Takeaways
* Loss is a numerical value indicating the difference between a model's predictions and the actual values.
* The goal of model training is to minimize loss, bringing it as close to zero as possible.
* Two common methods for calculating loss are Mean Absolute Error (MAE) and Mean Squared Error (MSE), which differ in their sensitivity to outliers.
* Choosing between MAE and MSE depends on the dataset and how you want the model to handle outliers, with MSE penalizing them more heavily.

In the following image, you can visualize loss as arrows drawn from the data points to the model. The arrows show how far the model's predictions are from the actual values.

![Contextual Image 2](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/loss-lines.png)

Figure 8. Loss is measured from the actual value to the predicted value.

### Distance of loss
In statistics and machine learning, loss measures the difference between the predicted and actual values. Loss focuses on the distance between the values, not the direction. For example, if a model predicts 2, but the actual value is 5, we don't care that the loss is negative ($2-5=-3$). Instead, we care that the distance between the values is $3$. Thus, all methods for calculating loss remove the sign.

The two most common methods to remove the sign are the following:
* Take the absolute value of the difference between the actual value and the prediction.
* Square the difference between the actual value and the prediction.

### Types of loss
In linear regression, there are five main types of loss, which are outlined in the following table.

| Loss type | Definition | Equation |
| :--- | :--- | :--- |
| **L1 loss** | The sum of the absolute values of the difference between the predicted values and the actual values. | $\sum \lvert actual\ value - predicted\ value \rvert$ |
| **Mean absolute error (MAE)** | The average of L1 losses across a set of N examples. | $\frac{1}{N} \sum \lvert actual\ value - predicted\ value \rvert$ |
| **L2 loss** | The sum of the squared difference between the predicted values and the actual values. | $\sum(actual\ value - predicted\ value)^2$ |
| **Mean squared error (MSE)** | The average of L2 losses across a set of N examples. | $\frac{1}{N} \sum (actual\ value - predicted\ value)^2$ |
| **Root mean squared error (RMSE)** | The square root of the mean squared error (MSE). | $\sqrt{\frac{1}{N} \sum (actual\ value - predicted\ value)^2}$ |

The functional difference between L1 loss and L2 loss (or between MAE/RMSE and MSE) is squaring. When the difference between the prediction and label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.

Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value.

> **Note:** MAE and RMSE can differ quite widely. MAE represents the average prediction error, whereas RMSE represents the "spread" of the errors, and is more skewed by larger errors.

When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE.

### Calculating loss example
Using the previous best fit line, we'll calculate L2 loss for a single example. From the best fit line, we had the following values for weight and bias:
* Weight: $-4.6$
* Bias: $34$

If the model predicts that a 2,370-pound car gets 23.1 miles per gallon, but it actually gets 26 miles per gallon, we would calculate the L2 loss as follows:

> **Note:** The formula uses 2.37 because the graphs are scaled to 1000s of pounds

| Value | Equation | Result |
| :--- | :--- | :--- |
| Prediction | $bias + (weight * feature\ value) \rightarrow 34 + (-4.6*2.37)$ | $23.1$ |
| Actual value | $label$ | $26$ |
| L2 loss | $(actual\ value - predicted\ value)^2 \rightarrow (26 - 23.1)^2$ | $8.41$ |

In this example, the L2 loss for that single data point is 8.41.

### Choosing a loss
Deciding whether to use MAE or MSE can depend on the dataset and the way you want to handle certain predictions. Most feature values in a dataset typically fall within a distinct range. For example, cars are normally between 2000 and 5000 pounds and get between 8 to 50 miles per gallon. An 8,000-pound car, or a car that gets 100 miles per gallon, is outside the typical range and would be considered an outlier.

An outlier can also refer to how far off a model's predictions are from the real values. For instance, 3,000 pounds is within the typical car-weight range, and 40 miles per gallon is within the typical fuel-efficiency range. However, a 3,000-pound car that gets 40 miles per gallon would be an outlier in terms of the model's prediction because the model would predict that a 3,000-pound car would get around 20 miles per gallon.

When choosing the best loss function, consider how you want the model to treat outliers. For instance, MSE moves the model more toward the outliers, while MAE doesn't. L2 loss incurs a much higher penalty for an outlier than L1 loss. For example, the following images show a model trained using MAE and a model trained using MSE. The red line represents a fully trained model that will be used to make predictions. The outliers are closer to the model trained with MSE than to the model trained with MAE.

![Contextual Image 3](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mse.png)

Figure 9. A model trained with MSE moves the model closer to the outliers.

![Contextual Image 4](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mae.png)

Figure 10. A model trained with MAE is farther from the outliers.

Note the relationship between the model and the data:
* **MSE.** The model is closer to the outliers but further away from most of the other data points.
* **MAE.** The model is further away from the outliers but closer to most of the other data points.

> **Click the icon for more guidelines on choosing a loss metric**
> * **Choose MSE:**
>   * If you want to heavily penalize large errors.
>   * If you believe the outliers are important and indicative of true data variance that the model should account for.
>
>   Note: The mathematical properties of MSE often make optimization smoother. Root Mean Squared Error (RMSE) is often used to get the error back into the same units as the label.
> * **Choose MAE:**
>   * If your dataset has significant outliers that you don't want to overly influence the model. MAE is more robust.
>   * If you prefer a loss function that is more directly interpretable as the average error magnitude.
>
> In practice, your metric choice can also depend on the specific business problem and what kind of errors are more costly.

### Check Your Understanding
Consider the following two plots of a linear model fit to a dataset:

![Contextual Image 5](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/mse-left.png) ![Contextual Image 6](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/mse-right.png)

Which of the two linear models shown in the preceding plots has the higher Mean Squared Error (MSE) when evaluated on the plotted data points?

* The model on the left.
  
  The six examples on the line incur a total loss of 0. The four examples not on the line are not very far off the line, so even squaring their offset still yields a low value: $MSE = \frac{0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 0^2} {10} = 0.4$
* **The model on the right.**
  
  The eight examples on the line incur a total loss of 0. However, although only two points lay off the line, both of those points are twice as far off the line as the outlier points in the left figure. Squared loss amplifies those differences, so an offset of two incurs a loss four times as great as an offset of one: $MSE = \frac{0^2 + 0^2 + 0^2 + 2^2 + 0^2 + 0^2 + 0^2 + 2^2 + 0^2 + 0^2} {10} = 0.8$

### Key terms:
* [Mean absolute error (MAE)](../Glossary.md#mae-mean-absolute-error)
* [Mean squared error (MSE)](../Glossary.md#mse-mean-squared-error)
* [L1 Loss](../Glossary.md#l1-loss)
* [L2 Loss](../Glossary.md#l2-loss)
* [Loss](../Glossary.md#loss)
* [Outlier](../Glossary.md#outlier)
* [Prediction](../Glossary.md#prediction)

---

**Previous:** [Linear Regression](Linear_Regression.md)

**Next:** [Gradient Descent](Gradient_Descent.md)

**Related:** [Practical Implementation](Practical_Implementation.md) | [Glossary](../Glossary.md) | [Roadmap](../Roadmap.md)

---

<small>Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates. Last updated 2025-08-25 UTC.</small>
