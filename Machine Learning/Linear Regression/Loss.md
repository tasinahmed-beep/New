# Linear regression: Loss

> **💡 AI-generated Key Takeaways**
>
> *   Loss is a numerical value indicating the difference between a model's predictions and the actual values [web:74].
> *   The goal of model training is to minimize loss, bringing it as close to zero as possible [web:74].
> *   Two common methods for calculating loss are Mean Absolute Error (MAE) and Mean Squared Error (MSE), which differ in their sensitivity to outliers [web:74].
> *   Choosing between MAE and MSE depends on the dataset and how you want the model to handle outliers, with MSE penalizing them more heavily [web:74].

Loss is a numerical metric that describes how wrong a model's predictions are [web:74]. Loss measures the distance between the model's predictions and the actual labels [web:74]. The goal of training a model is to minimize the loss, reducing it to its lowest possible value [web:74].

In the following image, you can visualize loss as arrows drawn from the data points to the model [web:74]. The arrows show how far the model's predictions are from the actual values [web:74].

![Contextual Image 2](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/loss-lines.png)
Figure 8. Loss is measured from the actual value to the predicted value [web:74].

### Distance of loss

In statistics and machine learning, loss measures the difference between the predicted and actual values [web:74]. Loss focuses on the distance between the values, not the direction [web:74]. For example, if a model predicts 2, but the actual value is 5, we don't care that the loss is negative (\( 2-5=-3 \)) [web:74]. Instead, we care that the distance between the values is \( 3 \) [web:74]. Thus, all methods for calculating loss remove the sign [web:74].

The two most common methods to remove the sign are the following:

*   Take the absolute value of the difference between the actual value and the prediction [web:74].
*   Square the difference between the actual value and the prediction [web:74].

### Types of loss

In linear regression, there are five main types of loss, which are outlined in the following table [web:74].

| Loss type                | Definition                                                                 | Equation                                                                 |
|--------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **L1 loss**              | The sum of the absolute values of the difference between the predicted values and the actual values. | \( \sum \lvert \text{actual value} - \text{predicted value} \rvert \) |
| **Mean absolute error (MAE)** | The average of L1 losses across a set of N examples.                       | \( \frac{1}{N} \sum \lvert \text{actual value} - \text{predicted value} \rvert \) |
| **L2 loss**              | The sum of the squared difference between the predicted values and the actual values. | \( \sum (\text{actual value} - \text{predicted value})^2 \) |
| **Mean squared error (MSE)** | The average of L2 losses across a set of N examples.                       | \( \frac{1}{N} \sum (\text{actual value} - \text{predicted value})^2 \) |
| **Root mean squared error (RMSE)** | The square root of the mean squared error (MSE).                           | \( \sqrt{\frac{1}{N} \sum (\text{actual value} - \text{predicted value})^2} \) |

The functional difference between L1 loss and L2 loss (or between MAE/RMSE and MSE) is squaring [web:74]. When the difference between the prediction and label is large, squaring makes the loss even larger [web:74]. When the difference is small (less than 1), squaring makes the loss even smaller [web:74].

Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value [web:74].  
**Note:** MAE and RMSE can differ quite widely [web:74]. MAE represents the average prediction error, whereas RMSE represents the "spread" of the errors, and is more skewed by larger errors [web:74].  
When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE [web:74].

### Calculating loss example

Using the previous best fit line, we'll calculate L2 loss for a single example [web:74]. From the best fit line, we had the following values for weight and bias:

*   \( \small{\text{Weight: -4.6}} \)
*   \( \small{\text{Bias: 34}} \)

If the model predicts that a 2,370-pound car gets 23.1 miles per gallon, but it actually gets 26 miles per gallon, we would calculate the L2 loss as follows [web:74]:  
**Note:** The formula uses 2.37 because the graphs are scaled to 1000s of pounds [web:74]

| Value       | Equation                                                                 | Result |
|-------------|--------------------------------------------------------------------------|--------|
| Prediction | \( \small{\text{bias + (weight * feature value)} \rightarrow 34 + (-4.6*2.37)} \) | \( \small{23.1} \) |
| Actual value | \( \small{\text{label}} \)                                               | \( \small{26} \) |
| L2 loss     | \( \small{ (\text{actual value - predicted value})^2 \rightarrow (26 - 23.1)^2 } \) | \( \small{8.41} \) |

In this example, the L2 loss for that single data point is 8.41 [web:74].

### Choosing a loss

Deciding whether to use MAE or MSE can depend on the dataset and the way you want to handle certain predictions [web:74]. Most feature values in a dataset typically fall within a distinct range [web:74]. For example, cars are normally between 2000 and 5000 pounds and get between 8 to 50 miles per gallon [web:74]. An 8,000-pound car, or a car that gets 100 miles per gallon, is outside the typical range and would be considered an outlier [web:74].

An outlier can also refer to how far off a model's predictions are from the real values [web:74]. For instance, 3,000 pounds is within the typical car-weight range, and 40 miles per gallon is within the typical fuel-efficiency range [web:74]. However, a 3,000-pound car that gets 40 miles per gallon would be an outlier in terms of the model's prediction because the model would predict that a 3,000-pound car would get around 20 miles per gallon [web:74].

When choosing the best loss function, consider how you want the model to treat outliers [web:74]. For instance, MSE moves the model more toward the outliers, while MAE doesn't [web:74]. L2 loss incurs a much higher penalty for an outlier than L1 loss [web:74]. For example, the following images show a model trained using MAE and a model trained using MSE [web:74]. The red line represents a fully trained model that will be used to make predictions [web:74]. The outliers are closer to the model trained with MSE than to the model trained with MAE [web:74].

![Contextual Image 3](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mse.png)  
Figure 9. A model trained with MSE moves the model closer to the outliers [web:74].

![Contextual Image 4](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mae.png)  
Figure 10. A model trained with MAE is farther from the outliers [web:74].

Note the relationship between the model and the data:

*   **MSE.** The model is closer to the outliers but further away from most of the other data points [web:74].
*   **MAE.** The model is further away from the outliers but closer to most of the other data points [web:74].

> Choose MSE:
>
> *   If you want to heavily penalize large errors [web:74].
> *   If you believe the outliers are important and indicative of true data variance that the model should account for [web:74].
>
> **Note:** The mathematical properties of MSE often make optimization smoother [web:74]. Root Mean Squared Error (RMSE) is often used to get the error back into the same units as the label [web:74].
>
> Choose MAE:
>
> *   If your dataset has significant outliers that you don't want to overly influence the model [web:74]. MAE is more robust [web:74].
> *   If you prefer a loss function that is more directly interpretable as the average error magnitude [web:74].
>
> In practice, your metric choice can also depend on the specific business problem and what kind of errors are more costly [web:74].

### Check Your Understanding

Consider the following two plots of a linear model fit to a dataset [web:74]:

![Contextual Image 5](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/mse-left.png)  
![Contextual Image 6](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/mse-right.png)

Which of the two linear models shown in the preceding plots has the higher Mean Squared Error (MSE) when evaluated on the plotted data points [web:74]?

*   The model on the left.  
    The six examples on the line incur a total loss of 0 [web:74]. The four examples not on the line are not very far off the line, so even squaring their offset still yields a low value: \( \text{MSE} = \frac{0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 1^2 + 0^2 + 0^2}{10} = 0.4 \) [web:74]
*   **The model on the right.**  
    The eight examples on the line incur a total loss of 0 [web:74]. However, although only two points lay off the line, both of those points are twice as far off the line as the outlier points in the left figure [web:74]. Squared loss amplifies those differences, so an offset of two incurs a loss four times as great as an offset of one: \( \text{MSE} = \frac{0^2 + 0^2 + 0^2 + 2^2 + 0^2 + 0^2 + 0^2 + 2^2 + 0^2 + 0^2}{10} = 0.8 \) [web:74]

### Key terms:

*   Mean absolute error (MAE) [web:74]
*   Mean squared error (MSE) [web:74]
*   L1 [web:74]
*   L2 [web:74]
*   Loss [web:74]
*   Outlier [web:74]
*   Prediction [web:74]
