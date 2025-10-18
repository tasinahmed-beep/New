# Introduction

**Machine learning (ML)**: It is mainly a process to train a model by huge amount of data to make predictions or to generate content like text, image, video etc.

## Example of models

- **ChatGPT models**
  - GPT5, GPT4, GPT3, O1, O3, O4
- **Gemini models**
  - Gemini 2.5 pro, Gemini 2.5 Flash, Gemini 2.5 Flash Lite
- **Claude models**
  - Sonnet 3.5, Sonnet 3.7, Sonnet 4

## The Power of Machine Learning

Machine learning (ML) powers some of the most important technologies we use, from translation apps to autonomous vehicles. This course explains the core concepts behind ML.

ML offers a new way to solve problems, answer complex questions, and create new content. ML can predict the weather, estimate travel times, recommend songs, auto-complete sentences, summarize articles, and generate never-seen-before images.

### ML Approach vs. Traditional Approach

For example, suppose we wanted to create an app to predict rainfall. We could use either a traditional approach or an ML approach.

- **Traditional Approach:** We'd create a physics-based representation of the Earth's atmosphere and surface, computing massive amounts of fluid dynamics equations. This is incredibly difficult.

- **ML Approach:** We would give an ML model enormous amounts of weather data until the ML model eventually *learned* the mathematical relationship between weather patterns that produce differing amounts of rain. We would then give the model the current weather data, and it would predict the amount of rain.



# Types of ML Systems

ML systems fall into one or more of the following categories based on how they learn to make predictions or generate content:

- Supervised learning
- Unsupervised learning
- Reinforcement learning
- Generative AI

---

## Supervised learning

**Supervised learning** models can make predictions after seeing lots of data with the correct answers and then discovering the connections between the elements in the data that produce the correct answers. This is like a student learning new material by studying old exams that contain both questions and answers. Once the student has trained on enough old exams, the student is well prepared to take a new exam. These ML systems are "supervised" in the sense that a human gives the ML system data with the known correct results.

Two of the most common use cases for supervised learning are **regression** and **classification**.

### Regression

A **regression model** predicts a numeric value. For example, a weather model that predicts the amount of rain, in inches or millimeters, is a regression model.

See the table below for more examples of regression models:

| Scenario | Possible input data | Numeric prediction |
| :--- | :--- | :--- |
| **Future house price** | Square footage, zip code, number of bedrooms and bathrooms, lot size, mortgage interest rate, property tax rate, construction costs, and number of homes for sale in the area. | The price of the home. |
| **Future ride time** | Historical traffic conditions (gathered from smartphones, traffic sensors, ride-hailing and other navigation applications), distance from destination, and weather conditions. | The time in minutes and seconds to arrive at a destination. |

### Classification

**Classification models** predict the likelihood that something belongs to a category. Unlike regression models, whose output is a number, classification models output a value that states whether or not something belongs to a particular category. For example, classification models are used to predict if an email is spam or if a photo contains a cat.

Classification models are divided into two groups: **binary classification** and **multiclass classification**.

- **Binary classification models** output a value from a class that contains only two values, for example, a model that outputs either *rain* or *no rain*.
- **Multiclass classification models** output a value from a class that contains more than two values, for example, a model that can output either *rain*, *hail*, *snow*, or *sleet*.


## Unsupervised learning

**Unsupervised learning** models make predictions by being given data that does not contain any correct answers. An unsupervised learning model's goal is to identify meaningful patterns among the data. In other words, the model has no hints on how to categorize each piece of data, but instead it must infer its own rules.

A commonly used unsupervised learning model employs a technique called **clustering**. The model finds data points that demarcate natural groupings.

![An ML model clustering similar data points.](https://developers.google.com/static/machine-learning/intro-to-ml/images/clustering-02.png)
*Figure 1. An ML model clustering similar data points.*

![Groups of clusters with natural demarcations.](https://developers.google.com/static/machine-learning/intro-to-ml/images/clustering-04.png)
*Figure 2. Groups of clusters with natural demarcations.*

Clustering differs from classification because the categories aren't defined by you. For example, an unsupervised model might cluster a weather dataset based on temperature, revealing segmentations that define the seasons. You might then attempt to name those clusters based on your understanding of the dataset.

![An ML model clustering similar weather patterns.](https://developers.google.com/static/machine-learning/intro-to-ml/images/clustering-01.png)
*Figure 3. An ML model clustering similar weather patterns.*

![Clusters of weather patterns labeled as snow, sleet, rain, and no rain.](https://developers.google.com/static/machine-learning/intro-to-ml/images/clustering-03.png)
*Figure 4. Clusters of weather patterns labeled as snow, sleet, rain, and no rain.*