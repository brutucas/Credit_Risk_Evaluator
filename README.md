# Predicting Credit Risk with Supervised Machine Learning

In this assignment, I will be building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

## Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans and buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

I will be using this data to create machine learning models to classify the risk level of given loans. Specifically, I will be comparing the Logistic Regression model and K-Nearest Neighbors.

## Instructions
Retrieve the data
In the Generator folder in Resources, there is a GenerateData.ipynb notebook that will download data from LendingClub and output two CSVs:

2019loans.csv
2020Q1loans.csv

I will be using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high-risk and low-risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric
I will create a training set from the 2019 loans using pd.get_dummies() to convert the categorical data to numeric columns. Similarly, I will create a testing set from the 2020 loans, also using pd.get_dummies(). Note that there are categories in the 2019 loans that do not exist in the testing set. If I fit a model to the training set and try to score it on the testing set as is, I will get an error. I need to use code to fill in the missing categories in the testing set.

## Consider the models
I will be creating and comparing two models on this data: a logistic regression, and a K-Nearest Neighbors. Before I create, fit, and score the models, I will make a prediction as to which model I think will perform better. I do not need to be correct! I will write down (in markdown cells in my Jupyter Notebook or in a separate document) my prediction and provide justification for my educated guess.

## Fit a LogisticRegression model and KNN model
I will create a LogisticRegression model, fit it to the data, and print the model's score. I will do the same for a K-Nearest Neighbors model. I may choose any starting hyperparameters I like. Which model performed better? How does that compare to my prediction?

## Revisit the Preprocessing: Scale the data
The data going into these models was never scaled, an important step in preprocessing. I will use StandardScaler to scale the training and testing sets. Before re-fitting the LogisticRegression and K-Nearest Neighbors models on the scaled data, I will make another prediction about how I think scaling will affect the accuracy of the models. I will write my predictions down and provide justification.

I will fit and score the LogisticRegression and K-Nearest Neighbors models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to my prediction? I will write down my results and thoughts.

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)
