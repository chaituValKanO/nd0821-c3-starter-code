# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

We have built an Logistic Regression model using the default hyperparamters. The purpose of model is to perform binary classification of people based on various input features

The target variable is >50K, <=50K.

## Intended Use

This model can be used by govt sources or private agencies to understand what are the driving factors that influence a persons income. 

## Training Data

80% of the data is used for the training purpose. The data has been cleaned, nulls have been handled. 

Numerical Features: Scaled
Categorical features: Encoded

## Evaluation Data

Evaluation data is passed through the same pipeline as training data to maintain reproducibality. 

## Metrics
Precision, Recall and F-score have been logged to understand the model performance

The test results are Precision=0.7248270561106841, recall=0.6103559870550161, fbeta=0.6626844694307801

## Ethical Considerations
As the data doesnt contain any PII data, we need not worry about any specific biases. However, the "Race" feature needs to handled cautiously. 

## Caveats and Recommendations
Build much more advanced models for better performance. even feature engg can be looked into