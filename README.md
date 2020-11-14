# Optimizing an ML Pipeline in Azure

Table of contents
=================

<!--ts-->
  * [Overview](##overview)
  * [Summary](##summary)
  * [Dataset](##Dataset)
  * [Scikit-learn pipeline](##scikit-learn-pipeline)
  * [AutoML pipeline](##autoML-pipeline)
  * [Pipelines comparison](##pipeline-comparison)
  * [Future work](##future-work)
  * [Cleanup](##proof-of-cluster-clean-up)
<!--te-->

## Overview
This project is part of the Udacity Azure ML developer Nanodegree program. In this project, we build and optimize an Azure ML pipeline using the Python SDK, Azure Hyper parameter tuning module and Scikit-learn logistic regression model. This model is then compared to an Azure AutoML run. 

## Summary
Comparison analysis is performed between hyper parameter tuning using Azure HyperDrive module for logistic regression (using Scikit library) and using AutoML feature provided as part of Microsoft Azure on same data set.
The best performing model is found using AutoML as AutoML evaluates various classification algorithms which automates lot of repeatable data science tasks.

## Dataset
Data set used for this project is classic marketing bank dataset uploaded originally in the UCI Machine Learning Repository. The dataset gives you information about a marketing campaign of a financial institution in which we will have to analyze in order  to find ways to look for future strategies in order to improve future marketing campaigns for the bank. Model will predict if customer will subscribe to fixed deposit or not.

## Scikit-learn Pipeline
Logistic regression is used for binary classification for bank market analysis. Data reading, cleaning, transformation and model training is performed by software engineer using pandas and scikit in train.py. `train.py` accepts `--C` (inverse of regularization effect) and `--max_iter` as arguments and is used by hyper drive config module for auto hyper parameter turnning in HyperDrive run. `C` are sampled using `choice(0.001, 0.01, 0.1, 1, 10, 100)`. Small `C` values correspond to lot of regularization and hight values of `C` corresponds to less regularization. During the training, the algorithm tries to minimize the loss. It always checks its convergence on computing the difference between loss at present iteration to its previous iteration. To limit the time taken to convergence `max_iter` are sampled using `choice(50, 100, 200, 400, 600, 900, 1000)` which covers low values and high values for `max_iter` values.

Logistic regression is a classification algorithm which assumes linear seperation i.e., straight line boundary for binary classification. Boundary line coefficents are found during training. (for example in 2-d model finds slope and intercept values during training)

Random parameter sampling is chosen values in hyper parameters value space. This allows best hyper-parameters to be chosen at random and there is high probability that best values are chosen fast. In the context of study Grid parameter sampling should have been worked as I have used only choice hyper parameters, but in general random parameter sampling has high probability of finding best values fast. In present context Bayesian sampling method should make much differnce but in general Bayesian optimization is used if we searched some points randomly and knew some of them were more promising than else, we will look around promising items. Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space.

By specifying early termination policy we can automatically terminate poorly performing runs. Early termination improves computational efficiency.Bandit early termination policy is used to stop training if performance of current run is not with in the best run limits to avoid resource usage. Median stopping is an early termination policy based on running averages of primary metrics reported by the runs. This policy computes running averages across all training runs and terminates runs with primary metric values worse than the median of averages. I have choosen Bandit early for aggressive termination, where as median stopping can be used if we don't want aggresive termination.

It is observed that best parameter values for **C** is `0.1` and **max_iter** is `200` and best accuracy achieved is `0.91237`

## AutoML Pipeline
Machine learning process is a labour intensive as it requires number of runs to get right model. AutoML will explore various combinations of features, algorithms, hyperparameters, and score each training pipeline on the primary metirc. AutoML peforms various actions like feature generation, dataset cross validation split, data gaurd rails which check if class are rightly balanced or not Missing feature values, and cardinality checks. We can also compare various algorithsm as AutoML continues its exploration. AutoML identifies best performing model. Beyond primary target metric, we can also review a comprehensive set of performance metrics and charts to further access the model performance. For same data set used for Scikit-learn pipeline above, with AutoML best performance is achieved is `0.9161` using `PrefittedSoftVotingClassififer`. (Output result can be seen in Udacity-Project.ipnyb)

## Pipeline comparison
HyperDrive module requires data science engineer to perform data cleaning, data transformation, feature engineering, select the classifier model, and range of parameter values where search happens. If wrong model is chosen or range of parameter values are not specified correctly there is high chance that model is flawed. Here training time and resources required are less compared to AutoML module. This step can be used after AutoML finds best model and use that idea to further tune parameters according to domain knowledge if required.

Machine learning process is a labour intensive as it requires number of runs to get right model. AutoML automates this process by analyzing various models, automates feature engineering, and cross validation. There is high chance that performance metric is higher than Hyperdrive module as we have observed in present context.

## Future work
AutoML output showed that data is imbalanced. It is possible that accuracy is not right metric to use. Use different metric like F1 score or AOC metric. Understanding how model is making predictions, and how to deploy trained models. By understanding how model is making predictions, will help us in understanding collect right data, able to explain why certain decisions are taken and corrections required if any can be made in collecting data.

## Proof of cluster clean up
![Delete cluster](https://github.com/venkataravikumaralladi/ComparingManualHyperParameterTuningAndAutoMLInAzure/blob/master/DeleteClustersnapshot.png)

