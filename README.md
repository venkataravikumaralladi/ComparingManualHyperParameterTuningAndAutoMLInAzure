# Optimizing an ML Pipeline in Azure

Table of contents
=================

<!--ts-->
  * [Overview] (#overview)
  * [Summary] (#summary)
  * [Scikit-learn pipeline] (#scikit-learn-pipeline)
  * [AutoML pipeline] (#autoML-pipeline)
  * [Pipelines comparison] (#pipeline-comparison)
  * [Future work] (#future-work)
  * [Cleanup] (#proof-of-cluster-clean up)
<!--te-->

## Overview
This project is part of the Udacity Azure ML developer Nanodegree program. In this project, we build and optimize an Azure ML pipeline using the Python SDK, Azure Hyper parameter tuning module and Scikit-learn logistic regression model. This model is then compared to an Azure AutoML run.

## Summary
Comparison analysis is performed between hyper parameter tuning using Azure HyperDrive module for logistic regression (using Scikit library) and using AutoML feature provided as part of Microsoft Azure on same data set.
The best performing model is found using AutoML as AutoML evaluates various classification algorithms which automates lot of repeatable data science tasks.


## Scikit-learn Pipeline
Logistic regression is used for binary classification for bank market analysis. Data reading, cleaning, transformation and model training is performed by software engineer using pandas and scikit in train.py. Hyper parameter tuning for parameters “C” (inverse of regularization effect) and “max_iter” are performed using HyperDrive run.

Random parameter sampling is chosen values in hyper parameters value space. This allows best hyper-parameters to be chosen at random and there is high probability that best values are chosen fast

Bandit early termination policy is used to stop training if performance of current run is not with in the best run limits to avoid resource usage. It is observed that best parameter values for “C” is 0.1 and “max_iter” is 200 and best accuracy achieved is 0.91237

## AutoML Pipeline
AutoML analyzed various model algorithms to get the best prediction.  From AutoML best performance is achieved is 0.9161 using “PrefittedSoftVotingClassififer.

## Pipeline comparison
HyperDrive module requires data science engineer to perform data cleaning, data transformation, feature engineering, select the classifier model, and range of parameter values where search happens. If wrong model is chosen or range of parameter values are not specified correctly there is high chance that model is flawed. Here training time and resources required are less compared to AutoML module. This step can be used after AutoML finds best model and use that idea to further tune parameters according to domain knowledge if required.

AutoML module analyze various models, automates feature engineering, and cross validation. There is high chance that performance metric is higher than Hyperdrive module. Cons of Auto ML is it requires high training time and resources.

## Future work
AutoML output showed that data is imbalanced. It is possible that accuracy is not right metric to use. Use different metric like F1 score or AOC metric. Understanding how model is making predictions, and how to deploy trained models. By understanding how model is making predictions, will help us in understanding collect right data, able to explain why certain decisions are taken and corrections required if any can be made in collecting data.

## Proof of cluster clean up
![Delete cluster](https://github.com/venkataravikumaralladi/ComparingManualHyperParameterTuningAndAutoMLInAzure/blob/master/DeleteClustersnapshot.png)

