# Store_Classification_Project

The aims of the project were to develop machine learning models that will identify whether stores will perform well or poorly. This was a binary classification task and store performance was classed as either ‘good’ or ‘bad’.

A model that can accurately predict store performance can be used to determine whether future stores will be profitable and help in deciding where to build new World of Bargains stores.
While it is important that the model can accurately predict stores that perform well, it is more important that the model is able to correctly identify poorly performing stores. This is to avoid building stores which may not be profitable and potentially impacting on the profitability and growth of the company. 

The dataset used consisted of 19 attributes of 136 World of Bargains stores. There were 18 dependent variables and the independent variable was Performance.

The raw data was processed as follows:
- The Car Park variable was found to contain two minority values of Y and N which are believed to be incorrectly recorded values. These were recoded to the correct values of 'Yes' and 'No', and now only 'Yes' and 'No' values remain.
- Store ID, Town, and Manager name were dropped as there were too many unique values. These variables won’t be helpful in data mining because the goal is to find general patterns from specific data; but no such patterns can exist if each data point is completely different. 
- Country excluded because the UK has a large majority value i.e most of the data is concentrated in the UK. There are only a small number of values in France
- The variables measuring population size at different distances from the store were looked at in more detail. This is because they could be correlated as they have a very similar measurement. The 30 min population variable was also dropped as it was highly correlated with the 10 and 20 min population variables - but was the least correlated with the target variable. 
- The Staff variable was highly skewed towards 0, with a few values over 100. The mean of this variable was 13 so this would tell me that the higher values are outliers. It is not possible to collect more data so these higher values were excluded from model development. This was done as outliers can affect the data mining process and give incorrect results. This was recoded so only values below 100 were included. It was also seen that there were negative values in this dataset. As it is not possible to have negative numbers in the staff variable these values were also excluded,

 
As this is a classification problem a common cost function to use would be the measure of accuracy. However the problem with accuracy in this context is that the cost of getting a false negative and a false positive must be the same. In the data set the target variable has a roughly even split of values in ‘good’ and ‘bad’ performance; however there are slightly more values classified as ‘good’. Also in this task it may be slightly more important to catch false positives, to avoid building stores that aren't profitable. In this case I will use the Area Under the Curve (AUC), also known as AUROC (Area Under the ROC curve) as a measure of which model is best. The ROC curve is a graph to display the True Positive and False Positive rates at all possible classification thresholds.The AUROC is the calculation of the area under the ROC curve. This was chosen because it gives a good indication of the performance of the model. A value of 0.5 indicates no discriminative power, whereas 1 is a perfect model. 

EvalML was implemented to test classification models. EvalML is an AutoML library that builds, optimises and evaluates machine learning models. This includes data processing, feature engineering and hyperparameter tuning.

The most successful model was Extra Trees Classifier which had a training AUC of 0.783.
The hyper parameters of the Extra Trees Classifier which were selected for final model testing were n_estimators = 398, max_features = log2, max_depth= 6, min_samples_split = 2, min_weight_fraction_leaf = 0.0, and n_jobs = -1.

The final test AUC was 0.65. This is not considered a good, as AUC values close to 0.5 have no discriminative power. 

We can look at a confusion matrix to describe the performance of a classification model. Before analysing the confusion matrix it would be useful to define the terms false/true positive/negative. In this report when I am referring to a false positive I am referring to the number of times the model incorrectly identified poorly performing stores as performing well. True positive is the number of times the model correctly identified stores that performed well as good. False negative is the number of times the model incorrectly identified stores that performed well as bad. True negative is the number of times the model correctly identified stores that do poorly as bad. 
These figures show that 40% of the time the model correctly classified store performance as bad. These are the true negatives.
67% of the time the model correctly classified store performance as ‘good’. These are the true positives. 
60% of the time the model incorrectly classified store performance as ‘good’, as they were in fact bad performing stores. These are the false negatives. 
33% of the time the model incorrectly classified store performance as ‘bad’, as they were in fact good performing stores. These are the false positives.

![image](https://github.com/llamb993/Store_Classification_Project/assets/66467630/6eb63b7b-6c62-49bb-84f7-67dbb5694ce4)

World of Bargains would want to avoid false positives most; that is it would be very important that the model that does not incorrectly label poor performing stores as ‘good’. This is because they would not want to build stores that actually perform ‘poorly’ and therefore are not profitable, and cause World of Bargains to lose money. As 40% of the time stores that performed bad were correctly classified as bad, I would say we do not have a good model to predict store performance. 

EvalML have default model figurations. It is possible these applied configurations were not suited to the data and the data science task. Future methods could look to test individual classification models with hyperparameter tuning. Some examples of models that could be tested are logistic regression, decision tree, random forrest, neural networks and XGBoost.
EvalML also has a cost matrix function while training models. This applies more weight to a sector of the confusion matrix of your choosing. This could be applied in the future with the most weight on false positives. This may give a model that is better in identifying false positives and therefore more suited to the data science task.  
