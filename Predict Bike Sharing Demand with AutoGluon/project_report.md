
# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Aisha El-Ghazaly

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
That the predictions need to be exported as a csv file first in order to be submitted.

### What was the top ranked model that performed?
WeightedEnsemble_L3 with a score of -53.008791

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The histogram of the training data revealed some things.

 1. *The season doesn't affect demand. it's constant across all 4
    seasons.* 
    
 2. *Demand is highest in fair weather. when it's not too hot
    or too cold.*
    
 3. *Demand is highest when the wind is a gentle breeze.*
 
 4. *The vast majority of demand is on working days.*

Regarding additional features, I have rerun this notebook with many different features. 

In the notebook I submit I've added 

 - `hour`, `day`, `month` features extracted from the date
 - `tempfeel` extracted from `temp` which categorizes the feel of the temperature as nice or not determined by `temp` being between 12 and 28
 - `humidityfeel` is similar to `tempfeel` and is extracted from humidity. The thresholds 40 & 90 were picked from looking at the histogram.

Extensive experimentation revealed that extra features improve training score but have negative effect on the kaggle score. In previous iterations the one consistent finding is that any new features at all cause the kaggle score to drop from `1.8` to `0.6-0.7`. 

### How much better did your model preform after adding additional features and why do you think that is?
As I've mentioned, the kaggle score of the model after additional features takes a massive hit; from `1.8` to `0.6`. However the training score improves from `-53` to `-30`. My suspicion is that with autogluon being an auto ml framework, the more processing we do to the data the worse the performance; that it performs best on raw data.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Increasing the time limit had noticable effect. the kaggle score went up from `0.6` to `0.73` but nowhere near the original `1.8`.

### If you were given more time with this dataset, where do you think you would spend more time?
I would spend more time on EDA and generating/removing features to try and confirm my suspicion that autogluon works best when it's just provided with raw data and basic hyperparameters.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|root_mean_squared_error|600|best_quality|1.80299|
|add_features|root_mean_squared_error|600|best_quality|0.60979|
|hpo|root_mean_squared_error|3600|best_quality|0.74970|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
After a lot of iterations, the consistent result has been that the initial score will always be far higher than any score achieved post data processing. while feature engineering might greatly improve results in traditional ml, in auto gluon it has the opposite effect. Adjusting the hyperparameters whether it be by tuning the individual models (an experiment in the previous iterations that didn't achieve a wildly different result than what's presented here) or just increasing the time limit to let the model run longer, improves the bad score obtained after adding new features but is still nowhere near the initial score.
