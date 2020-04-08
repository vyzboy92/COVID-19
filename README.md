# COVID19 ML
A regression based model to forecast number of people confimed, recovered and died of COVID19

## Requirements
1. Flask
2. Pandas
3. lightgbm
4. sklearn
5. numpy

## Run the prediction
``` python LGBM_regressor.py <number-of-days-to-predict>```

This will initiate the data cleanup process which will use ```covid_19_data.csv``` file to extract number of confirmed, dead and recovered cases and fits an LGBM Regressor.

Then this model will be used to predict confirmed cases, recovered cases and number of deaths for upcoming days.  
