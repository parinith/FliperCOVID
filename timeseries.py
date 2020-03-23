#THIS IS TIMESERIES MODEL PREDICTING THE diuresis COLUMN OF TRAIN FOR 27/3/2020
import numpy as np
import pandas as pd

df = pd.read_csv('p2.csv')
de = pd.read_csv('Diuresis.csv')
ID = df['people_ID']
df.drop("people_ID", axis = 1, inplace = True)
df = df.T
temp = np.mean(df,axis=1)
ANS =[]

#autoarima model for time series
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(temp,  seasonal=True,
                            trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

for i in range(10714):
    stepwise_model.fit(df[i])
    ANS.append(stepwise_model.predict(1))
    
d = pd.DataFrame(ANS) 
df_su = pd.concat([ID, de], axis=1, sort=False)
df_su = pd.DataFrame({'people_ID': ID, 'Diuresis': de})
print(df_su.head())
df_su.to_csv('Duiresis.csv',index=False)