import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats


train = pd.read_csv('Train_dataset.csv')
test = pd.read_csv('Test_dataset.csv')
ypp = pd.read_csv('compare1.csv')
ypp = ypp['Parikshith\'s values']

ntrain = train.shape[0]
ntest = test.shape[0]


#Save the 'Id' column
train_ID = train['people_ID']
test_ID = test['people_ID']

di = pd.read_csv('Diuresis.csv')
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("people_ID", axis = 1, inplace = True)
test.drop("people_ID", axis = 1, inplace = True)


train.Infect_Prob.describe()


#visualization
fig, ax1 = plt.subplots()
sns.distplot(train['Infect_Prob'], ax=ax1, fit=stats.norm)
fig, ax2 = plt.subplots()
stats.probplot(train['Infect_Prob'], plot=plt)



y_train = train.Infect_Prob.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Infect_Prob'], axis=1, inplace=True)


#find missing data
all_data_na = (all_data.isnull().sum() / len(train)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)



f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


#Correlation map to see how features are correlated 
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)



#Finding the correlations in numeric features
corr = train.corr()   # or df_train[num_columns].corr()
top_corr_feat = corr['Infect_Prob'].sort_values(ascending=False)[:25]
print(top_corr_feat)



# Most correlated variables
threshold = 0.01
top_corr = corr.index[np.abs(corr["Infect_Prob"]) > threshold]

plt.figure(figsize=(10,8))
sns.heatmap(train[top_corr].corr(),annot=True,cmap="RdBu_r")



for col in top_corr_feat.index[:15]:
    print('{} - unique values: {} - mean: {:.2f}'.format(col, train[col].unique()[:5], np.mean(train[col])))



#d-dimer
all_data['d-dimer'] = all_data['d-dimer'].fillna(all_data['d-dimer'].mode()[0])

#most frequent
#all_data['d-dimer'].value_counts() / len(all_data) * 100

#Heart rate
all_data['Heart rate'] = all_data['Heart rate'].fillna(all_data['Heart rate'].mode()[0])

#insurance

all_data['Insurance'] = all_data['Insurance'].fillna(all_data['Insurance'].mode()[0])

#Platelets
all_data['Platelets'] = all_data['Platelets'].fillna(all_data['Platelets'].mode()[0])


# The most frequent value
all_data['Occupation'].value_counts() / len(all_data) * 100

all_data['Occupation'].fillna('Researcher', inplace=True)

#FT/month
all_data['FT/month'] = all_data['FT/month'].fillna(all_data['FT/month'].mode()[0])

#Diuresis
all_data['Diuresis'] = all_data['Diuresis'].fillna(all_data['Diuresis'].mode()[0])

all_data['Children'] = all_data['Children'].fillna(0)

all_data['comorbidity'].fillna('None', inplace=True)    
all_data['cardiological pressure'].fillna('None', inplace=True)      


all_data['HDL cholesterol'] = all_data['HDL cholesterol'].fillna(all_data['HDL cholesterol'].mode()[0])

all_data['HBB'] = all_data['HBB'].fillna(all_data['HBB'].mode()[0])


all_data['Mode_transport'] = all_data['Mode_transport'].fillna(all_data['Mode_transport'].mode()[0])


#droping columns
d = ['Mode_transport','Region','Name','Deaths/1M']
all_data = all_data.drop(d, axis = 1)

num_cols = all_data.select_dtypes(exclude='object').columns
print('{} Numeric columns \n{}'.format(len(num_cols), num_cols))

categ_cols = all_data.select_dtypes(include='object').columns
print('\n{} Categorical columns \n{}'.format(len(categ_cols), categ_cols))

col = ['comorbidity']
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
for c in col:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


all_data = pd.get_dummies(all_data)

    
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
all_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in all_data.columns.values]

#test and train
train = all_data[:ntrain]
test = all_data[ntrain:]



#from sklearn.model_selection import train_test_split
#Xtrain, Xtest, ytrain, ytest = train_test_split(train, y_train, shuffle=True, 
 #                                               test_size=0.2, random_state=5)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb



#piplined model 
regressor = RandomForestRegressor(n_estimators = 150, random_state = 15) 
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=0))


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=5))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =10)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.05, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=1000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
    
   
averaged_models = AveragingModels(models = (regressor, ENet, GBoost, KRR, lasso,model_lgb,model_xgb))
averaged_models.fit(train, y_train)

y_averaged_models = averaged_models.predict(test)

#randomForest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 170, random_state = 15)
regressor.fit(train, y_train)

y_regressor=regressor.predict(test)

#xgboot
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.05, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=5000,
                             reg_alpha=0.5, reg_lambda=0.8571,
                             subsample=0.5, silent=1,
                            random_state =5, nthread = -1) 

model_xgb.fit(train,y_train)
y_xgboost = model_xgb.predict(test)

#combining all models values to one
yfinal=y_averaged_models*.4+y_xgboost*.2+y_regressor*.4

df_ = pd.DataFrame({'original ':test_ID , 'predict': yfinal})
df_.to_csv('prediction1(20/3/2020).csv',index=False)



#########
#part_2#
########

X = train['Diuresis'] 
Y = di['D']
x = test['Diuresis']
x=x.to_frame()
X=X.to_frame()
Y=Y.to_frame()
#model for 27th train value
#randomForest
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor2.fit(X, Y)
y=regressor2.predict(x)
y = pd.DataFrame(y)

test.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
#exchange 20th and 27th value
test2 = test
test2['Diuresis']=y

#xgboost
y_xgboost2 = model_xgb.predict(test2)

#regressor
y_regressor2=regressor.predict(test2)
#piplined model
y_averaged_models2 = averaged_models.predict(test2)

   

yfinal2=y_averaged_models2+y_xgboost2*.2+y_regressor2*.4


df_ = pd.DataFrame({'original ':test_ID , 'predict': yfinal2})
df_.to_csv('prediction2(27/3/2020).csv',index=False)

