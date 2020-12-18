import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error,mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv('data/TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])

df.sort_values(by=['saledate'],inplace=True,ascending=True)

df_tmp=df.copy()

df_tmp['saleYear']=df_tmp.saledate.dt.year
df_tmp['saleMonth']=df_tmp.saledate.dt.month
df_tmp['saleDay']=df_tmp.saledate.dt.day
df_tmp['saleDayOfWeek']=df_tmp.saledate.dt.dayofweek
df_tmp['saleDayOfYear']=df_tmp.saledate.dt.dayofyear

df_tmp.drop('saledate',axis=True,inplace=True)

for label,content in df_tmp.items():
	if(pd.api.types.is_string_dtype(content)):
		df_tmp[label]=content.astype('category').cat.as_ordered()

for label,content in df_tmp.items():
	if(pd.api.types.is_numeric_dtype(content)):
		if(pd.isnull(content)).sum():
			df_tmp[label+'_is_missing']=pd.isnull(content)

			df_tmp[label]=content.fillna(content.median())

for label,content in df_tmp.items():
	if not(pd.api.types.is_numeric_dtype(content)):
		
		df_tmp[label+'_is_missing']=pd.isnull(content)

		df_tmp[label]=pd.Categorical(content).codes+1

df_val=df_tmp[df_tmp.saleYear==2012]
df_train=df_tmp[df_tmp.saleYear!=2012]

x_train=df_train.drop('SalePrice',axis=1)
y_train=df_train['SalePrice']

x_valid,y_valid=df_val.drop('SalePrice',axis=1),df_val['SalePrice']


def rmsle(y_test,y_preds):
	return(np.sqrt(mean_squared_log_error(y_test,y_preds)))

def show_scores(model):
	train_preds=model.predict(x_train)
	val_preds=model.predict(x_valid)
	scores={'Training MAE':mean_absolute_error(y_train,train_preds),
			'valid MAE':mean_absolute_error(y_valid,val_preds),
			'Training rmsle':rmsle(y_train,train_preds),
			'valid rmsle':rmsle(y_valid,val_preds),
			'Training r^2':model.score(x_train,y_train),
			'valid r^2':model.score(x_valid,y_valid)}
	return scores



rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}


# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
#                                                     random_state=42),
#                               param_distributions=rf_grid,
#                               n_iter=10,
#                               cv=5,
#                               verbose=True)

# rs_model.fit(x_train, y_train)
# rs_model.fit(x_train,y_train)

# scores=show_scores(rs_model)

# print(scores)

ideal_model=RandomForestRegressor(n_estimators=40,min_samples_split=14,min_samples_leaf=1,max_features=0.5,n_jobs=-1,max_samples=None,random_state=42)
ideal_model.fit(x_train,y_train)

df_test=pd.read_csv('data/Test.csv',low_memory=False,parse_dates=['saledate'])

def preprocess_data(df):
	df['saleYear']=df.saledate.dt.year
	df['saleMonth']=df.saledate.dt.month
	df['saleDay']=df.saledate.dt.day
	df['saleDayOfWeek']=df.saledate.dt.dayofweek
	df['saleDayOfYear']=df.saledate.dt.dayofyear

	df.drop('saledate',axis=1,inplace=True)

	for label,content in df.items():
		if(pd.api.types.is_string_dtype(content)):
			df[label]=content.astype('category').cat.as_ordered()

	for label,content in df.items():
		if(pd.api.types.is_numeric_dtype(content)):
			if(pd.isnull(content)).sum():
				df[label+'_is_missing']=pd.isnull(content)

				df[label]=content.fillna(content.median())
	
	for label,content in df.items():
		if not(pd.api.types.is_numeric_dtype(content)):
		
			df[label+'_is_missing']=pd.isnull(content)

			df[label]=pd.Categorical(content).codes+1
		
	return df

df_test_mod=preprocess_data(df_test)
df_test['auctioneerID_is_missing']=False

test_preds=ideal_model.predict(df_test)

df_preds=pd.DataFrame()
df_preds['SalesID']=df_test['SalesID']
df_preds['SalesPrice']=test_preds

df_preds.to_csv('predicted.csv',index=False)










