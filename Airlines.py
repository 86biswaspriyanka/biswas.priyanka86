#importing dataset
import pandas as pd
data = pd.read_excel("../18.Forecasting/Airlines+Data.xlsx")
data.columns

# Converting the normal index of data to time stamp for getting year on x axis while running time plot
data.index=pd.to_datetime(data.Month,format="%b-%y")
# time series plot - included Year in X-axis 
data.Passengers.plot() 

# Converting the index to normal
import numpy as np
data.index = np.arange(0,96)

# Creating a Date column to store the actual Date format for the given Month column
from datetime import datetime,time
data["Date"] = pd.to_datetime(data.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

data["month"] = data.Date.dt.strftime("%b") # month extraction
#data["Day"] = data.Date.dt.strftime("%d") # Day extraction
#data["wkday"] = data.Date.dt.strftime("%A") # weekday extraction
data["year"] = data.Date.dt.strftime("%Y") # year extraction

# Creating a new column in data joining the month with year
data['Month'] = data.Date.dt.strftime("%Y_%b")
del data['Date']

## EDA on Time series data ##
# Heat map visualization 
import seaborn as sns
heatmap_y_month = pd.pivot_table(data=data,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=data)
sns.boxplot(x="year",y="Passengers",data=data)
sns.factorplot("month","Passengers",data=data,kind="box")

# Line plot for Passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=data)


# moving average for the time series to understand better about the trend character in data
import matplotlib.pylab as plt
data.Passengers.plot(label="org") # time series plot
for i in range(2,24,6):
    data["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=4)
    
# Time series decomposition plot 
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(data.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(data.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
tsa_plots.plot_acf(data.Passengers,lags=10)
tsa_plots.plot_pacf(data.Passengers)

# creating dummy variable 
dummy= pd.get_dummies(data.Month)
d= data.join(dummy)

# creating time index column
import numpy as np
d['t'] = np.arange(1,97)

# creating t- square column
d['t_squre'] = d.t*d.t

# log value of Sales
d['log_Passengers']= np.log(d.Passengers)

# Time plot
d.Passengers.plot() # Assumptions based on time plot ; it have multiplicative seasonality with linear upward trend

# Spliting Train and Test
Train = d.head(84)
Test = d.tail(12)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear # 53.199

##################### Exponential ##############################
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 46.05

#################### Quadratic ###############################
Quad = smf.ols('Passengers~t+t_squre',data = Train).fit()
pred_Quad = Quad.predict(Test)
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad)))**2)
rmse_Quad # 15.97
################### Additive seasonality ########################
X = Train.iloc[:,2:96] 
Y = Train['Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:98]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_add_sea = pred_passengers_test # just for renaming purpose

rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea # 138.69

################## Additive Seasonality Quadratic ############################
X = Train.iloc[:,2:100]
Y = Train['Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:100]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_add_sea_quad = pred_passengers_test # just for renaming purpose

rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad ))**2))
rmse_add_sea_quad #48.05

################## Multiplicative Seasonality ##################
X = Train.iloc[:,2:98]
Y = Train['log_Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:98]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_sea = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_sea ))**2))
rmse_Mult_sea # 146.6

#################Multiplicative Additive Seasonality ###########
X = Train.iloc[:,2:99]
Y = Train['log_Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:99]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_add_sea = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_add_sea))**2))
rmse_Mult_add_sea # 46.05

################## Multiplicative Quadratic trend ################
X = Train.iloc[:,2:100]
Y = Train['log_Passengers']
 # with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:100]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_sea_quad = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_sea_quad))**2))
rmse_Mult_sea_quad # 49.33

################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","rmse_Mult_sea_quad"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_sea_quad])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_Exp  has the least value among the models prepared so far 
# selecting Qudratic model for Forecasting.

