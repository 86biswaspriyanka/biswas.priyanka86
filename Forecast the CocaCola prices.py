#importing dataset
import pandas as pd
data = pd.read_excel("../p49/Delhi.xlsx")
data.columns

# Time plot
data.plot()

# creating dummy variable 
dummy= pd.get_dummies(data.Quarter)
d= data.join(dummy)

# creating time index column
import numpy as np
d['t'] = np.arange(1,43)

# creating t- square column
d['t_squre'] = d.t*d.t

# log value of Sales
d['log_sales']= np.log(d.Sales)

# Time plot
d.Sales.plot()

# Spliting Train and Test
Train = d.head(39)
Test = d.tail(4)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear # 538.44

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 441.36

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squre',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squre"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad #504.79

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea # 1605.86

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squre+Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad # 403.02

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #2067.93

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea # 1937.08

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far 
# selecting this model for Forecasting

#               MODEL  RMSE_Values
#0        rmse_linear   538.447094
#1           rmse_Exp   441.366734
#2          rmse_Quad   504.797433
#3       rmse_add_sea  1605.860430
#4  rmse_add_sea_quad   403.028232
#5      rmse_Mult_sea  2067.931124
#6  rmse_Mult_add_sea  1937.084005

