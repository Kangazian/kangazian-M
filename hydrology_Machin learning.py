import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import genextreme
from scipy.stats import gumbel_r
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
'**********************read Data from excel File*****************************'
file_path=r'C:\Users\USER\PycharmProjects\pythonProject3\reza.xlsx'
df=pd.read_excel(file_path)
df1=pd.read_excel(file_path)
datam=df.to_numpy()
newdata=datam[:,3:]
'*********************cal percentage of NAN***********************************'
miss_data=np.count_nonzero(np.isnan(newdata))
total_newdata=newdata.size
percentage_missing=(miss_data/total_newdata)*100
print(f'{percentage_missing}***percentage data*** as misiing data(NAN)')
plt.bar(['miss_data','total_data'],[miss_data,total_newdata],color=['green','blue'])
plt.xlabel('type of DATA')
plt.ylabel('amount_DATA')
plt.title('total DATA,miss_data(NAN)')
plt.show()
' *****************************************************************************'
df['DATE']=pd.to_datetime(df[['year','month','day']])
df1['DATE']=pd.to_datetime(df1[['year','month','day']])
df1.drop(['year','month','day'], axis=1, inplace=True)
df1.set_index('DATE',inplace=True)
df.drop(['year','month','day'], axis=1, inplace=True)
df.set_index('DATE',inplace=True)
#df.reset_index(inplace=True)
'*******************************daily_mean*****************************'
daily_sum=df.resample('D').sum()
daily_sum1=daily_sum.sum(axis=1)
#print(daily_sum1)
daily_mean=daily_sum.mean(axis=1)
daily_Max=daily_sum.max(axis=1)
daily_min=daily_sum.min(axis=1)
'*************************monthly.........................'
monthly_mean=daily_sum1.resample('M').mean()
monthly_SUM=daily_sum1.resample('M').sum()
monthly_Max=daily_sum1.resample('M').max()
monthly_MIN=daily_sum1.resample('M').min()
'***************************year mean****************************************'
yearly_mean=daily_sum1.resample('Y').mean()
yearly_Max=daily_sum1.resample('Y').max()
yearly_SUM=daily_sum1.resample('Y').sum()
yearly_MIN=daily_sum1.resample('Y').min()
#print(year_mean)
'**************************daily var************************************************'
daily_std=daily_sum.std(axis=1)
#print(daily_std)
'**************************monthly var************************************************'
monthly_std=daily_sum1.resample('M').std()
#print(monthly_std)
'*********************************year var************************************************'
year_std=daily_sum1.resample('y').std()
#print(year_std)
'**************************daily std************************************************'
daily_var=daily_sum.var(axis=1)
#print(daily_var)
'**************************monthly var************************************************'
monthly_var=daily_sum1.resample('M').var()
#print(monthly_var)
'*********************************year var************************************************'
yearly_var=daily_sum1.resample('y').var()
#print(yearly_var)
'*******************skewness  per day**********************************'
daily_skewness=daily_sum.skew(axis=1)
#print(daily_skewness)
'***********************************monthly skewness********************'
monthly_skewnes=daily_sum1.resample('M').apply(lambda x: x.skew())
#print(monthly_skewnes)
yearly_skewnes=daily_sum1.resample('y').apply(lambda x: x.skew())
#print(yearly_skewnes)
'******************************DRY Percentage**************************************'
daily_dry=(1-(daily_mean)/daily_Max)*100
monthly_dry=(1-(monthly_mean)/monthly_Max)*100
#print(monthly_dry)
yearly_dry=(1-(yearly_mean)/yearly_Max)*100
'******************************annual maxima sereis***********************************************'
annual_Maxiam_per_hours=daily_Max.resample('Y').max()
#print(annual_Maxiam_per_hours)
monthly_Maxiam_per_hours=daily_Max.resample('M').max()
#print(monthly_Maxiam_per_hours)
'*********************************************************************************************'
Aggregate_daily=daily_sum1.resample('D').sum().cumsum()
Aggregate_15_daily=daily_sum1.resample('15D').sum().cumsum()
Aggregate_weekly=daily_sum1.resample('W').sum().cumsum()
'**************************************************************************************'
df_value=df1.values
change_rows=df_value.reshape(df_value.size,1)
hours_values=change_rows.reshape(-1)
data_range=pd.date_range(start='1969-01-01',end='2004-01-01',freq='H',inclusive='right')
new_DF=pd.DataFrame({'Date':data_range,'Hour':hours_values})
new_DF.set_index('Date',inplace=True)
'************************aggregate 1H,3H,6H,8H,9H,12H*****************************'
aggregate_hourly_Df=new_DF.resample('H').sum().cumsum()
aggregate_3Hhourly_Df=new_DF.resample('3H').sum().cumsum()
aggregate_6Hhourly_Df=new_DF.resample('6H').sum().cumsum()
aggregate_8Hhourly_Df=new_DF.resample('8H').sum().cumsum()
aggregate_9Hhourly_Df=new_DF.resample('9H').sum().cumsum()
aggregate_12Hhourly_Df=new_DF.resample('12H').sum().cumsum()
'****************************mean() of 1H,3H,6H,8H,9H,12H*********************************************'
mean_hourly_Df=new_DF.resample('y').max()
mean_3Hhourly_Df=new_DF.resample('3H').mean()
mean_6Hhourly_Df=new_DF.resample('6H').mean()
mean_8Hhourly_Df=new_DF.resample('8H').mean()
mean_9Hhourly_Df=new_DF.resample('9H').mean()
mean_12Hhourly_Df=new_DF.resample('12H').mean()
Max_hourly_Df=new_DF.resample('H').max()
gh=Max_hourly_Df.values.reshape(-1)
times = list(new_DF.index)
values = list(new_DF.values)
plt.figure(figsize=(10, 6))
plt.plot(times, values, label='precipitation')
plt.xlabel('time')
plt.ylabel('precipitation')
plt.title('Time Series')
plt.legend()
plt.show()

hourly_precipitation_data =annual_Maxiam_per_hours
#  GEV
params =genextreme.fit(hourly_precipitation_data,method='MLE')
x_scale = np.linspace(min(annual_Maxiam_per_hours), max(annual_Maxiam_per_hours), 100)
# محاسبه CDF و PDF با توزیع GEV
cdf = stats.genextreme.cdf(x_scale,*params)
pdf = stats.genextreme.pdf(x_scale,*params)
#  CDF
plt.figure(figsize=(10, 5))
plt.plot(x_scale, cdf, label='CDF')
plt.xlabel('values')
plt.ylabel('CDF')
plt.legend()
plt.grid()
#  PDF
plt.figure(figsize=(10, 5))
plt.plot(x_scale, pdf, label='PDF')
plt.xlabel('values(H)')
plt.ylabel('PDF')
plt.legend()
plt.grid()
plt.show()
 #produce IDF parameters
#return_periods = [2, 5, 10, 20, 50, 100,200,500,1000] # مثال: بازه‌های بازگشت مورد نظر
return_periods=np.geomspace(2,1000,len(annual_Maxiam_per_hours),dtype=int)
print(return_periods)
print(return_periods.shape,return_periods.ndim)
idf_values=intensity = genextreme.ppf(1 - 1 / np.array(return_periods), *params)
plt.figure(figsize=(8, 6))
plt.plot(return_periods, idf_values, marker='o', linestyle='-', color='b')
plt.xlabel('Return Periods')
plt.ylabel('Rainfall intensity (mm/H)')
plt.title('Return period, T (years)')
plt.grid(True)
plt.show()
'********************************find iDF prameters***********************************'
y=intensity_depend_V=np.log(intensity)
print(intensity_depend_V)
print(intensity_depend_V.ndim,intensity_depend_V.shape)
x1=return_periods_independ_V=np.log(return_periods)
x2=timescale_independ_V=np.log(np.arange(1,len(annual_Maxiam_per_hours)+1,1))
regression_model = LinearRegression()
X = np.column_stack((x1,x2))
regression_model.fit(X,y)
x1_coef, x2_coef = regression_model.coef_
intercept_C = regression_model.intercept_
print(intercept_C,x1_coef,x2_coef)
Alfa=np.exp(intercept_C)
Beta=x1_coef
Gama=np.abs((x2_coef))
print(Alfa,Beta,Gama)
'*******************************mean Squared_error********************************'
predicted_C = regression_model.predict(X)
# محاسبه MSE (Mean Squared Error)
mse = mean_squared_error(y, predicted_C)
# محاسبه R-squared (Coefficient of Determination)
r2 = r2_score(y, predicted_C)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)
'********************************************************idf***********************************************'
time=np.arange(1,len(annual_Maxiam_per_hours)+1,1)
T2=np.array([2])
i_TP2=((Alfa*(T2**Beta))/(time**Gama))
T10=np.array([10])
i_TP10=((Alfa*(T10**Beta))/(time**Gama))
T5=np.array([5])
i_TP5=((Alfa*(T5**Beta))/(time**Gama))
plt.plot(time, i_TP2 , label='2 years')
plt.plot(time,i_TP5 , label='5 years')
plt.plot(time, i_TP10 , label='10 years')
plt.xlabel('timescale')
plt.ylabel('Rainfall intensity (mm/H),i')
plt.title(' Intensity-Duration-Frequency (IDF) ')
plt.legend()
plt.grid(True)
plt.show()
'***********************************Msc***************************************'
tc=2.8 #time
p6T=60 #mm
CN=87
A=8 #km^2
t1=np.arange(0,12.5,0.5)
percentage_p_SCS=np.array([0,0.02,0.08,0.15,0.22,0.6,0.7,0.78,0.84,0.88,0.92,0.96,1])
P=cumulate_p=np.round((percentage_p_SCS)*p6T)
S=((25400/CN)-254)
R=np.round((((P-0.2*S)**2)/(P+0.8*S)),decimals=1)
R1=np.diff(R,axis=0)
m=np.insert(R1,0,0,axis=0)
e=np.maximum(m,0)
tp=0.7*tc
qp=np.round((2.08*e/tp))
ratio_T_Tp=np.round(t1/tp,2)
ratio_q_qp_scs=np.array([0,0.12,0.43,0.83,1,0.88,0.66,0.45,0.32,0.22,0.15,0.11,0.08,0.05,0.04,0.03,0.02,0.01,0.01,0.01,0,0,0,0,0])
q_dt=np.outer(qp[4:],ratio_q_qp_scs)
z=np.zeros([35,35])
for i in range(9):
    z[i, 4+i:24+i] = q_dt[i, 1:21]
q_Total_dt=np.sum(z,axis=0) [0:25]
q_Total=q_Total_dt.cumsum()
plt.plot(ratio_T_Tp,ratio_q_qp_scs,label='HYDROGRAPH')
plt.xlabel('dimenslession,T/Tp')
plt.ylabel('dimensionless,q/qp')
plt.title('synthetic hydrograph')
plt.legend()
plt.grid(True)
plt.show()
plt.plot(t1,q_Total,label='q')
plt.xlabel('timescale(H)')
plt.ylabel('q(L.sec^-1.KM^-1)')
plt.title(' Cumulative hydrograph curve (runoff) ')
plt.legend()
plt.grid(True)
plt.show()