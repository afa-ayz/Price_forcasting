import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.dates import date2num
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

fileflod_list = ['QLD1','VIC1','TAS1','SA1']
acf_list = [4,2,2,2]
pacf_list = [5,2,2,2]
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(48).mean()
    rol_weighted_mean = timeseries.ewm(span=48).mean()
    rolstd = timeseries.rolling(48).std()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    weighted_mean = plt.plot(rol_weighted_mean, color='green', label='weighted Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)
	
j=0
for i in fileflod_list:
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d %H:%M:%S')
	data = pd.read_csv('./'+i+'/'+i+'_2008.csv', parse_dates=['SETTLEMENTDATE'], index_col='SETTLEMENTDATE', date_parser=dateparse)
	print(data)
	ts = data['RRP']
	plt.plot(ts)
	plt.savefig('./'+i+'/'+i+'_original_series.jpeg')
	plt.show()

	test_stationarity(ts)
	plt.savefig('./'+i+'/'+i+'_rolling_mean.jpeg')
	plt.show()
	# estimating
	ts_log = np.log(ts)
	# plt.plot(ts_log)
	# plt.show()
	moving_avg = ts_log.rolling(48).mean()
	# plt.plot(moving_avg)
	# plt.plot(moving_avg,color='red')
	# plt.show()
	ts_log_moving_avg_diff = ts_log - moving_avg
	# print ts_log_moving_avg_diff.head(12)
	ts_log_moving_avg_diff.dropna(inplace=True)
	test_stationarity(ts_log_moving_avg_diff)
	plt.savefig('./'+i+'/'+i+'_weight_rolling_mean.jpeg')
	plt.show()

	# differencing
	ts_log_diff = ts_log.diff(1)
	ts_log_diff.dropna(inplace=True)
	test_stationarity(ts_log_diff)
	plt.savefig('./'+i+'/'+i+'_std.jpeg')
	plt.show()
	
	ts_log_diff1 = ts_log.diff(1)
	ts_log_diff2 = ts_log_diff1.diff(1)
	plt.plot(ts_log_diff1, label='diff 1')
	plt.plot(ts_log_diff2, label='diff 2')
	plt.legend(loc='best')
	plt.savefig('./'+i+'/'+i+'_test_stationarity.jpeg')
	plt.show()
	
	# decomposing
	decompfreq = 7*48
	fileObject = open('sampleList.txt','w')  
	for ip in list(ts_log):  
		fileObject.write(str(ip))
		fileObject.write('\n')  
	fileObject.close()
	where_are_nan = np.isnan(ts_log)
	ts_log[where_are_nan] = 1
	decomposition = seasonal_decompose(ts_log,freq=decompfreq)

	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid

	plt.subplot(411)
	plt.plot(ts_log,label='Original')
	plt.legend(loc='best')
	plt.subplot(412)
	plt.plot(trend,label='Trend')
	plt.legend(loc='best')
	plt.subplot(413)
	plt.plot(seasonal,label='Seasonarity')
	plt.legend(loc='best')
	plt.subplot(414)
	plt.plot(residual,label='Residual')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('./'+i+'/'+i+'_residuals.jpeg')
	plt.show()


	lag_acf = acf(ts_log_diff, nlags=20)
	lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
	plt.subplot(121)
	plt.plot(lag_acf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
	plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
	plt.title('Autocorrelation Function')
	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
	plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
	plt.title('Partial Autocorrelation Function')
	plt.tight_layout()
	plt.savefig('./'+i+'/'+i+'_acf_pacf.jpeg')
	plt.show()
	# AR model
	model = ARIMA(ts_log, order=(acf_list[j], 1, 0))
	result_AR = model.fit(disp=-1)
	plt.plot(ts_log_diff)
	print(len(result_AR.fittedvalues))
	print(len(ts_log_diff))
	plt.plot(result_AR.fittedvalues, color='red')
	plt.title('AR model RSS:%.4f' % sum(result_AR.fittedvalues - ts_log_diff) ** 2)
	plt.savefig('./'+i+'/'+i+'_AR_model.jpeg')
	plt.show()
	# MA model
	model = ARIMA(ts_log, order=(0, 1, pacf_list[j]))
	result_MA = model.fit(disp=-1)
	plt.plot(ts_log_diff)
	plt.plot(result_MA.fittedvalues, color='red')
	plt.title('MA model RSS:%.4f' % sum(result_MA.fittedvalues - ts_log_diff) ** 2)
	plt.savefig('./'+i+'/'+i+'_MA_model.jpeg')
	plt.show()
	# ARIMA model
	try:
		model = ARIMA(ts_log, order=(acf_list[j], 1, pacf_list[j]))
		result_ARIMA = model.fit(disp=-1)
		j+=1
		plt.plot(ts_log)
		plt.plot(result_ARIMA.fittedvalues, color='red')
		plt.savefig('./'+i+'/'+i+'_ARIMA_model.jpeg')
		plt.show()
		predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)

		predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

		predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
		predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

		predictions_ARIMA = np.exp(predictions_ARIMA_log)
		plt.plot(ts)
		plt.plot(predictions_ARIMA)
		plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
		plt.savefig('./'+i+'/'+i+'_predictions_Arima.jpeg')
		plt.show()
	except:
		print ("exception\n")
		try:
			model = ARIMA(ts_log_diff, order=(acf_list[j], 1, pacf_list[j]))
			result_ARIMA = model.fit(disp=-1)
			j+=1
			plt.plot(ts_log_diff)
			plt.plot(result_ARIMA.fittedvalues, color='red')
			plt.savefig('./'+i+'/'+i+'_ARIMA_model.jpeg')
			plt.show()
			predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)

			predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

			predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
			predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

			predictions_ARIMA = np.exp(predictions_ARIMA_log)
			plt.plot(ts)
			plt.plot(predictions_ARIMA)
			plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
			plt.savefig('./'+i+'/'+i+'_predictions_Arima.jpeg')
			plt.show()
		except: print ("exception\n")











