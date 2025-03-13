import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import itertools


def plot_data(df,title,save_path):
    fig = px.line(df,x='timestamp', y='avg_value', title=title)
    if '.png' in save_path:
        fig.write_image(save_path, width=3500, height=1000)
    elif '.html' in save_path:
        fig.write_html(save_path)

def load_data_v2(file_path):
    csv_header = ['instance_id','timestamp','metric_name','device','total_value','max_value','min_value','total_count','square_sum','avg_value']
    raw_df = pd.read_csv(file_path,header=None,names=csv_header)
    _instance_id = 'host_10.196.35.233'
    _metric_name = 'host.memory.use.rate'
    selected_df = raw_df[((raw_df['instance_id'] == _instance_id) & (raw_df['metric_name'] == _metric_name))]
    # selected_df['timestamp'] = pd.to_datetime(selected_df['timestamp'])
    sorted_df = selected_df.sort_values(by='timestamp',ascending=True)
    print(sorted_df)
    df = sorted_df[['timestamp','total_value']]
    print(df.head())
    plot_data(df,'raw_data','./pictures/raw_data_v2.html')

def load_data(file_path):
    _instance_id = 'esop_host_10.196.35.233'
    raw_df = pd.read_csv(file_path)
    selected_df = raw_df[raw_df['instance_id'] == _instance_id]
    sorted_df = selected_df.sort_values(by='timestamp',ascending=True)
    df = sorted_df[['timestamp','avg_value']]
    print(df.head())
    plot_data(df,'raw_data','./pictures/raw_data_v1.html')
    return df
    # return df[:int((len(df)*0.7))]

def timing_decomposition(NGE):
    print(NGE)
    decomposition = sm.tsa.STL(NGE, seasonal=24).fit()  # `seasonal` 参数设置为合适的窗口大小
    decomposition.plot()
    plt.savefig('./pictures/timing_decomposition.jpg')

def prediction_analysis(data,model,start,dynamic=False):
    pred=model.get_prediction(start=start,dynamic=dynamic,full_results=True)
    pci=pred.conf_int()#置信区间
    pm=pred.predicted_mean#预测值
    truth=data[start:]#真实值
    pc=pd.concat([truth,pm,pci],axis=1)#按列拼接
    pc.columns=['true','pred','up','low']#定义列索引
    print("1、MSE:{}".format(mse(truth,pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth,pm))))
    print("3、MAE:{}".format(mae(truth,pm)))
    return pc

def prediction_plot(pc):
    plt.figure(figsize=(10,8))
    plt.fill_between(pc.index,pc['up'],pc['low'],color='grey',alpha=0.15,label='confidence interval')#画出置信区间
    print(pc['true'])
    print(pc['pred'])
    plt.plot(pc['true'],label='base data')
    plt.plot(pc['pred'],label='prediction curve')
    plt.ylim(-100, 100)
    plt.legend()
    plt.savefig('./pictures/prediction.jpg')
    return True

#搜索法定阶
def SARIMA_search(data):
    p=q=range(0,3)
    s=[12]#周期为12
    d=[1]#做了一次季节性差分
    PDQs=list(itertools.product(p,d,q,s))#itertools.product()得到的是可迭代对象的笛卡儿积
    pdq=list(itertools.product(p,d,q))#list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
    params=[]
    seasonal_params=[]
    results=[]
    grid=pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            #建立模型
            mod= sm.tsa.SARIMAX(data,order=param,seasonal_order=seasonal_param,\
                            enforce_stationarity=False, enforce_invertibility=False)
            #实现数据在模型中训练
            result=mod.fit()
            print("ARIMA{}x{}-AIC:{}".format(param,seasonal_param,result.aic))
            #format表示python格式化输出，使用{}代替%
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid["pdq"]=params
    grid["PDQs"]=seasonal_params
    grid["aic"]=results
    print(grid[grid["aic"]==grid["aic"].min()])

if __name__ == '__main__':
    # file_path = '/code/SARIMA/data/esop_host_1213.csv'
    file_path = '/code/SARIMA/data/host.memory.use.rate.csv'
    df = load_data(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    print(df)
    NGE = df["avg_value"]
    # SARIMA_search(NGE) # 网格搜索
    # timing_decomposition(df)
    model = sm.tsa.SARIMAX(NGE,order=(1,1,1),seasonal_order=(1,1,1,12))
    SARIMA_m=model.fit()
    print(SARIMA_m.summary())
    pred = prediction_analysis(NGE,SARIMA_m,'2024-10-06 08:08:00',dynamic=True)
    prediction_plot(pred)

    # Forecast
    print('====Forecast====')
    start_time = pd.Timestamp("2024-10-06 08:08:00")
    end_time = pd.Timestamp("2024-11-05 10:05:00")
    forecast=SARIMA_m.get_forecast(steps=7200)
    forecast_index = pd.date_range(start=end_time, periods=7200, freq="T")  # 按分钟预测
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_df = pd.DataFrame({
    "timestamp": forecast_index,
    "predicted_mean": predicted_mean,
    "lower_bound": conf_int.iloc[:, 0],
    "upper_bound": conf_int.iloc[:, 1]
    })
    print(forecast_df)
    print('========')

    fig,ax=plt.subplots(figsize=(20,16))

    # RAW Data
    NGE.plot(ax=ax,label="base data")
    # Predicted Data & Region
    plt.plot(forecast_df["timestamp"],forecast_df["predicted_mean"], label="Predicted Mean", color="blue")
    plt.fill_between(
        forecast_df["timestamp"], forecast_df["lower_bound"], forecast_df["upper_bound"],
    color="gray", alpha=0.3, label="Confidence Interval"
    )
    # forecast.predicted_mean.plot(ax=ax,label="forecast data")
    ax.legend(loc="best",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-100,100)
    end_time = start_time + pd.DateOffset(months=2)
    plt.xlim(start_time,end_time)
    plt.savefig('./pictures/forecast.jpg')