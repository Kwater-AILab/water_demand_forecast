import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#import plotly.io
from matplotlib import rc
rc('font', family='HCR Dotum')
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

###################################
# System Config
###################################

# default train size rate
trainSize_rate = 0.8  # 학습 및 예측 셋 구분

current_time = datetime.today().strftime("%Y%m%d_%H%M%S")


model_list = ["GBM", "RF"]   # 분석 모델 리스트 설정 : LSTM, GBM, RF
performance_list = ["RMSE", "R2", "MSE"]    # 분석 성능평가 리스트 설정 : RMSE, R2, MSE, MAE



###################################

# RandomForest Regression Algorithm
def AL_RandomForest(trainX, trainY, testX, testY):
    rf_clf = RandomForestRegressor(n_estimators=500)
    rf_clf.fit(trainX, np.ravel(trainY, order="C"))
    #rf_clf.fit(trainX, trainY)

    # relation_square = rf_clf.score(trainX, trainY)
    # print('RandomForest 학습 결정계수 : ', relation_square)

    y_pred1 = rf_clf.predict(trainX)
    y_pred2 = rf_clf.predict(testX)

    return y_pred2

def AL_RF_Class(trainX, trainY, testX, testY):
    rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf_clf.fit(trainX, np.ravel(trainY, order="C"))

    # relation_square = rf_clf.score(trainX, trainY)
    # print('RandomForest 학습 결정계수 : ', relation_square)

    y_pred1 = rf_clf.predict(trainX)
    y_pred2 = rf_clf.predict(testX)

    return y_pred2

def AL_GradientBoosting(trainX, trainY, testX, testY):

    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.05)
    gbr_model.fit(trainX, np.ravel(trainY, order="C"))

    y_pred = gbr_model.predict(trainX)
    y_pred2 = gbr_model.predict(testX)

    return y_pred2

def Performance_index(obs, pre, mod_str):
    if mod_str == 'R2':
        pf_index = r2_score(obs, pre)
    elif mod_str == 'RMSE':
        s1 =  mean_squared_error(obs, pre)
        pf_index = np.sqrt(s1)
    elif mod_str == 'MSE':
        pf_index = mean_squared_error(obs, pre)
    elif mod_str == 'MAE':
        pf_index = mean_absolute_error(obs, pre)

    return pf_index

def basic_chart(obsY, preY, str_part):
    if str_part == 'line':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(obsY)), obsY, '-', label="Original Y")
        ax.plot(range(len(preY)), preY, '-x', label="predict Y")
    plt.legend(loc='upper right')


# 입출력 자료 통합차트
def total_chart(df1, list1):
    columns = list(range(0, len(list1)))
    i = 1
    values = df1.values
    plt.figure(figsize=(9, 40))
    for variable in columns:
        plt.subplot(len(columns), 1, i)
        plt.plot(values[:, variable])
        plt.title(df1.columns[variable], y=0.5, loc='right')
        i += 1
    plt.show()

def makeDir(model):
    path_result_data = "./result_data/" + current_time + "/" + model + "/"
    if not os.path.isdir(path_result_data):
        os.makedirs(path_result_data)
    path_result_graph = "./result_graph/" + current_time + "/" + model + "/"
    if not os.path.isdir(path_result_graph):
        os.makedirs(path_result_graph)
    path_result_model = "./result_model/" + current_time + "/" + model + "/"
    if not os.path.isdir(path_result_model):
        os.makedirs(path_result_model)

    path_list = pd.DataFrame([(path_result_data, path_result_graph, path_result_model)], columns=['data','graph','model'])
    return path_list

def buildDataSet3(timeSeries):

    xdata = pd.DataFrame(timeSeries.iloc[:, 1:-1])
    ydata = pd.DataFrame(timeSeries.iloc[:, -1])
    return xdata, ydata

# 데이터 결측 범위 지정 결측값 도출
def set_outlier(df1):
    columns = list(range(1, df1.shape[1]))
    for i in columns:
        df1[df1.columns[i]].mask(
            (df1[df1.columns[i]] >= 30000000) | (df1[df1.columns[i]] < 5000000), inplace=True)
    return df1

# 선택열 형변환
def coerce_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')

# data test
###########

#col_dataset = ['date','평균기온','강수량','습도','일조량','지중온도','풍량','기압','sup']
#col_dataset = ['date','세대수','남자','여자','고령화','전력사용량','산업체 수','산업단지 면적','평균기온','강수량','습도','일조량','지중온도','풍량','기압','sup']
#col_dataset = ['date','세대수','남자','여자','고령화','전력사용량','산업체 수','산업단지 면적','가계소득','취업률','평균기온','강수량','습도','일조량','지중온도','풍량','기압','sup']
col_dataset = ['date','year','month','전력사용량','산업체 수','산업체 규모','평균기온','강수량','습도','일조량','지중온도','풍량','기압','sup']



raw_data = pd.read_csv('./종속변수.csv', sep=',', encoding='CP949')
raw_input = pd.read_csv('./대전시.csv', sep=',', encoding='CP949', thousands=',')
raw_input.columns = ['연도','월','총인구수','세대수','세대당 인구','남자','여자','남녀 비율','고령화','전력사용량',
                     '산업체 수','산업체 규모','산업단지 면적','가계소득','학력수준','인건비','인건비 평균','취업률','취약계층',
                     '평균기온','강수량','습도','일조량','지중온도','풍량','기압']


raw_temp = raw_data[['지자체명','대전광역시']]
raw_temp = pd.DataFrame(raw_temp)
raw_temp.columns = ['date', 'sup']
raw_temp = raw_temp.drop(index=0, axis=0)
raw_temp['sup'] = raw_temp['sup'].str.replace(',', '').astype('int64')

raw_temp = set_outlier(raw_temp)
raw_temp = raw_temp.dropna(how='any')

raw_input['date'] = raw_input['연도'].apply(str) + "-" + raw_input['월'].apply(str).str.zfill(2)
raw_input['year'] = pd.DatetimeIndex(raw_input['date']).year
raw_input['month'] = pd.DatetimeIndex(raw_input['date']).month

raw_set = pd.merge(left=raw_input, right=raw_temp, how='left' , on= 'date')
raw_set = pd.DataFrame(raw_set, columns=col_dataset)
raw_set = raw_set.dropna(how='any').reset_index(drop=True)


raw_set.info()
print(raw_set.head())


##################################################
# 시계열 데이터 분해(Time series data decomposition)
##################################################

# from statsmodels.tsa.seasonal import seasonal_decompose
#
# ts = raw_set['평균기온']
# result =seasonal_decompose(ts, model='additive', period=12)
#
# #plt.rcParams['figure.figsize'] = [12,8]
# result.plot()
# plt.show()
#
# exit()



##################################################
# 시계열 데이터 예측 : Prophet 알고리즘
##################################################


#raw_set = raw_set.apply(pd.to_numeric)


# from fbprophet import Prophet
#
#
# df = raw_set[['date', 'sup']]
# df.columns = ['ds', 'y']
# df['ds'] = pd.to_datetime(df['ds'])
#
# df.info()
# print(df.head())
#
# # model = Prophet()
# # model.fit(df)
#
# last_1year = list()
# for i in range(1,13):
#     last_1year.append(['2021-%02d' % i])
#
#
# last_1year = pd.DataFrame(last_1year, columns= ['ds'])
# last_1year['ds'] = pd.to_datetime(last_1year['ds'])
#
#
#
# train = df.drop(df.index[-12:])
# y_true = df['y'][-12:].values
#
# model = Prophet()
# model.fit(train)
#
# forecast = model.predict(last_1year)
# y_pred = forecast['yhat'].values
#
# #mae = mean_absolute_error(y_true, y_pred)
# mae = r2_score(y_true, y_pred)
# print('R2 : %.3f' % mae)
#
# plt.plot(y_true, label = 'Actual')
# plt.plot(y_pred, label = 'Predicted')
# plt.legend()
# plt.show()
#
#
# exit()



plt.subplot(211)
plt.plot(raw_set['date'], raw_set['sup'])
plt.subplot(212)
plt.plot(raw_set['date'], raw_set.iloc[:,4])
plt.show()




#######
# main
#######

temp_list = [raw_set]
temp_list_name = ["daejeon"]




###################################
# 각 지점마다 모델 생성 및 학습
###################################
for md in model_list:
    print("=" * 30)
    print("model : ", md)
    print("=" * 30)
    count = 0
    for df in temp_list:
        """
        Modulation 2 : 학습데이터 정제
        """
        print("Modulation 2 : 학습데이터 정제")
        # 날짜 데이터를 제외한 나머지 값
        xy = df

        #dataDim = len(xy.columns)  # 매개변수의 개수

        # train셋과 test셋을 분류(0.8 비율)
        trainSize = int(len(xy) * trainSize_rate)
        trainSet = xy[0:trainSize]
        testSet = xy[trainSize:]
        #trainSet = xy[len(xy)-trainSize:]
        #testSet = xy[0:len(xy)-trainSize]

        trainX, testX, trainY, testY = train_test_split(xy.iloc[:, 1:-1], xy.iloc[:, -1], test_size=0.2,
                                                        random_state=42)

        #trains = pd.DataFrame([trainX. trainY])
        #trains.to_csv('./trainset.csv', index=True)
        #trainX, trainY = buildDataSet3(trainSet)
        #testX, testY = buildDataSet3(testSet)

        """
        Modulation 3 : 모델 학습
        """
        print("Modulation 3 : 모델 학습")

        if md == "GBM":
            # 각 지점의 알맞는 학습 배치 데이터 생성
            #trainX, trainY = buildDataSet3(trainSet)
            #testX, testY = buildDataSet3(testSet)

            print(testX.head())
            print(testY.head())

            predict = AL_GradientBoosting(trainX, trainY, testX, testY)

        elif md == "RF":
            # 각 지점의 알맞는 학습 배치 데이터 생성
            #trainX, trainY = buildDataSet3(trainSet)
            #testX, testY = buildDataSet3(testSet)



            predict = AL_RandomForest(trainX, trainY, testX, testY)

        yhat = pd.DataFrame(predict).iloc[:,-1]
        actual = pd.DataFrame(testY).iloc[:,-1]

        #yhat = predict
        #actual = testY

        # 성과지표 표출 부분 : 적용 항목은 confing > performance_list[] 참조
        for pi in performance_list:
            rmse = Performance_index(actual, yhat, pi)
            print(temp_list_name[count] + " " + md + ' 예측 ' + pi + ' : ', rmse)


        """
        Modulation 4 : 결과 데이터 저장
        """
        print("Modulation 4 : 결과 데이터 저장")

        dir_list = makeDir(md)
        # # 복원된 데이터 저장
        # pd_actual_save = pd.DataFrame(actual)
        # pd_actual_save.to_csv(dir_list['data'][0] + temp_list_name[count] + "_actual", mode='w')
        #
        # pd_predict_save = pd.DataFrame(yhat)
        # pd_predict_save.to_csv(dir_list['data'][0] + temp_list_name[count], mode='w')

        # 그래프 저장
        basic_chart(actual, yhat, 'line')
        plt.title(md + " : " + temp_list_name[0])
        plt.savefig(dir_list['graph'][0] + temp_list_name[0] + '.png')

        count += 1

plt.show()