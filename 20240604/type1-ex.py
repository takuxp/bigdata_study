"""
자동차 데이터 셋에서 qsec 컬럼을 Min-Max Scale로 변환 후 0.5보다 큰 값을 가지는 레코드(row) 수는?
문제분석
    - 문제에서 qsec 컬럼만 묻고 있음 (다른 컬럼 신경 쓸 필요 없음)
    - MinMax Scale 변환
    - 조건 0.5보다 큰 값
"""
import pandas as pd

def load_data():
    return pd.read_csv("./20240604/mtcars.csv")

# data = load_data()
# print(data.head())
#스케일링
def sol1():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = load_data()
    data["qsec"] = scaler.fit_transform(data[["qsec"]]) #데이터 프레임 형식으로 대괄호 두개
    print(data)
    return data

def sol2():  #좀 더 편함
    from sklearn.preprocessing import minmax_scale
    data = load_data()
    data["qsec"] = minmax_scale(data["qsec"])
    print(data)
    return data

# sol2()

def my_minmix(data):
    data = (data - min(data)) / (max(data) - min(data))
    return data

def sol3():
    data = load_data()
    data['qsec'] = my_minmix(data['qsec'])
    print(data)
    return data


# data = sol2()

# cond = data['qsec'] > 0.5
# print("")
# print("")
# print("")
# print(len(data[cond]))

# 결측치가 있는 데이터 생성
import numpy as np
df = pd.DataFrame(
    {
        'a':[1,2,3,4,5,6,7,8,9],
        'b':[1.3,2.2,3.3,np.nan,5.8,6.9,np.nan,8.2,9.0]
    }
)
print(sum(df['b'] > 3))
