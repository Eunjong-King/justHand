import pandas as pd
import numpy as np
import joblib
import random
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv('files/righthand.csv')
value_data = pd.DataFrame(pd.value_counts(data[list(data)[-1]].values, sort=False))
my_min = value_data[0].min()
my_max = value_data[0].max()
drop_list = []
for i in range(0, 35):
    temp_list = data.index[data['C'] == i].tolist()
    my_abs = len(temp_list) - my_min
    if my_abs <= 0:
        continue
    a = random.sample(range(0, len(temp_list)), my_abs)  # temp_list의 인덱스
    for j in a:
        drop_list.append(temp_list[j])
print("label c의 최소 개수는 ", my_min, " / 최대 개수는 ", my_max)
data = data.drop(drop_list)
data.to_csv("files/righthand.csv", mode='w', index=False)
print("데이터 전처리 완료")

data = np.round(data, decimals=5)
feature_list = list(data)[:-1]
data_input = data[feature_list].to_numpy()
data_target = data['C'].to_numpy()
train_input, test_input, train_target, test_target = tts(data_input, data_target, train_size=0.1)
kn = KNC(n_neighbors=3)
kn.fit(train_input, train_target)
print("모델 점수 : ", kn.score(test_input, test_target))
joblib.dump(kn, 'files/righthand_model.pkl')