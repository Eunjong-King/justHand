import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv('files/righthand.csv')
data = np.round(data, decimals=5)
feature_list = list(data)[:-1]
data_input = data[feature_list].to_numpy()
data_target = data['C'].to_numpy()
train_input, test_input, train_target, test_target = tts(data_input, data_target, train_size=0.1)
kn = KNC(n_neighbors=3)
kn.fit(train_input, train_target)
print("모델 점수 : ", kn.score(test_input, test_target))
joblib.dump(kn, 'files/righthand_model.pkl')