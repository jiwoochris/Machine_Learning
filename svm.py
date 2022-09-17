import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate


df = pd.read_csv('breast-cancer-wisconsin.data', sep=',', header=None)
df = df.replace("?", np.nan)
df = df.dropna()

X, Y = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]], df[10]

X_train, X_test, y_train, y_test  = model_selection.train_test_split(X, Y, random_state=42)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)



# scores = cross_val_score(clf, X, Y, cv = 5)

# print(scores)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# SVM 결과로 시각화(PCA 2차원 축소 후 결과 확인)
pca = PCA(n_components=2)

# test 데이터셋 기준 시각화 진행
X_test_pca = pca.fit_transform(X_test)
y_find = y_test.reset_index(drop = True)

# target 마다 index 가져오기(꽃 종류마다 색깔을 다르게 시각화 목적) : 실제 라벨 기준
index_2 = y_find[y_find == 2].index
index_4 = y_find[y_find == 4].index

# target 마다 index 가져오기(꽃 종류마다 색깔을 다르게 시각화 목적) : 예측 라벨 기준
y_pred_Series = pd.Series(pred)
index_2_p = y_pred_Series[y_pred_Series == 2].index
index_4_p = y_pred_Series[y_pred_Series == 4].index

# 시각화
plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.scatter(X_test_pca[index_2, 0], X_test_pca[index_2, 1], color = 'purple', alpha = 0.6, label = 'setosa')
plt.scatter(X_test_pca[index_4, 0], X_test_pca[index_4, 1], color = 'green', alpha = 0.6, label = 'versicolor')
plt.title('Real target', size = 13)
plt.legend()

plt.subplot(122)
plt.scatter(X_test_pca[index_2_p, 0], X_test_pca[index_2_p, 1], color = 'purple', alpha = 0.6, label = 'setosa')
plt.scatter(X_test_pca[index_4_p, 0], X_test_pca[index_4_p, 1], color = 'green', alpha = 0.6, label = 'versicolor')
plt.title('SVM result', size = 13)
plt.legend()
plt.show()