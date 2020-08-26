from sklearn.ensemble import IsolationForest
import numpy as np

crrc_matrix = np.load('../windows/crrc_matrix.npy')
model_isof = IsolationForest()
# 计算有无异常的标签分布
outlier_label = model_isof.fit_predict(crrc_matrix)

outlier_counter = 0
for outlier in outlier_label:
    if outlier == -1:
        outlier_counter += 1

print(outlier_counter/len(outlier_label))
