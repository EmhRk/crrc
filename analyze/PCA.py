"""
kernel learning, using Gaussian kernel function
to find a new space that this 77 labeled recorded is closed to each other
"""
import numpy as np
import numpy.linalg as linalg


crrc_res_label = np.load('crrc_result.npy')
mean_col_scale = crrc_res_label.mean(axis=0)
std_cole_scale = crrc_res_label.std(axis=0)
crrc_res_label = (crrc_res_label - mean_col_scale) / std_cole_scale


crrc_res = np.array(crrc_res_label)
eig_value, eig_vector = np.linalg.eig(np.cov(crrc_res.T))
co = 0
for i, e in enumerate(eig_value):
    co += e / np.sum(eig_value)
    print(i+1, co)


new_data = crrc_res.dot(eig_vector[:, :12])

print(new_data.shape)

distance_recorder = 0
for i in range(len(crrc_res_label - 1)):
    for j in range(i + 1, len(crrc_res_label)):
        distance_recorder += linalg.norm(crrc_res_label[i, :]-crrc_res_label[j, :])
total = 56 * 55 / 2
print(distance_recorder/total/15)


mean_col_scale = new_data.mean(axis=0)
std_cole_scale = new_data.std(axis=0)
new_data = (new_data - mean_col_scale) / std_cole_scale

distance_recorder = 0
for i in range(len(crrc_res_label - 1)):
    for j in range(i + 1, len(crrc_res_label)):
        distance_recorder += linalg.norm(new_data[i, :]-new_data[j, :])
total = 56 * 55 / 2
print(distance_recorder/total/12)
