"""
kernel learning, using Gaussian kernel function
to find a new space that this 77 labeled recorded is closed to each other
"""
import csv
import numpy as np
import matplotlib.pyplot as plt


def Kernel(x_i, x_j, gamma):
    vector = (x_i - x_j)
    return np.exp(- np.dot(vector, vector.T) / 2 * gamma)


def distance(x_i, x_j, gam):
    """
    :param gamma:
    :param x_i: vector one
    :param x_j: vector two
    :return: K(x_i, x_i) - 2 K(x_i, x_j) + K(x_j, x_j)
    """
    return (Kernel(x_i, x_i, gam) - 2 * Kernel(x_i, x_j, gam) + Kernel(x_j, x_j, gam)) ** 0.5


crrc_res_labeled = []
with open('crrc_result_2020.csv', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    print("abnormal element: ", end='')
    for i, row in enumerate(csv_reader):
        st = ''.join(row)
        data = st.split('\t')
        if len(data) != 33:
            data = data[:17] + data[17].split(' ') + data[18:]
        if len(data) != 33:
            data = data[:18] + data[18].split(' ') + data[19:]
        extracted_data = data[14:29]
        temp_list = []
        try:
            for element in extracted_data:
                if element == 'NULL':
                    temp_list.append(0)
                else:
                    temp_list.append(float(element))
            crrc_res_labeled.append(temp_list)
        except ValueError:
            print(element, end='|')
            continue
    print('\n total legal row: %s, total row read: %s' % (len(crrc_res_labeled), i + 1))

crrc_res_labeled = np.array(crrc_res_labeled)

crrc_res_labeled = (crrc_res_labeled - np.mean(crrc_res_labeled, axis=0)) / np.std(crrc_res_labeled, axis=0)

# distance_recorder = []
# b = np.linalg.norm(crrc_res_labeled[0, :] - crrc_res_labeled[1, :])
# for i in range(len(crrc_res_labeled - 1)):
#     for j in range(i + 1, len(crrc_res_labeled)):
#         distance_recorder.append(np.linalg.norm(crrc_res_labeled[i, :] - crrc_res_labeled[j, :]))
# print('liner', '{:.4f}'.format(np.mean(distance_recorder) / b), '{:.4f}'.format(np.std(distance_recorder)))
#
# for gam in range(5, 205, 5):
#     b = distance(crrc_res_labeled[0, :], crrc_res_labeled[1, :], gam / 100)
#     distance_recorder = []
#     for i in range(len(crrc_res_labeled-1)):
#         for j in range(i+1, len(crrc_res_labeled)):
#             distance_recorder.append(distance(crrc_res_labeled[i, :], crrc_res_labeled[j, :], gam / 100))
#     print(gam/100, '{:.4f}'.format(np.mean(distance_recorder) / b), '{:.4f}'.format(np.std(distance_recorder)))

crrc_matrix = np.load('../windows/crrc_matrix.npy')
crrc_matrix = (crrc_matrix - np.mean(crrc_matrix, axis=0)) / np.std(crrc_matrix, axis=0)
size = 2000
data_matrix = crrc_matrix[:size, :]
del crrc_matrix
print(data_matrix.shape)
distance_recorder = []
distance_recorder_liner = []
for i in range(size):
    for j in range(56):
        reconstruct_data_0 = crrc_res_labeled[j, :].tolist()
        reconstruct_data_1 = data_matrix[i, :].tolist()
        reconstruct_data_0 = reconstruct_data_0[0:4] + reconstruct_data_0[6:-4]
        reconstruct_data_1 = reconstruct_data_1[:3] + [reconstruct_data_1[4]] + [reconstruct_data_1[3]] + reconstruct_data_1[8:12]
        distance_recorder_liner.append(np.linalg.norm(np.array(reconstruct_data_0) - np.array(reconstruct_data_1)))
        distance_recorder.append(distance(np.array(reconstruct_data_0), np.array(reconstruct_data_1), 0.5))

x = [i for i in range(2000)]
plt.xlabel('x')
plt.ylabel('y')
plt.title('requests')
plt.scatter(x, distance_recorder_liner, color='m', label='liner')
plt.scatter(x, distance_recorder, color='c', label='kernel')
plt.show()
