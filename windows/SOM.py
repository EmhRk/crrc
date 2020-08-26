from sklearn.ensemble import IsolationForest
import numpy as np
import random
import psutil
import matplotlib.pyplot as plt
from minisom import MiniSom

key_list = np.load('./keys/key_list.npy')

print('Start Process...')

MAX_NUM = 11321
rate = 60 / 11321
memory_threshold = 0.85

abnormal_detect_list = np.sort(random.sample(range(1, MAX_NUM), int(MAX_NUM * rate)))

crrc_matrix = None
crrc_key = None

for i, index in enumerate(abnormal_detect_list):
    if (i+1) % 50 == 0:
        print('processing:{:.2f}'.format(100 * (i + 1) / MAX_NUM / rate), end='')
        print('%', end='')
        print(' (%s/%s)' % (i + 1, int(MAX_NUM * rate)))
    matrix = np.load('./windows/window_matrix_%s.npy' % str(index))
    key = np.load('./keys/keys_%s.npy' % str(index))
    if crrc_matrix is None:
        crrc_matrix = matrix
        crrc_key = key
    else:
        try:
            crrc_matrix = np.concatenate((crrc_matrix, matrix), axis=0)
        except ValueError:
            print('Illegal matrix shape in index %s (%s)' % (str(index), str(matrix.shape)))

        try:
            crrc_key = np.concatenate((crrc_key, key), axis=0)
        except ValueError:
            print('Illegal key shape in index %s (%s)' % (str(index), str(key.shape)))

    memory = psutil.virtual_memory()
    if memory.used / memory.total >= memory_threshold:
        print('Memory almost full. Process interrupted')
        break

memory = psutil.virtual_memory()
print('Memory usage: {:.1f}'.format(100 * memory.used / memory.total), end='')
print('%')
print('crrc_matrix shape: %s' % str(crrc_matrix.shape))
print('crrc_key shape: %s' % str(crrc_key.shape))

size = int(np.sqrt(5 * np.sqrt(crrc_matrix.shape[0])))

som = MiniSom(size, size, crrc_matrix.shape[1], sigma=0.3, learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)
som.one_train(crrc_matrix, 500, verbose=True)  # trains the SOM with 100 iterations

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(crrc_matrix[i, :]) for i in range(crrc_matrix.shape[0])]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, (size, size))

plt.scatter(range(len(cluster_index)), cluster_index)
plt.show()

# # plotting the clusters using the first 2 dimentions of the data
# for c in np.unique(cluster_index):
#     plt.scatter(crrc_matrix[cluster_index == c, 0],
#                 crrc_matrix[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
#
# # plotting centroids
# for centroid in som.get_weights():
#     plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
#                 s=80, linewidths=35, color='k', label='centroid')
# plt.legend()
# plt.show()
