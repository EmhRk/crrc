from sklearn.ensemble import IsolationForest
import numpy as np
import random
import psutil
import matplotlib.pyplot as plt

# key_list = np.load('./keys/key_list.npy')


print('Start Process...')

len_list = []

# MAX_NUM = 11321
MAX_NUM = 1200
rate = 0.5
memory_threshold = 0.85
threshold = 0.4
num_sampling = 25
abnormal_counter_dict = {}
number_tickets = {}
total_abnormal_counter = 0
total_counter = 0

for step in range(num_sampling):
    abnormal_detect_list = random.sample(range(1, MAX_NUM), int(MAX_NUM * rate))
    abnormal_detect_list.sort()

    crrc_matrix = None
    crrc_key = None

    for i, index in enumerate(abnormal_detect_list):
        if (i + 1) % 50 == 0:
            print('Sampling step %s (%s/%s), processing:{:.2f}'.format(100 * (i + 1) / MAX_NUM / rate)
                  % (step+1, step+1, num_sampling), end='')
            print('%', end='')
            print(' (%s/%s)' % (i + 1, int(MAX_NUM * rate)))
        matrix = np.load('./windows/window_matrix_%s.npy' % str(index))
        key = np.load('./keys/keys_%s.npy' % str(index))

        if matrix.shape[0] != key.shape[0]:
            print('Unequal shape %s ' % index)
        len_list.append(key.shape[0])

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

    if crrc_matrix.shape[0] != crrc_key.shape[0]:
        input('Illegal shape')

    model_isof = IsolationForest(contamination='auto')
    outlier_label = model_isof.fit_predict(crrc_matrix)

    one_step_abnormal_counter_dict = {}
    one_step_number_tickets = {}
    total_counter += len(outlier_label)
    for i, lier in enumerate(outlier_label):
        if lier == -1:
            total_abnormal_counter += 1
        for one_train in crrc_key[i, :].tolist():
            if one_train not in one_step_abnormal_counter_dict.keys():
                one_step_abnormal_counter_dict[one_train] = 0

            if one_train not in one_step_number_tickets.keys():
                one_step_number_tickets[one_train] = 1
            else:
                one_step_number_tickets[one_train] += 1

            if lier == -1:
                one_step_abnormal_counter_dict[one_train] += 1

    for key in list(one_step_abnormal_counter_dict.keys()):
        if one_step_number_tickets[key] != 20:
            continue
        else:
            if key in abnormal_counter_dict.keys():
                abnormal_counter_dict[key] += one_step_abnormal_counter_dict[key]
                number_tickets[key] += one_step_number_tickets[key]
            else:
                abnormal_counter_dict[key] = one_step_abnormal_counter_dict[key]
                number_tickets[key] = one_step_number_tickets[key]

# plt.plot(range(len(outlier_label)), outlier_label)
# plt.title('outlier_label')
# plt.show()
#
# plt.plot(range(len(len_list)), len_list)
# plt.title('len_list')
# plt.show()
#
# plt.plot(range(len(list(abnormal_counter_dict.values()))), list(abnormal_counter_dict.values()))
# plt.title('abnormal_counter_dict.values')
# plt.show()
#
plt.scatter(range(len(list(number_tickets.values()))), list(number_tickets.values()), s=1, marker='.')
plt.title('number_tickets.values')
plt.show()

for key in list(abnormal_counter_dict.keys()):
    abnormal_counter_dict[key] /= number_tickets[key]

ones_counter = 0
abnormal_detect_result = list(abnormal_counter_dict.values())
level = [0] * 101
total_level = len(abnormal_detect_result)
for r in abnormal_detect_result:
    level[int(r * 100)] += 1 / total_level
    if r == 1:
        ones_counter += 1

print(level)

axis_x = []
axis_y = []
for x, y in zip(range(101), level):
    if y != 0:
        axis_x.append(x)
        axis_y.append(y)

plt.scatter(axis_x, axis_y, s=20, marker='x', c='r')
plt.plot(range(101), level)
plt.title('level.plot')
plt.show()

plt.scatter(range(len(abnormal_detect_result)), abnormal_detect_result, s=1, marker='.')
# plt.plot(range(len(abnormal_detect_result)), abnormal_detect_result, label='distribution')
plt.title('iForest Vote(abnormal rate: {:.3f})'.format(total_abnormal_counter / total_counter))
# plt.legend()
plt.show()
print('Voted 100% abnormal: {:.2f}'.format(ones_counter / len(abnormal_detect_result)))
