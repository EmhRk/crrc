import numpy as np

MAX_NUM = 11321

crrc_matrix = None
key_list = []

for i, index in enumerate(range(MAX_NUM)):
    crrc_matrix = np.load('./keys/keys_%s.npy' % str(index))
    print('{:.2f}'.format(100 * (i + 1) / MAX_NUM), end='')
    print('%')
    for single_record in crrc_matrix.tolist():
        if single_record not in key_list:
            key_list.append(single_record)

key_list = np.array(key_list)
print(key_list.shape)
np.save('./keys/key_list', key_list)
