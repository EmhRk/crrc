import numpy as np
# import matplotlib.pyplot as plt


crrc_matrix = np.load('./crrc_matrix.npy')
crrc_time = np.load('./crrc_time.npy')  # 发车年月日 发车时间 机车型号

start_sign = 0

continue_data = []
continue_keys = []

# min_len = int(30 * 60 / 8)
window_size = 20

handle_size = crrc_matrix.shape[0]

counter = 0

key_dict = {}


for i in range(handle_size - 1):
    if (i+1) % 5000 == 0:
        print("{:.2f}".format(100 * (i + 1) / handle_size), end='')
        print('%')

    if crrc_time[i] == crrc_time[i + 1] and crrc_matrix[i, 0] - crrc_matrix[i + 1, 0] == -1:
        continue_data.append(crrc_matrix[i, 1:])
        continue_keys.append(str(crrc_time[i]) + '^' + str(crrc_matrix[i, 0]))

    else:
        continue_data.append(crrc_matrix[i, 1:])
        continue_keys.append(str(crrc_time[i]) + '^' + str(crrc_matrix[i, 0]))

        if len(continue_data) - window_size == 0:
            assisted_list = []
            assisted_key_list = []
            for j in range(window_size):
                assisted_list.extend(continue_data[j].tolist())
                assisted_key_list.append(continue_keys[j])

            if len(assisted_key_list) != 0:
                np.save('./windows/window_matrix_' + str(counter), np.array(assisted_list))
                np.save('./keys/keys_' + str(counter), np.array(assisted_key_list))
                counter += 1
        else:
            data_matrix = []
            keys = []
            for j in range(len(continue_data) - window_size):
                assisted_list = []
                assisted_key_list = []
                for k in range(window_size):
                    assisted_list.extend(continue_data[j + k].tolist())
                    assisted_key_list.append(continue_keys[j + k])

                    # if continue_keys[j + k] not in key_dict.keys():
                    #     key_dict[continue_keys[j + k]] = 1
                    # else:
                    #     key_dict[continue_keys[j + k]] += 1

                data_matrix.append(assisted_list)
                keys.append(assisted_key_list)
            if len(keys) != 0:
                if len(data_matrix) != len(keys):
                    input('Illegal shape')
                np.save('./windows/window_matrix_' + str(counter), np.array(data_matrix))
                np.save('./keys/keys_' + str(counter), np.array(keys))  # [0, 1, ..., n] * num_windows
                counter += 1
        continue_data.clear()
        continue_keys.clear()

