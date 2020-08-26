# -*- coding: utf-8 -*-

import csv
import numpy as np
import psutil


# title = '行号,里程标,特殊数据,距离,信号机编号,信号状态,LKJ速度,LKJ限速,工况－零位,工况－前后位,工况－牵引位,机车管压,机车缸压,转速（电流）,均衡风缸压力1,均衡风缸压力2,总纲压力1,总纲压力2,数据记录有效位,预留字段,机车型号,机车车号,机车车次,机车司机号,LKJ记录事件代码,设备厂家,设备类型,设备,开车时间,LKJ记录事件发生时间,文件创建日期,CSV文件名,隶属文件夹,文件日期,标识字段'
# title_list = title.split(',')
# data_Mat = []

crrc_matrix = []
crrc_time = []

broken = False
temp_list = []
temp_list_time = []
# min_len = 1
memory_threshold = 0.85
min_len = 30 * 60 / 8
with open('C:\\Users\\Emh\\Desktop\\crrc_suck\\crrc\\crrc_rawdata_2020.csv', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        memory = psutil.virtual_memory()
        if (i+1) % 100000 == 0:
            print("{:.2f}".format(memory.used/memory.total))
            print(len(crrc_matrix))
        if memory.used/memory.total >= memory_threshold:
            break
        st = ''
        for r in row:
            st = st + r
        data = st.split('\t')
        # a new serious record
        if broken:
            # if len(temp_list) != 0:
            #     print(1)
            for flitted_data in temp_list:
                crrc_matrix.append(flitted_data)
            for flitted_data_time in temp_list_time:
                crrc_time.append(flitted_data_time)
            temp_list = []
            temp_list_time = []
            broken = False
        # if data[-6] == '2020-01-01 10:51:05' and data [21] == '6408':
        # print(data)
        single_data = []
        # print(data[0])
        chosen_data = [data[0], data[1], data[3]] + data[5:19]
        try:
            # single_data.append(data[-6] + ' ' + data[21])
            for element in chosen_data:
                if element == 'NULL':
                    single_data.append(0)
                else:
                    single_data.append(float(element))
            temp_list.append(single_data)
            temp_list_time.append(data[-7] + ' ' + data[21])
        except ValueError:
            if len(temp_list) <= min_len:
                temp_list = []
                temp_list_time = []
            broken = True
            continue

print(np.array(crrc_matrix).shape)
np.save('D:/workspace/PycharmProjects/PythonWorkspace/crrc/windows/crrc_matrix', crrc_matrix)
np.save('D:/workspace/PycharmProjects/PythonWorkspace/crrc/windows/crrc_time', crrc_time)
