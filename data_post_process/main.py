#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME     : 2018/2/22 11:34
# @Author  : Gyt

import numpy as np
import matplotlib.pyplot as plt
data_predict = np.array(np.loadtxt('../self_predict/predict_stress.txt'))
data_DNS = np.array(np.loadtxt('../data_pre_process/clean_data/SAD/output_layer.txt'))

yz_grid = np.array(np.loadtxt('./grid_data/SAD/yz.dat'))
y_axis = yz_grid[:, 0].reshape([258, 258])
z_axis = yz_grid[:, 1].reshape([258, 258])

# 画图
plt.figure(1)
# plt.plot(data_predict[:, 3], data_DNS[:, 3], linestyle='-')
plt.scatter(data_predict[:, 3], data_DNS[:, 3])
plt.plot([-0.5, 2.5], [-0.5, 2.5], color='r', linestyle='-')
# plt.show()

# retrieve the data : step 1 multiply by mean value, get back to real stress
mean_output = np.array(np.loadtxt('../data_pre_process/clean_data/SAD/mean_output.txt'))
data_predict = data_predict * mean_output
data_DNS = data_DNS * mean_output
# mean error of component of stresses

# retrieve the data : step 2 add boundary value
n_row = 0
for i in range(258):
    for j in range(258):
        if ((i == 0) or (i == 257) or (j == 0) or (j == 257)) or ((i >= 65 and i <= 192) and (j >= 65 and j <= 192)):
            data_predict = np.insert(data_predict, n_row, 0, axis=0)
            data_DNS = np.insert(data_DNS, n_row, 0, axis=0)
        n_row += 1

# mean processing
data_predict_mp = np.zeros([6, 258, 258])
data_DNS_mp = np.zeros([6, 258, 258])
for i in range(6):
    data_predict_mp[i] = data_predict[:, i].reshape(258, 258)
    data_DNS_mp[i] = data_DNS[:, i].reshape(258, 258)

for j in range(128):
    for i in range(128):
        data_predict_mp[0, i, j] = (data_predict_mp[0, i, j] + data_predict_mp[0, 257 - i, j]
                                    + data_predict_mp[0, i, 257 - j] + data_predict_mp[0, 257 - i, 257 - j])/4
        data_predict_mp[0, 257 - i, j] = data_predict_mp[0, i, j]
        data_predict_mp[0, i, 257 - j] = data_predict_mp[0, i, j]
        data_predict_mp[0, 257 - i, 257 - j] = data_predict_mp[0, i, j]

for j in range(128):
    for i in range(128):
        data_predict_mp[1, i, j] = (data_predict_mp[1, i, j] - data_predict_mp[1, 257 - i, j]
                                    + data_predict_mp[1, i, 257 - j] - data_predict_mp[1, 257 - i, 257 - j])/4
        data_predict_mp[1, 257 - i, j] = -data_predict_mp[1, i, j]
        data_predict_mp[1, i, 257 - j] = data_predict_mp[1, i, j]
        data_predict_mp[1, 257 - i, 257 - j] = -data_predict_mp[1, i, j]

for j in range(128):
    for i in range(128):
        data_predict_mp[2, i, j] = (data_predict_mp[2, i, j] + data_predict_mp[2, 257 - i, j]
                                    - data_predict_mp[2, i, 257 - j] - data_predict_mp[2, 257 - i, 257 - j])/4
        data_predict_mp[2, 257 - i, j] = data_predict_mp[2, i, j]
        data_predict_mp[2, i, 257 - j] = -data_predict_mp[2, i, j]
        data_predict_mp[2, 257 - i, 257 - j] = -data_predict_mp[2, i, j]

for j in range(128):
    for i in range(128):
        data_predict_mp[3, i, j] = (data_predict_mp[3, i, j] + data_predict_mp[3, 257 - i, j]
                                    + data_predict_mp[3, i, 257 - j] + data_predict_mp[3, 257 - i, 257 - j])/4
        data_predict_mp[3, 257 - i, j] = data_predict_mp[3, i, j]
        data_predict_mp[3, i, 257 - j] = data_predict_mp[3, i, j]
        data_predict_mp[3, 257 - i, 257 - j] = data_predict_mp[3, i, j]

for j in range(128):
    for i in range(128):
        data_predict_mp[4, i, j] = (data_predict_mp[4, i, j] + data_predict_mp[4, 257 - i, j]
                                    + data_predict_mp[4, i, 257 - j] + data_predict_mp[4, 257 - i, 257 - j])/4
        data_predict_mp[4, 257 - i, j] = -data_predict_mp[4, i, j]
        data_predict_mp[4, i, 257 - j] = -data_predict_mp[4, i, j]
        data_predict_mp[4, 257 - i, 257 - j] = data_predict_mp[4, i, j]

for j in range(128):
    for i in range(128):
        data_predict_mp[5, i, j] = (data_predict_mp[5, i, j] + data_predict_mp[5, 257 - i, j]
                                    + data_predict_mp[5, i, 257 - j] + data_predict_mp[5, 257 - i, 257 - j])/4
        data_predict_mp[5, 257 - i, j] = data_predict_mp[5, i, j]
        data_predict_mp[5, i, 257 - j] = data_predict_mp[5, i, j]
        data_predict_mp[5, 257 - i, 257 - j] = data_predict_mp[5, i, j]

# save file
file_path_RANS = './predict_stress_for_RANS/'
file_path_draw = './predict_stress_for_drawing/'
stress_name = ["upup", "upvp", "upwp", "vpvp", "vpwp", "wpwp"]
file_title = '''TITLE    ="Plot3D DataSet"\nVARIABLES = "y" "z" "{}"\nDATASETAUXDATA Common.SpeedOfSound="1.0"
DATASETAUXDATA Common.VectorVarsAreVelocity="FALSE"\nZONE T="Zone-original grid"\nSTRANDID=0, SOLUTIONTIME=0
I=258, J=258, K=1, ZONETYPE=Ordered\nDATAPACKING=POINT\nDT=(SINGLE SINGLE SINGLE )'''

for i in range(6):
    np.savetxt(file_path_RANS + stress_name[i] + '.txt', data_predict[:, i])
    np.savetxt(file_path_draw + stress_name[i] + '.txt', np.hstack((yz_grid, data_predict[:, i].reshape([-1,1]))),
               header=file_title.format(stress_name[i]), comments='')

np.savetxt(file_path_draw + 'all_stress' + '.txt', np.hstack((yz_grid, data_predict)),
           header=file_title.format('" "'.join(stress_name)), comments='')
