#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME     : 2018/2/9 22:11
# @Author  : Gyt

import pandas as pd
import numpy as np

# read strain rate data, components are S21, S22, S31, S32, S33, R32
file_name = './original_data/SAD/strain rate and rotation tensor.dat'
strain_rate_data = np.array(np.loadtxt(file_name))

S21 = strain_rate_data[:, 0]
S22 = strain_rate_data[:, 1]
S31 = strain_rate_data[:, 2]
S32 = strain_rate_data[:, 3]
S33 = strain_rate_data[:, 4]
R32 = strain_rate_data[:, 5]
# strain_rate_data = strain_rate_data.reshape([1, -1])

# self_predict layer data reading
file_name = './original_data/SAD/Reynold stresses.dat'
stress_data = np.array(np.loadtxt(file_name))
# stress_data = stress_data.reshape([1, -1])

# get efficient points and clean data
point_matrix = np.ones([258, 258])
for i in range(258):
    for j in range(258):
        if ((i == 0) or (i == 257) or (j == 0) or (j == 257)) or ((i >= 65 and i <= 192) and (j >= 65 and j <= 192)):
            point_matrix[i, j] = 0
point_matrix = np.where(np.tile(point_matrix.reshape(-1, 1), [1, 6])>0)
input_layer = np.reshape(strain_rate_data[point_matrix], [-1, 6])
output_layer = np.reshape(stress_data[point_matrix], [-1, 6])

# mean value
mean_input = np.mean(np.abs(input_layer), axis=0)
mean_output = np.mean(np.abs(output_layer), axis=0)

# standardization with mean value
new_input_layer = input_layer / mean_input
new_output_layer = output_layer / mean_output

# self_predict the data
file_path = './clean_data/SAD/'
np.savetxt(file_path + 'mean_input.txt', mean_input)
np.savetxt(file_path + 'mean_output.txt', mean_output)
np.savetxt(file_path + 'input_layer.txt', new_input_layer)
np.savetxt(file_path + 'output_layer.txt', new_output_layer)
