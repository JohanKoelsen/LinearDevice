#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:48:25 2023

@author: johan
"""

import numpy as np
import sys
import os
import glob
import pyfftw
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy
from scipy.fft import rfftfreq
import math
t1, t2 = 0.8e-7, 1.4e-7
data_pol = ['Ex', 'Ex', 'Ez']
path = "/home/johan/epoch/scripts/wave_trapping_analysis/PDI/" #os.path.dirname(__file__)

def sort_files(files_l):
    files_dict = dict()
    for type_index, data_type in enumerate(list(set(data_pol))):
        files = []
        for file_index, file in enumerate(files_l):
            field_polarisation = file.split('.')[0][-2:]
            if field_polarisation == data_type:
                files.append(file)
        sim_times = [X.split('_')[-2] for X in files]
        sim_times = [float(X.split('-')[0]) for X in sim_times]
        files = [x for _, x in sorted(zip(sim_times, files), key=lambda pair: pair[0])]
        files_dict[data_type] = files
    return files_dict

def get_files(files, data_pol):
    files_l = []
    for file in files:
        if not os.path.isdir(file) and file.split('.')[0][-2:] in data_pol:
            files_l.append(file)
    files_dict = sort_files(files_l)
    print(files_dict)
    return files_dict
    
def load_fields(data_pol, files_dict, path):
    fields = dict()
    time_tot = dict()
    for data_type in list(set(data_pol)):
        fields[data_type] = []
        time_tot = []
        for file in files_dict[data_type]:
            data = np.load(file)
            field    = data['ElectricField{}'.format(data_type)]
            grid     = data['GridGrid'][0]
            print('loading in', file[len(path):])
            time     = data['time']
            fields[data_type].extend(field)
            time_tot.extend(time)
    return fields, time_tot, grid

def load_in_data_master(data_pol, path):
    files = glob.glob(path + '/*')
    files_dict = get_files(files, data_pol)
    fields, time_tot, grid = load_fields(data_pol, files_dict, path)
    return fields, time_tot, grid
    


def get_time_elements(t1, t2, time_tot):
    time_tot = np.array(time_tot)
    l = np.where((time_tot > t1) & (time_tot < t2))
    return [l[0][0], l[0][-1]]

def cut_fields(eles, time_tot, grid, fields, data_pol):
    t1 = int(eles[0])
    t2 = int(eles[1])
    x01 = int(len(grid)//2 - 150)
    x02 = int(len(grid)//2 + 150)
    for data_type in list(set(data_pol)):
        fields[data_type] = np.array(fields[data_type])[t1:t2, x01:x02]
    time_tot = np.array(time_tot)[t1:t2]
    grid = np.array(grid)[x01:x02]
    return fields, time_tot, grid
   #fields = np.array(fields[])



def FFT1Dr_f(a, threads=1):
    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid),    dtype='float32')
    a_out = pyfftw.empty_aligned((grid//2+1),dtype='complex64')
    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
    # put input array into delta_r and perform FFTW
    a_in = a
    fftw_plan(a_in, a_out)
    return a_out

def compute_FFTs(fields, grid, time_tot):
    fields_k = dict()
    for data_type in list(set(data_pol)):
        amount_of_freqs = np.shape(fields[data_type][:, 0])[0]//2 +1
        amount_of_grid = len(grid)
        fields_k[data_type] = np.zeros([amount_of_freqs, amount_of_grid], dtype=np.complex64)
        for x0 in range(amount_of_grid):
            field_x0 = fields[data_type][:, x0]
            fields_k[data_type][:, x0] = FFT1Dr_f(field_x0)*1e-10
         
    fs = 1/np.mean(np.diff(time_tot))
    freqs = rfftfreq(len(time_tot), 1/fs)
    

    return fields_k, freqs

def interp_and_window_fields(fields, data_pol, time_tot, grid):
    time_lin = np.linspace(time_tot[0], time_tot[-1], len(time_tot))
    window = scipy.signal.windows.hann(len(time_tot))
    for data_type in data_pol:
        for x0 in range(len(grid)):
            field_x0 = fields[data_type][:,x0]
            bspl = make_interp_spline(time_tot, field_x0, k=5)
            fields[data_type][:, x0] = window*bspl(time_lin)
            
            
            
def plot_x_f(fields_k, grid, freqs, pol='Ex'):
    plt.figure(0)
    print(np.shape(fields_k[pol]), np.shape(grid), np.shape(freqs))
    plt.pcolormesh(grid, freqs, np.log(abs(fields_k[pol])))
    plt.ylabel(r'$f$')
    plt.xlabel('r$x$')

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def fill_out_diag(fs1_len, ele_mapping, ele_mapping_temp):
    for i in range(2*fs1_len - 1):
        if i < fs1_len:
            np.fill_diagonal(ele_mapping[-i-1:, :], ele_mapping_temp[i])
        else:
            np.fill_diagonal(ele_mapping[:, i-len(ele_mapping_temp):], ele_mapping_temp[i])
    ele_mapping = np.flipud(ele_mapping)
    
def f_ele(freqs, fs):
    fmin, fmax = fs
    fmin_ele = np.where(freqs > fmin)[0][0]
    fmax_ele = np.where(freqs < fmax)[0][-1]
    return fmin_ele, fmax_ele
    
def create_ele_mapping(f1s, f2s, freqs):
    f1min_ele, f1max_ele = f_ele(freqs, f1s)
    f2min_ele, f2max_ele = f_ele(freqs, f2s)
    fs1_len = np.shape(freqs[f1min_ele:f1max_ele])[0]
    fs2_len = np.shape(freqs[f2min_ele:f2max_ele])[0]
    
    ele_mapping = np.zeros([fs1_len, fs2_len], dtype=int)
    if ((f1min_ele == f2min_ele) and (f1max_ele == f2max_ele)):
        print('f1min == f2min and f1max == f2max')
        ele_mapping_temp = np.zeros(2*fs1_len - 1, dtype=int)
        for index, i in enumerate(range(f1min_ele,f1max_ele)):
            fs_sum = freqs[i] + freqs[i]
            ele_val = find_nearest(freqs, fs_sum)
            ele_val_ele = np.where(freqs == ele_val)[0][0]
            ele_mapping_temp[2*index] = ele_val_ele
            if i != f1max_ele-1:
                fs_sum = freqs[i] + freqs[i+1]
                ele_val = find_nearest(freqs, fs_sum)
                ele_val_ele = np.where(freqs == ele_val)[0][0]
                ele_mapping_temp[2*index + 1] = ele_val_ele
        fill_out_diag(fs1_len, ele_mapping, ele_mapping_temp)
        ele_mapping = np.flipud(ele_mapping)
    else:
        print('f1min != f2min or f1max != f2max')
        for i in range(fs1_len):
            for j in range(fs2_len):
                fs_sum = freqs[f1min_ele:f1max_ele][i] + freqs[f2min_ele:f2max_ele][j]
                fs_sum_val = find_nearest(freqs, fs_sum)
                fs_sum_ele = np.where(freqs == fs_sum_val)[0][0]
                ele_mapping[i, j] = int(fs_sum_ele)
    return ele_mapping

def cut_specs(freqs, fs, fields, data_pol):
    fields_k_cut = dict()
    for index, data_type in enumerate(data_pol[:-1]):
        fmin_ele, fmax_ele = f_ele(freqs, fs[index])
        print(fmin_ele, fmax_ele)
        fields_k_cut[index] = fields_k[data_type][fmin_ele: fmax_ele]
    return fields_k_cut

def compute_bicoherence(fields_k, fields_k_cut, data_pol, ele_mapping, bicoherence_boolean=True):
    
    field_val = np.zeros(np.shape(fields_k[data_pol[-1]][ele_mapping][:,:, 0]), dtype=np.complex64)
    temp = fields_k_cut[0][:, None, :] * fields_k_cut[1][None, :, :]
    if bicoherence_boolean:
        norm1 = np.mean(abs(temp)**2, axis=2)
        norm2 = np.mean(abs(np.conjugate(fields_k[data_pol[-1]][ele_mapping]))**2, axis=2)
    temp *= np.conjugate(fields_k[data_pol[-1]][ele_mapping])
    field_val += np.mean(temp, axis=2)
    del temp
    if bicoherence_boolean:
        field_val = abs(field_val)**2
        field_val /= (norm1 * norm2)
        
    return field_val

def plot_field(freqs, fs, field, data_pol, bicoherence_bool=True):
    f1min_ele, f1max_ele = f_ele(freqs, fs[0])
    f2min_ele, f2max_ele = f_ele(freqs, fs[1])
    plt.figure(1)
    if bicoherence_bool:
        plt.pcolormesh(freqs[f1min_ele:f1max_ele], freqs[f2min_ele:f2max_ele], field.T)
        plt.colorbar()
        plt.clim(0,1)
    else:
        plt.pcolormesh(freqs[f1min_ele:f1max_ele], freqs[f2min_ele:f2max_ele], np.log(abs(field.T)))
        plt.colorbar()
    plt.xlabel(r'$f$[GHz]' + str(data_pol[0]))
    plt.ylabel(r'$f$[GHz]' + str(data_pol[1]))
    
#%%
fields, time_tot, grid = load_in_data_master(data_pol, path)

#%%
eles = get_time_elements(t1, t2, time_tot)
fields, time_tot, grid = cut_fields(eles, time_tot, grid, fields, data_pol)
#%%
interp_and_window_fields(fields, data_pol, time_tot, grid)
#%%
fields_k, freqs = compute_FFTs(fields, grid, time_tot)

print(np.shape(fields_k['Ex'][:,0]), np.shape(freqs))
#%%
#plot_x_f(fields_k, grid, freqs)
#%%
f1max = 75e9
f1min = 55e9
f2max = 75e9
f2min = 55e9
f1s = [f1min, f1max]
f2s = [f2min, f2max]
ele_mapping = create_ele_mapping(f1s, f2s, freqs)
fields_k_cut = cut_specs(freqs, [f1s, f2s], fields_k, data_pol)


#%%
fmin_ele, fmax_ele = f_ele(freqs, f1s)
#plot_x_f(fields_k_cut,grid, freqs[fmin_ele:fmax_ele],pol=1)

#%%
bicoherence_bool = True
field_val = compute_bicoherence(fields_k, fields_k_cut, data_pol, ele_mapping, bicoherence_bool)
print(field_val)
plot_field(freqs, [f1s, f2s], field_val, data_pol)
print(np.shape(field_val))
