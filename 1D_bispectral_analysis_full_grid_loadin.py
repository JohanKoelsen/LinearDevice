#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:35:54 2023

@author: johan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:16:46 2023

@author: johdew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from tqdm import tqdm
import pprint




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
            # !!!
            sim_end = file.split('-')[-1]
            sim_end = sim_end.split('_')[0]
            if sim_end not in ['13745', '8790', '30833']:
                files_l.append(file)
    files_dict = sort_files(files_l)
    pprint.pprint(files_dict, width=100)
    return files_dict

def find_closest_element(input_value, input_list):
    closest_element = None
    min_difference = float('inf')

    for element in input_list:
        difference = abs(input_value - element)
        if difference < min_difference:
            min_difference = difference
            closest_element = element

    return closest_element

def load_fields(data_pol, files_dict, path):
    fields = dict()
    time_tot = dict()
    print('\nLoading in data ...')
    for data_type in list(set(data_pol)):
        fields[data_type] = []
        time_tot = []
        for file in tqdm(files_dict[data_type]):
            data = np.load(file)
            grid     = data['GridGrid'][0]

            field    = data['ElectricField{}'.format(data_type)]
            
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


def cut_fields(time_eles, time_tot,fields, data_pol, grid, x0):
    x0_ele = np.where(find_closest_element(x0, grid) == grid)[0][0]
    t1 = int(time_eles[0])
    t2 = int(time_eles[1])

    fields_cut = dict()
    for data_type in list(set(data_pol)):
        fields_cut[data_type] = np.array(fields[data_type])[t1:t2, x0_ele]
    time_tot_cut = np.array(time_tot)[t1:t2]
    
    return fields_cut, time_tot_cut


def interp_and_window_fields(field_segment, time_tot, grid):
    time_lin = np.linspace(time_tot[0], time_tot[-1], len(time_tot))
    window_time = scipy.signal.windows.hann(len(time_tot))
    bspl = make_interp_spline(time_tot, field_segment, k=5)
    field_segment = window_time*bspl(time_lin)

    
    return time_lin, field_segment
def get_STFT_time_ele(time_tot, Nsegments, Noverlap, offset=0):
    Nelements = len(time_tot)/Nsegments
   
    Noverlap_elements = Nelements * Noverlap #0.80 #0.90
    starts = np.arange(0, len(time_tot), Nelements - Noverlap_elements, dtype=int)
    starts = starts[starts + Nelements < len(time_tot)]
    t_start = [time_tot[start] for start in starts]
    t_end = [time_tot[start + int(Nelements)] for start in starts]
    return t_start, t_end
 

# Returns the cut Fourier Transformed fields of shape (k_cut, frequencies_cut, time_ele)
def cut_FFT_fields(time_tot, fields, grid, t_starts, t_ends, data_pol):
    field_k = dict()
    for data_type in data_pol:
        t_ele = get_time_elements(t_starts[0], t_ends[0], time_tot)
        field_k[data_type] = np.zeros([len(t_starts), (t_ele[1] - t_ele[0])//2 + 1], dtype=np.complex128)
    print('Applying interpolation and window, and carrying out 2D FFT for', len(t_starts), 'time elements.')
    for index, i in enumerate(tqdm(range(len(t_starts)))):
        field_segment = dict()
        
        for data_type in list(set(data_pol)):
            t_start = t_starts[i]
            t_end = t_ends[i]
            
            t_ele = get_time_elements(t_start, t_end, time_tot)
            time_segment = time_tot[t_ele[0]:t_ele[-1]]
            field_segment[data_type] = fields[data_type][t_ele[0]:t_ele[-1]]
            time_lin, field_segment[data_type] = interp_and_window_fields(field_segment[data_type], time_segment, grid)

            
            field_k[data_type][index] = FFT1Dr_f(field_segment[data_type], threads=1)


            if index == 0:
                dt = np.mean(np.diff(time_lin))
                freqs = rfftfreq(len(time_lin), dt)

      
    return freqs, field_k

                   


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
    



def compute_bicoherence(field_k_cut, data_pol, bicoherence_boolean=True):
    
    field_val = np.zeros(np.shape(field_k_cut[2][0]), dtype=np.complex128)

    temp = field_k_cut[0][:, :, None] * field_k_cut[1][:, None, :]
    if bicoherence_boolean:
        norm1 = np.mean(abs(temp)**2, axis=0)
        norm2 = np.mean(abs(np.conjugate(field_k_cut[2]))**2, axis=0)
    temp *= np.conjugate(field_k_cut[2])
    field_val += np.mean(temp, axis=0)
    del temp
    if bicoherence_boolean:
        field_val = abs(field_val)**2
        field_val /= (norm1 * norm2)
        
    return field_val



def plot_field(freqs, field):
    
    plt.figure()
    plt.plot(freqs, field)
    plt.xlabel(r'$f$ [mm]')
    plt.ylabel(r'$\tilde{S}$')

def create_fancy_window(text, horizontal_char="-", vertical_char="|", border_width=1, use_bold=True):
    bold_code = "\033[1m" if use_bold else ""
    reset_code = "\033[0m"

    lines = text.splitlines()
    max_length = max(len(line) for line in lines)

    # Create the top border
    top_border = '\n\n' + horizontal_char * (max_length + 2 * border_width)
    result = [top_border]

    # Create the middle part with text and borders
    for line in lines:
        padding = " " * (max_length - len(line) + 2)
        bordered_line = f"{vertical_char}{bold_code} {line}{padding}{reset_code}{vertical_char}"
        result.append(bordered_line)

    # Create the bottom border
    bottom_border = horizontal_char * (max_length + 2 * border_width)
    result.append(bottom_border)

    return "\n".join(result)



def load_in_data():
    fields, time_tot, grid = load_in_data_master(data_pol, path)

    text=f"""
    Simulation parameters:
        
    Length of grid: {len(grid)}
    grid min/max: {grid[0]}, {grid[-1]}
    
    Length of time: {len(time_tot)}
    time min/max: {time_tot[0]}, {time_tot[-1]}
    """
    
    fancy_window = create_fancy_window(text, horizontal_char="-", vertical_char="|", border_width=2)
    print(fancy_window)
    return fields, time_tot, grid


def compute_bispectrum(fields, x0, t1, t2, time_tot, grid, Nsegments, Noverlap, data_pol):
    

    
    fields_x0 = dict()
    print('Creating field_x0 with x0 = ', str(x0))
    #for data_type in data_pol:
    #    fields_x0[data_type] = np.zeros(len(time_tot), dtype='float')
    #    fields_x0[data_type] = np.array(fields[data_type])[:, x0_ele]
    time_eles = get_time_elements(t1, t2, time_tot)
    fields, time_tot = cut_fields(time_eles, time_tot, fields, data_pol, grid, x0)
    t_starts, t_ends = get_STFT_time_ele(time_tot, Nsegments, Noverlap)

    freqs, field_k = cut_FFT_fields(time_tot, fields, grid, t_starts, t_ends, data_pol)
    return freqs, field_k



def fill_out_diag(fs1_len, ele_mapping, ele_mapping_temp):
    for i in range(2*fs1_len - 1):
        if i < fs1_len:
            np.fill_diagonal(ele_mapping[-i-1:, :], ele_mapping_temp[i])
        else:
            np.fill_diagonal(ele_mapping[:, i-len(ele_mapping_temp):], ele_mapping_temp[i])
    ele_mapping = np.flipud(ele_mapping)
    
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
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

def cut_specs(freqs, fs, field_k, data_pol, ele_mapping):
    field_k_cut = dict()
    for index, data_type in enumerate(data_pol[:-1]):
        fmin_ele, fmax_ele = f_ele(freqs, fs[index])
        field_k_cut[index] = field_k[data_type][:, fmin_ele: fmax_ele]
    

    field_k_cut[len(data_pol)-1] = field_k[data_pol[-1]][:, ele_mapping]
    return field_k_cut

def plot_2d_field(freqs, fi, fj, field_val):
    f1min, f1max = f_ele(freqs, fi)
    f2min, f2max = f_ele(freqs, fj)
    
    f1s = freqs[f1min:f1max]
    f2s = freqs[f2min:f2max]
    plt.figure()
    plt.pcolormesh(f1s*1e-9, f2s*1e-9, field_val, cmap='plasma')
    plt.colorbar()
    plt.clim(0,1)
    plt.xlabel(r'$f$ [GHz]')
    plt.ylabel(r'$f$ [GHz]')
    plt.show()
    
    
    
def cut_fields_compute_bispectrum(freqs, field_k, fi, fj, data_pol, bicoherence_bool):
    ele_mapping = create_ele_mapping(fi, fj, freqs)
    field_k_cut = cut_specs(freqs, [fi, fj], field_k, data_pol, ele_mapping)
    
    field_val = compute_bicoherence(field_k_cut, data_pol, bicoherence_bool)

    plot_2d_field(freqs, fi, fj, field_val.T)
    
    return field_val
#%%



data_pol = ['Ex', 'Ex', 'Ex']
path = "Z:/Fys/FYS-PPFE/PDI/EPOCH Data/Sorted Data Sets/PD7_Beat_10" #"Z:/Fys/FYS-PPFE/PDI/EPOCH Data/Sorted Data Sets/STPD2"

fields, time_tot, grid = load_in_data()
            

#%%
t1, t2 = 16e-9, 60e-9
Nsegments = 10
Noverlap = 0.7
x0 = 40e-3
freqs, field_k = compute_bispectrum(fields, x0, t1, t2, time_tot, grid, Nsegments, Noverlap, data_pol)

#%%
fi = [0e9, 20e9]
fj = [80e9, 120e9]
bicoherence_bool = True
cut_fields_compute_bispectrum(freqs, field_k, fi, fj, data_pol, bicoherence_bool)