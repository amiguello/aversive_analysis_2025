# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 03:08:39 2023

@author: Albert

Scripts to create the figures in "Transformations of the spatial activity manifold convey aversive information in CA3" (2025)


"""




import time
import scipy
import scipy.io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

import h5py
import pandas as pd



from sklearn import decomposition

# from decoders import WienerFilterRegression, WienerCascadeRegression, KalmanFilterRegression, SVRegression, NaiveBayesRegression,XGBoostRegression
# from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

#Global project variables
import project_parameters as pparam
from project_parameters import (FAT_CLUSTER_PATH, OUTPUT_PATH, SESSION_NAMES, SESSION_REPEATS, MOUSE_TYPE_LABELS)

#Project scripts
import processing_functions as pf
import mCCA as mCCA_funs
import APdecoding_funs as APfuns


plt.rcParams['font.family'] = 'Arial' 

def main():
    
    # save_pca_data(np.arange(8), np.arange(9)) #Uncomment this once to pre-compute PCA data. 
    
    ''' Figure 1 plots '''
    # plot_fig1_C_firing_rates()
    # plot_fig1_D_pcas()
    # plot_fig1_E_position_prediction_example()
    # plot_fig1_F_prediction_across_sessions()
    
    ''' Figure 1 SI plots '''
    # plot_fig1SI_A_traces()
    # plot_fig1SI_B_C_variance_explained()
    # plot_fig1SI_D_prediction_vs_dimensionality()
    
    ''' Figure 2 SI plots '''
    # fig2_CCA_example()
    # fig2_CCA_aligned_pcas(shuffle=False)
    fig2_D_E_F_CCA_quantification()
    
    ''' Figure 2 SI plots '''
    # fig2_CCA_aligned_pcas(shuffle=True)

    ''' Figure 3 plots '''
    # fig3_A_and_fig3SI_A_TCA_factors()
    # fig3_B_LDA_on_TCA()
    # fig3_C_D_f1_plots_and_fig3SI_B_C_accuracy_plots()
    # fig3_F_and_3SI_A_B_belt_restriction_plots()

    ''' Figure 3 SI plots '''
    # fig3SI_C()
    # fig3SI_D()
    # fig3SI_F1_by_CCA_and_TCA_dimension_and_mouse()

    ''' Figure 4 plots '''
    # fig4_A_session_comparisons()  
    # fig4_C_D_E_distance_measures()
    
    ''' Figure 4 SI plots '''
    # fig4SI_G_H_I_distance_measures()
    
    ''' Figure 5 plots '''
    # fig5_A_F1_by_TCA_dimension()
    # fig5_B_C_D_and_fig5SI_A_B_C_D_E_decoding_weight_plots()
    
    ''' Figure 5 SI plots '''
    # fig5SI_F_decoding_weight_control()

    # <>
    

######## Functions related to saving and retrieving data ########

def compare_parameter_dictionaries(dict1, dict2):
    ''' Checks if two dictionaries are equal, return "False" if they are not '''
    are_dictionaries_equal = False
    try:
        np.testing.assert_equal(dict1, dict2)
        are_dictionaries_equal = True
    except AssertionError:
        are_dictionaries_equal = False
    
    return are_dictionaries_equal


def get_analysis_dict_filename(analysis_name):
    ''' name is string (no .npy or previous path)
        if "name" corresponds to an analysis name, give the default name from pparams
        otherwise treat it as a custom name
    '''
    
    if analysis_name in pparam.ANALYSIS_NAME_LIST:
        filename = pparam.default_param_dicts_names[analysis_name]
    else:
        filename = analysis_name
        
    analysis_dict_filename = OUTPUT_PATH + filename + ".npy"
    
    return analysis_dict_filename

def load_previous_analysis(analysis_name):
    '''

    Parameters
    ----------
    analysis_name: 'preprocessing', 'alignment', 'APdecoding'
                    if neither of these, it must be the name to a different analysis dict filename (assumed to be in OUTPUT_PATH, without the .npy)


    Returns
    -------
    Previous analysis dictionary

    '''
    
    analysis_dict_filename = get_analysis_dict_filename(analysis_name)

    try:
        analysis_dict = np.load(analysis_dict_filename, allow_pickle=True)[()]
    except FileNotFoundError:
        analysis_dict = None
    
    return analysis_dict
    

def process_input_parameter_dict(param_dict_new, analysis_name, custom_filename = None):
    ''' Checks incoming parameter dict:
        - Adds missing parameters from default dictionary
        - Compares with old dictionary and returns "are_different" = False if they are different
        
        analysis_name: 'preprocessing', 'alignment', 'APdecoding'
        (from pparam.ANALYSIS_NAME_LIST[0])
    '''
    
    if custom_filename is None:
        analysis_filename = analysis_name
    else:
        analysis_filename = custom_filename


    analysis_dict_old = load_previous_analysis(analysis_filename)
    param_dict_default = pparam.default_param_dicts[analysis_name]
    
    #Substitute default parameters if missing
    for k,v in param_dict_default.items():
        if k not in param_dict_new:
            param_dict_new[k] = v
            
    #If no previous analysis was found, indicate that it must be repeated
    if analysis_dict_old is None:
        print('No previous analysis found, starting from scratch')
        return False, param_dict_new, {}
        
    #If previous analysis existss, check if it has the same parameters
    try:
        param_dict_old = analysis_dict_old['param_dict']
    except KeyError: #There's not parameter dict saved in the dictionary
        print('Warning: no previous parameter dict found in the analysis dictionary for "%s"'%analysis_name)
        have_parameters_been_used = False

    else:
        have_parameters_been_used = compare_parameter_dictionaries(param_dict_old, param_dict_new)
    
    return have_parameters_been_used, param_dict_new, analysis_dict_old


    
    
    
def save_figure(fig, fig_name):
    fig.savefig(pparam.FIGURES_PATH + fig_name + ".svg") 

def save_pca_data(mouse_list, session_list):
    ''' Convenience function to pre-compute PCA information '''
    
    ## Get PCA weights ##
    components_dict = {} #(mnum, snum) = (pca_dims, num_neurons)
    variance_explained_dict = {} #(mnum, snum) = (pca_dims, num_neurons)
    PCA_dict = {} #(mnum, snum) = (position, pca_data) with size(position) = timepoints and size(pca_data) = (pca_dims X timepoints)
    input_data_dict = {} #(mnum, snum) = (position, preprocessed_spikes) with size(position) = timepoints and size(preprocessed_spikes) = (num_neurons X timepoints)

    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = False
    pos_max = 1500
    
    for mnum in mouse_list:        
        
        for snum in session_list:
            
            
            #Get data
            data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                                    only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pos_max)
            pca_input_data, position, times = pf.get_data_from_datadict(data_dict, data_used)
            
            # place_cell_bool = pf.load_place_cell_boolean(mnum, snum, criteria='dombeck').astype(bool)
            # place_cell_idxs = np.where(place_cell_bool)[0]
            # nplace_cell_idxs = np.where(np.invert(place_cell_bool))[0]
            
            # pca_input_data = pca_input_data[nplace_cell_idxs]
            # print(mnum, snum, pca_input_data.shape)            
            # if pca_input_data.shape[0] < 3:
            #     continue
            # print(mnum, snum, pca_input_data.shape)

            
            input_data_dict[mnum, snum] = (position, pca_input_data)            
            num_neurons = pca_input_data.shape[0]
        
            #PCA
            pca = decomposition.PCA(n_components=num_neurons)
            pca.fit(pca_input_data.T)
            
            #PCA over time
            pca_data = pf.project_spikes_PCA(pca_input_data, pca_instance = pca, num_components = num_neurons)

            if eliminate_v_zeros == True:
                position, pca_data, _ = pf.compute_velocity_and_eliminate_zeros(position, pca_data, pos_max = pos_max)
            print(pca_data.shape)
            PCA_dict[mnum, snum] = (position, pca_data)
            
            #Components
            components = pca.components_
            components_dict[mnum, snum] = components
            
            #Variance explained
            variance_explained = pca.explained_variance_ratio_
            variance_explained_dict[mnum, snum] = variance_explained
            
    np.save(OUTPUT_PATH + "input_data_dict.npy", input_data_dict)
    np.save(OUTPUT_PATH + "pca_components_dict.npy", components_dict)
    np.save(OUTPUT_PATH + "variance_explained_dict.npy", variance_explained_dict)
    np.save(OUTPUT_PATH + "PCA_dict.npy", PCA_dict)




def perform_pca_on_multiple_mice_param_dict(pca_param_dict):
    
    have_parameters_been_used, pca_param_dict, pca_analysis_dict_previous = process_input_parameter_dict(pca_param_dict, pparam.ANALYSIS_NAME_LIST[0])
    print("Are pca parameters repeated?", have_parameters_been_used)
    if have_parameters_been_used == True:
        return pca_analysis_dict_previous
    
    mouse_list = pca_param_dict['mouse_list']
    session_list = pca_param_dict['session_list']

    #Preprocessing
    time_bin_size = pca_param_dict['time_bin_size']  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = pca_param_dict['distance_bin_size']  # mm, track is 1500mm, data is in mm
    gaussian_size = pca_param_dict['gaussian_size']  # Why not
    data_used = pca_param_dict['data_used']
    running = pca_param_dict['running']
    eliminate_v_zeros = pca_param_dict['eliminate_v_zeros']
    
    #PCA
    num_components = pca_param_dict['num_components']
    
    
    ## Store PCA weights ##
    components_dict = {} #(mnum, snum) = (pca_dims, num_neurons)
    variance_explained_dict = {} #(mnum, snum) = (pca_dims, num_neurons)
    PCA_dict = {} #(mnum, snum) = (position, pca_data) with size(position) = timepoints and size(pca_data) = (pca_dims X timepoints)
    input_data_dict = {} #(mnum, snum) = (position, preprocessed_spikes) with size(position) = timepoints and size(preprocessed_spikes) = (num_neurons X timepoints)

    for mnum in mouse_list:
        for snum in session_list:
            
            #Get data
            data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                                    only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
            pca_input_data, position, times = pf.get_data_from_datadict(data_dict, data_used)
            
    
            input_data_dict[mnum, snum] = (position, pca_input_data)            
            num_neurons = pca_input_data.shape[0]

            #PCA
            pca = decomposition.PCA(n_components=num_neurons)
            pca.fit(pca_input_data.T)
            
            #PCA over time
            pca_data = pf.project_spikes_PCA(pca_input_data, pca_instance = pca, num_components = num_components)
    
            if eliminate_v_zeros == True:
                position, pca_data, _ = pf.compute_velocity_and_eliminate_zeros(position, pca_data, pos_max = pparam.MAX_POS)
            
            PCA_dict[mnum, snum] = (position, pca_data)

            #Components
            components = pca.components_
            components_dict[mnum, snum] = components
            
            #Variance explained
            variance_explained = pca.explained_variance_ratio_
            variance_explained_dict[mnum, snum] = variance_explained

    
    PCA_analysis_dict = {
        'param_dict':pca_param_dict,
        'mouse_list':mouse_list,
        'session_list':session_list,
        'input_data_dict':input_data_dict,
        'PCA_dict':PCA_dict,
        'components_dict':components_dict,
        'variance_explained_dict':variance_explained_dict
        }
    
    np.save(OUTPUT_PATH + pparam.preprocessing_dict_name + ".npy", PCA_analysis_dict)

    
    return PCA_analysis_dict
    
    
def filter_repeated_sessions(mnum, session_list):
    ''' Given a mouse number and a list of session numbers, return the list with non-repeated sessions.
        If two sessions are repeats, keep only the first instance of the copy.
        Uses "SESSION_REPEATS" from parameter file
    '''
    
    session_list = list(np.copy(session_list))
    if mnum in SESSION_REPEATS:
        for repeated_pair in SESSION_REPEATS[mnum]:
            if repeated_pair[0] in session_list and repeated_pair[1] in session_list:
                session_list.remove(repeated_pair[1])
                
    return session_list
    
def pca_and_pos_from_dict_to_lists(pca_dict, mnum, session_list):
    ''' Given "pca_dict" from "save_pca_data", return a list of positions and pcas for each session'''
    pos_list = []
    pca_list = []
    for snum in session_list:
        pos, pca = pca_dict[mnum, snum]
        # pca, _, _ = pf.normalize_data(pca, axis=1)
        pos_list.append(pos)
        pca_list.append(pca)
    return pos_list, pca_list


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fig 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_fig1SI_A_traces(fig_num = None):
    ''' 
        Plots example Ca2+ traces
    '''
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    mnum = 6
    snum = 3
    
    
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 0  # Why not
    running = False
    eliminate_v_zeros = False
    
    ## Plot params ##
    fs = 15
    
    #Get raw data
    data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                            only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
    amplitudes = data_dict['amplitudes_binned_normalized']
    
    data = amplitudes

    fig = plt.figure(fig_num, figsize=(7,7)); fig_num += 1
    ax = plt.gca()
    neurons_to_plot = [0,2,3,4,6,7,8,9,10,11,15,20]
    
    for nidx, n in enumerate(neurons_to_plot):
        trace = data[n]
        trace = (trace - np.min(trace))/(np.max(trace) - np.min(trace))
        trace = 0.8*trace
        yy = trace + nidx
        xx = np.arange(len(yy))/(60*10)
        ax.plot(xx, yy, color='black')

    ax.set_xlabel('Time (min)', fontsize=fs+4)
    ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.set_ylabel('Axon #', fontsize=fs+4)
    ax.set_title('$\Delta$ F / F', fontsize=fs+4)
    
    
    fig_name = "fig1SI_traces"    
    save_figure(fig,fig_name)  
    
    
    return fig_num

def plot_fig1_C_firing_rates(fig_num = None):
    ''' 
        Plots example cell rates
    '''
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    mnum = 6
    snum = 2
    
    
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 15  # Why not
    running = False
    eliminate_v_zeros = False
    
    ## Plot params ##
    fs = 20
    bins = 30
    spine_width = 3
    tick_length = 10
    
    #Get raw data
    data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                            only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
    amplitudes = data_dict['amplitudes_binned_normalized']
    amplitudes, _, _ = pf.normalize_data(amplitudes)
    
    pos = data_dict['distance']
    pos, data, _ = pf.warping(pos, amplitudes, bins, max_pos = pparam.MAX_POS, warp_sampling_type = 'interpolation', warp_based_on = 'position', return_flattened = False)
    num_neurons, num_bins, num_trials = data.shape


    neurons_to_plot = [0,2,9]
    colors = ['royalblue', 'darkblue', 'mediumslateblue']
    
    
    
    
    ### Non-overlapped ###
    fig, axs = plt.subplots(len(neurons_to_plot), 1, num=fig_num, figsize=(7,6)); fig_num += 1
    # axs = plt.gca()
    for nidx, n in enumerate(neurons_to_plot):
        ax = axs.ravel()[nidx]
        rate = data[n]
        avg = np.mean(rate, axis=1)
        std = np.std(rate, axis=1)/np.sqrt(num_trials)

        xx = pos[:,0]
        ax.plot(xx, avg, color=colors[nidx], linewidth=5, label='Neuron %d'%n)
        ax.fill_between(xx, avg-std, avg+std, color='royalblue', alpha=0.3)

        if nidx == 0:
            neuron_list_string = [str(n) for n in neurons_to_plot]
            neuron_list_string = ','.join(neuron_list_string)
            ax.set_title('Animal %d / %s / Neurons %s'%(mnum, pparam.SESSION_NAMES[snum], neuron_list_string), fontsize=fs+4, pad=20)
            
        #Set y label on middle neuron
        if nidx == len(neurons_to_plot)//2:
            ax.set_ylabel('$\Delta F / F$', fontsize=fs+8)
            
        if nidx == len(neurons_to_plot)-1:
            #X axis for bottom plot
            ax.set_xlabel('Position (mm)', fontsize=fs+4)
            ax.spines.bottom.set_position(('outward', 10))
            ax.tick_params(axis='x', labelsize=fs+4)
            

        else:
            
            #Eliminate X axis for the rest
            ax.get_xaxis().set_visible(False)
            ax.spines[['bottom']].set_visible(False)
            
        
        #Set the Y axis ticks
        ymin, ymax = np.min(avg), np.max(avg)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([np.around(ymin, 1), np.around(ymax, 1)])
        ax.xaxis.set_tick_params(width=spine_width, length=tick_length)
        ax.yaxis.set_tick_params(width=spine_width, length=tick_length)
        ax.tick_params(axis='y', labelsize=fs+4)

        #Align all the X axis and move them apart
        ax.set_xlim([0,pparam.MAX_POS])
        ax.spines.left.set_position(('outward', 10))

        ax.spines[['right', 'top']].set_visible(False)
   
        #Spine width
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(spine_width)
    
    
    fig.tight_layout()
    
    fig_name = "fig1_firing_rates"    
    save_figure(fig,fig_name)  


    return fig_num

def plot_fig1_D_pcas(fig_num = None):
    ''' Plot PCA examples for figure 1 '''
    
    if fig_num is None:
        fig_num = plt.gcf().number + 1

    mlist = [6,6,6]
    slist = [2,3,8]
    angle_list = [75,50,-90]
    angle_azim_list = [-90, None, -100]
    
    rows = 2 #One row per session
    cols = len(slist) #raw data and trial averaged
    fig, axs = plt.subplots(rows, cols, subplot_kw={"projection": "3d"}, figsize=(9,7)) 
    fig_num = fig_num+1

    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    
    ## Plot params ##
    fs = 15
    pca_plot_bin_size = 30
    max_pos = pparam.MAX_POS

    

    
    for idx, snum in enumerate(slist):
        
        
        mnum = mlist[idx]
        angle = angle_list[idx]
        angle_azim = angle_azim_list[idx]
        
        
        #Get PCA
        data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                                only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
        pca_input_data, position, times = pf.get_data_from_datadict(data_dict, data_used)
        pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
        position, pca, _ = pf.warping(position, pca, 200, max_pos=pparam.MAX_POS, 
                                      warp_sampling_type = 'interpolation', warp_based_on = 'time', return_flattened=True)
        
        # print(pca.shape)
        # fig = plt.figure()
        ax = axs[0,idx]
        pf.plot_pca_with_position(pca, position, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, scatter=True, cbar=False, cbar_label='Position (mm)',
                                    alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                    ms = 10, lw=3)

        
        
        # fig_num += 1
        ax = axs[1, idx]
        position_unique, pca_average, pca_std = pf.compute_average_data_by_position(pca, position, position_bin_size=pca_plot_bin_size, max_pos=max_pos)
        pf.plot_pca_with_position(pca_average, position_unique, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, cbar=False, cbar_label='Position (mm)',
                                    alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                    scatter=False, ms = 250, lw=6)
        


    slist_names = [pparam.SESSION_NAMES[snum] for snum in slist]
    fig.suptitle('M%d %s, M%d %s, M%d %s' %(mlist[0], slist_names[0], mlist[1], slist_names[1], mlist[2], slist_names[2]),
                 fontsize=30)
    
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    
    fig_name = "fig1_pcas"    
    save_figure(fig,fig_name)  


    
        

    plt.figure(fig_num); fig_num += 1
    fig = plt.gcf()
    cbar = pf.add_distance_cbar(fig, pparam.PCA_CMAP, vmin = 0, vmax = pparam.MAX_POS, fs=fs, 
                                cbar_label = '', 
                                cbar_kwargs = {'fraction':0.055, 'pad':0.04, 'aspect':10})
    cbar.ax.tick_params(axis='y', labelsize=25)  

    
    # #Put everything on left
    cbar.ax.set_ylabel('Position (mm)', fontsize=30)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    
    fig.tight_layout()
    ax = plt.gca()
    ax.set_axis_off()

    fig_name = "fig1_pca_colorbar"    
    save_figure(fig,fig_name)  
    
    return fig_num


def plot_fig1_E_position_prediction_example(fig_num = None):
    
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    
    ########## PARAMETERS ###########
    mnum = 6
    snum = 4

    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = False
    

    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    ## Plot params ##
    fs = 20
    markersize=50
    
    #Get PCA
    data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                            only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
    pca_input_data, position, times = pf.get_data_from_datadict(data_dict, data_used)
    pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
    # position, pca, _ = pf.warping(position, pca, 200, max_pos=pparam.MAX_POS, 
    #                               warp_sampling_type = 'interpolation', warp_based_on = 'time', return_flattened=True)
    
    
    pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                            predictor_name=predictor_name, predictor_default=None, return_error=error_type)
    
    
    ###### OPTIONAL: ADD SHUFFLE ######
    np.random.seed(100)
    timepoints = len(position)
    idxs = list(np.arange(timepoints))
    shift = np.random.randint(-timepoints, timepoints)
    idxs_shifted = idxs[shift:] + idxs[:shift]
    idxs_shifted = np.random.choice(idxs_shifted, size=len(idxs_shifted), replace=False)
    position_shuffled = position[idxs_shifted]
    pca_shuffled = np.copy(pca)
    
    pos_pred_shuffle, error_shuffle, _ = pf.predict_position_CV(pca_shuffled, position_shuffled, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                            predictor_name=predictor_name, predictor_default=None, return_error=error_type)
    
    fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
    ax = plt.gca()
    
    timesteps = np.arange(len(position))

    for plot_idx, p in enumerate([position, pos_pred]):
        label = pparam.PREDICTION_LABELS[plot_idx]
        color = pparam.PREDICTION_COLORS[label]
        ax.scatter(timesteps, p, s=markersize, color=color, label=label)
        
    ax.scatter(timesteps, pos_pred_shuffle, s=markersize, color='red', label='shuffle')

        
    ax.set_xlabel('Timestep', fontsize=fs+4)
    # ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.set_ylabel('Position (mm)', fontsize=fs+4)
    ax.set_title('Animal %d, session %s, SSE=%.1f (cm)'%(mnum, pparam.SESSION_NAMES[snum], error), 
                 pad=20, fontsize=fs+4)
    ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
    fig.tight_layout()
    
    
    fig_name = "fig1_prediction_example"    
    save_figure(fig,fig_name)  
    return fig_num


def plot_fig1_F_prediction_across_sessions(fig_num = None):
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    
    ########## PARAMETERS ###########
    mlist = list(range(8))
    # mlist = [2,3,5,6]
    
    # slist = list(range(1,9))
    slist = list(range(9))

    # slist = [2,3,6]
    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mlist,
        'session_list':slist,
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'.9',
        }
    

    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    shuffle_reps = 10
    
    ## Plot params ##
    fs = 20
    
    num_mice = len(mlist)
    num_sessions = len(slist)
    error_array = np.ones((num_mice, num_sessions))*-1
    error_array_shuffle = np.ones((num_mice, num_sessions))*-1
    
    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)
    PCA_dict = PCA_analysis_dict['PCA_dict']
    
    for midx, mnum in enumerate(mlist):
        for sidx, snum in enumerate(slist):
            # #Get PCA
            position, pca = PCA_dict[mnum, snum]
            pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                    predictor_name=predictor_name, predictor_default=None, return_error=error_type)
            
            error_array[midx, sidx] = error
            
            if shuffle_reps != 0:
                timepoints = len(position)
                idxs = list(range(timepoints))
                error_shuffle_list = []
                for rep in range(shuffle_reps):
                    shift = np.random.randint(-timepoints, timepoints)
                    idxs_shifted = idxs[shift:] + idxs[:shift]
                    idxs_shifted = np.random.choice(idxs_shifted, size=len(idxs_shifted), replace=False)
                    position_shuffled = position[idxs_shifted]
                    pca_shuffled = np.copy(pca)
                    
                    # position_shuffled = np.copy(position)
                    # pca_input_data_shifted = pca_input_data[:, idxs_shifted]
                    
                    
                    pos_pred_shuffle, error_shuffle, _ = pf.predict_position_CV(pca_shuffled, position_shuffled, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                            predictor_name=predictor_name, predictor_default=None, return_error=error_type)
                    error_shuffle_list.append(error_shuffle)
                error_array_shuffle[midx, sidx] = np.average(error_shuffle_list)
            
    fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
    ax = plt.gca()
    
    mice_types = pparam.MOUSE_TYPE_LABELS
    for mtype_idx, mtype in enumerate(pparam.MOUSE_TYPE_LABELS):
        
        midx_list_type = [mlist.index(mnum) for mnum in mlist if mnum in pparam.MOUSE_TYPE_INDEXES[mice_types[mtype_idx]]]
        error_array_type = error_array[midx_list_type]
        num_mice_type, num_sessions_type = error_array_type.shape
        _, xx_scatter = np.mgrid[:num_mice_type, :num_sessions_type]
        xx_scatter = xx_scatter.ravel()
        error_scatter = error_array_type.ravel()
        color = pparam.MOUSE_TYPE_COLORS[mtype_idx]
        ax.scatter(xx_scatter, error_scatter, color=color)
        
        avg = np.average(error_array_type, axis=0)
        std = np.std(error_array_type, axis=0)/np.sqrt(error_array_type.shape[0])
        # print(error_array.shape)
        xx = np.arange(num_sessions)
        ax.plot(xx, avg, lw=3, color=color, label=mtype)
        ax.fill_between(xx, avg-std, avg+std, color=color, alpha=0.5)
        
                
    # #Significance testing (session by session)
    # for sidx, snum in enumerate(slist):
    #     id_vals = [error_array[mnum, sidx] for mnum in mlist if mnum in pparam.MOUSE_TYPE_INDEXES[mice_types[0]]]
    #     dd_vals = [error_array[mnum, sidx] for mnum in mlist if mnum in pparam.MOUSE_TYPE_INDEXES[mice_types[1]]]
        
    #     tstat, pval = scipy.stats.ttest_ind(id_vals, dd_vals, equal_var=True, permutations=None, alternative='two-sided')
    #     pval_label = pf.get_significance_label(np.abs(pval), [0.05], asterisk = True, ns=False)
    #     print(pval, pval_label)

    #     if pval_label != 'ns':
    #         max_val = np.maximum(np.max(id_vals), np.max(dd_vals))
    #         ax.text(sidx-0.1, max_val + 3, pval_label, fontsize=fs+10, style='normal', color='black')


    
    
    if shuffle_reps != 0:
        avg = np.average(error_array_shuffle, axis=0)
        std = np.std(error_array_shuffle, axis=0)/np.sqrt(error_array_shuffle.shape[0])
        xx = np.arange(num_sessions)
        ax.plot(xx, avg, lw=3, color=pparam.SHUFFLE_DEFAULT_COLOR, label='Shuffle')
        ax.fill_between(xx, avg-std, avg+std, color=pparam.SHUFFLE_DEFAULT_COLOR, alpha=0.5)
        
        
        #Significance testing (overall)
        # id_vals = error_array[]
        midx_list_type = [[mlist.index(mnum) for mnum in mlist if mnum in pparam.MOUSE_TYPE_INDEXES[mtype]] for mtype in pparam.MOUSE_TYPE_LABELS]
        id_vals = error_array[midx_list_type[0]].ravel()
        dd_vals = error_array[midx_list_type[1]].ravel()
        shuffle_vals = error_array_shuffle.ravel()
        
        # tstat, pval_id_shuffle = scipy.stats.ttest_ind(id_vals, shuffle_vals, equal_var=False, permutations=None, alternative='two-sided')
        # tstat, pval_dd_shuffle = scipy.stats.ttest_ind(dd_vals, shuffle_vals, equal_var=False, permutations=None, alternative='two-sided')
        # tstat, pval_id_dd = scipy.stats.ttest_ind(id_vals, dd_vals, equal_var=False, permutations=None, alternative='two-sided')
        
        tstat, pval_id_shuffle = scipy.stats.mannwhitneyu(id_vals, shuffle_vals, use_continuity=False, alternative='two-sided')
        tstat, pval_dd_shuffle = scipy.stats.mannwhitneyu(dd_vals, shuffle_vals, use_continuity=False, alternative='two-sided')
        tstat, pval_id_dd = scipy.stats.mannwhitneyu(id_vals, dd_vals, use_continuity=False, alternative='two-sided')

        print("Pval id against shuffle", pval_id_shuffle)
        print("Pval dd against shuffle", pval_dd_shuffle)
        print("Pval id against dd", pval_id_dd)       
        
    #X axis
    ax.set_xlabel('Session', fontsize=fs+4)
    # ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
    ax.set_xticks(np.arange(num_sessions), [pparam.SESSION_NAMES[snum] for snum in slist])
    ax.tick_params(axis='x', labelsize=fs)
    
    #Y axis
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_ylabel('Error (cm)', fontsize=fs+4)

    #Both axis
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
                 
    #Legend
    ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
    
    #Figure params
    fig.tight_layout()
    
    
    fig_name = "fig1_prediction_across_sessions"    
    save_figure(fig,fig_name)  

        

    
    
    return fig_num


def plot_fig1SI_B_C_variance_explained(fig_num = None):
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    
    mlist = np.arange(8)
    # mlist = np.arange(4)  
    num_mice = len(mlist)
    
    slist = np.arange(9)
    # slist = np.arange(3)
    num_sessions = len(slist)
    
    
    #PCA params
    variance_to_explain = 0.8
    
    fig_num = 1
    
    variance_explained_dict = np.load(OUTPUT_PATH +"variance_explained_dict.npy", allow_pickle=True)[()]
    PCA_dict = np.load(OUTPUT_PATH +"PCA_dict.npy", allow_pickle=True)[()]
    
    dimensions_array = np.zeros((num_mice, num_sessions))
    dimensions_array_abs = np.zeros((num_mice, num_sessions))
    
    dimension_bins = np.arange(0,105,5)
    var_explained_by_dimension_bin = [[] for i in range(len(dimension_bins))]
    cumvar_explained_by_dimension_bin = [[] for i in range(len(dimension_bins))]

    for midx, mnum in enumerate(mlist):
        for sidx, snum in enumerate(slist):
            
           

            pos, pca_data = PCA_dict[mnum, snum]
            num_neurons = pca_data.shape[0]
            variance_explained = variance_explained_dict[mnum, snum]            
            variance_explained_cum = np.cumsum(variance_explained)                

            dimensions_for_x = pf.dimensions_to_explain_variance(variance_explained, variance_to_explain)
            
            dimensions_array[midx, sidx] = int(np.around(100 * dimensions_for_x/float(num_neurons)))
            dimensions_array_abs[midx, sidx] = dimensions_for_x
            
            dimensions_percentage = np.array(100 * np.arange(num_neurons)/num_neurons, dtype=int)
            for d_idx,d in enumerate(dimensions_percentage):
                dim_bin = np.argmax(d <= dimension_bins)
                var_explained_by_dimension_bin[dim_bin].append(variance_explained[d_idx])
                cumvar_explained_by_dimension_bin[dim_bin].append(variance_explained_cum[d_idx])
                
                
    fig = plt.figure(fig_num); fig_num += 1
    ax = plt.gca()
    fs = 18
    color1 = 'royalblue'
    color2 = 'indianred'

    #Var explained ratio
    color1 = 'royalblue'
    var_explained_avg = np.array([np.average(varlist) for varlist in var_explained_by_dimension_bin])
    var_explained_std = np.array([np.std(varlist) for varlist in var_explained_by_dimension_bin])
    # var_explained_std = [np.std(varlist)/np.sqrt(len(varlist)) for varlist in var_explained_by_dimension_bin]

    ax.plot(dimension_bins, var_explained_avg, color=color1, lw=3)
    ax.fill_between(dimension_bins, var_explained_avg - var_explained_std, var_explained_avg + var_explained_std, 
                      color=color1, alpha=0.5)
    ax.set_xlabel('Dimensions (%)', fontsize=fs)
    ax.set_ylabel('Var. explained (ratio)', color=color1, fontsize=fs)
    ax.tick_params(axis='x', labelsize=fs)                
    ax.tick_params(axis='y', labelcolor=color1, labelsize=fs)
    
    
    #Cumulative variance explained

    ax2 = ax.twinx()
    cumvar_explained_avg = np.array([np.average(varlist) for varlist in cumvar_explained_by_dimension_bin])
    cumvar_explained_std = np.array([np.std(varlist) for varlist in cumvar_explained_by_dimension_bin])
    # var_explained_std = [np.std(varlist)/np.sqrt(len(varlist)) for varlist in var_explained_by_dimension_bin]
    
    ax2.plot(dimension_bins, cumvar_explained_avg, color=color2, lw=3)
    ax2.fill_between(dimension_bins, cumvar_explained_avg - cumvar_explained_std, cumvar_explained_avg + cumvar_explained_std, 
                      color=color2, alpha=0.5)
    ax2.set_ylabel('Cumulative var. explained', color=color2, fontsize=fs)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=fs)


    xlims = ax.get_xlim()
    ylims = ax2.get_ylim()
    #Lines at cumulative variance explained
    avg_var_explained_idx = np.argmax(variance_to_explain <= cumvar_explained_avg)
    dim_for_avg_var_explained = dimension_bins[avg_var_explained_idx]
    yy = np.linspace(ax.get_ylim()[0], 0.8)
    ax2.plot([dim_for_avg_var_explained]*len(yy), yy, '--', alpha=1, color='gray')
    xx = np.linspace(dim_for_avg_var_explained, xlims[1])
    ax2.plot(xx, [0.8]*len(xx), '--', alpha=1, color='gray')
    
    ax.set_xlim(xlims)
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    
    
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax2.spines[['left', 'top']].set_visible(False)
    ax2.spines['right'].set_linewidth(3)
    fig.tight_layout()
    save_figure(fig, 'fig1SI_B_variance_explained')
            
    

    print(r'Average dimension %% to explain %d of the variance: %.1f +- %.1f, median: %d'%(100*variance_to_explain, np.average(dimensions_array), np.std(dimensions_array), np.median(dimensions_array)))
    print(r'ID Average dimension %% to explain %d of the variance: %.1f +- %.1f, median: %d'%(100*variance_to_explain, np.average(dimensions_array[:4]), np.std(dimensions_array[:4]), np.median(dimensions_array[:4])))
    print(r'Average dimension %% to explain %d of the variance: %.1f +- %.1f, median: %d'%(100*variance_to_explain, np.average(dimensions_array[4:]), np.std(dimensions_array[4:]), np.median(dimensions_array[4:])))



    # Plot
    figsize= (7, 6)
    fs = 15
    condition_color = ['forestgreen', 'khaki']
    fig, axs = plt.subplots(2,1, figsize=figsize); fig_num += 1
    
    for sidx, snum in enumerate(slist):
        
        ax = axs[0]
        dim_list = dimensions_array[:4, sidx].ravel()
        dim_mean = np.mean(dim_list)
        dim_std = np.std(dim_list)
        ax.bar([SESSION_NAMES[sidx]], dim_mean, yerr=dim_std, width=.7, zorder=1, color=condition_color[0])
        ax.scatter([sidx] * len(dim_list), dim_list, s=12, zorder=2, color='dimgray')
        
        ax = axs[1]
        dim_list = dimensions_array[4:8, sidx].ravel()
        dim_mean = np.mean(dim_list)
        dim_std = np.std(dim_list)
        ax.bar([SESSION_NAMES[sidx]], dim_mean, yerr=dim_std, width=.7, zorder=1, color=condition_color[1])
        ax.scatter([sidx] * len(dim_list), dim_list, s=12, zorder=2, color='dimgray')        
        
        
    fig.suptitle('Dimensions to explain %d%% of the variance' %int(variance_to_explain*100), fontsize=fs)
    # fig.suptitle('Dimensions to explain % of the variance', fontsize=fs)

    axs[0].set_title('%s'%pparam.MOUSE_TYPE_LABELS[0], fontsize=fs+3)
    axs[0].set_ylabel('Dimensions (%)', fontsize=fs)
    axs[0].tick_params(axis='x', labelsize=fs)                
    axs[0].tick_params(axis='y', labelsize=fs)
    axs[0].spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(3)
        
    axs[1].set_title('%s'%pparam.MOUSE_TYPE_LABELS[0], fontsize=fs+3)
    axs[1].set_ylabel('Dimensions (%)', fontsize=fs)
    axs[1].tick_params(axis='x', labelsize=fs)                
    axs[1].tick_params(axis='y', labelsize=fs)
    axs[1].spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        axs[1].spines[axis].set_linewidth(3)
        
    fig.tight_layout()
    save_figure(fig, 'fig1SI_C_prop_dimensions_to_explain_%d_of_the_variance'%int(variance_to_explain*100))

    figsize= (7, 6)
    fs = 15
    condition_color = ['forestgreen', 'khaki']
    fig, axs = plt.subplots(2,1, figsize=figsize); fig_num += 1
    
    for sidx, snum in enumerate(slist):
        
        ax = axs[0]
        dim_list = dimensions_array_abs[:4, sidx].ravel()
        dim_mean = np.mean(dim_list)
        dim_std = np.std(dim_list)
        ax.bar([SESSION_NAMES[sidx]], dim_mean, yerr=dim_std, width=.7, zorder=1, color=condition_color[0])
        ax.scatter([sidx] * len(dim_list), dim_list, s=12, zorder=2, color='dimgray')
        
        ax = axs[1]
        dim_list = dimensions_array_abs[4:8, sidx].ravel()
        dim_mean = np.mean(dim_list)
        dim_std = np.std(dim_list)
        ax.bar([SESSION_NAMES[sidx]], dim_mean, yerr=dim_std, width=.7, zorder=1, color=condition_color[1])
        ax.scatter([sidx] * len(dim_list), dim_list, s=12, zorder=2, color='dimgray')        
        
        
    fig.suptitle('Dimensions to explain %d%% of the variance' %int(variance_to_explain*100), fontsize=fs)
    # fig.suptitle('Dimensions to explain % of the variance', fontsize=fs)

    axs[0].set_title('%s'%pparam.MOUSE_TYPE_LABELS[0], fontsize=fs+3)
    axs[0].set_ylabel('Dimensions', fontsize=fs)
    axs[0].tick_params(axis='x', labelsize=fs)                
    axs[0].tick_params(axis='y', labelsize=fs)
    axs[0].spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(3)
        
    axs[1].set_title('%s'%pparam.MOUSE_TYPE_LABELS[0], fontsize=fs+3)
    axs[1].set_ylabel('Dimensions', fontsize=fs)
    axs[1].tick_params(axis='x', labelsize=fs)                
    axs[1].tick_params(axis='y', labelsize=fs)
    axs[1].spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        axs[1].spines[axis].set_linewidth(3)
        
    fig.tight_layout()  
    save_figure(fig, 'fig1SI_D_raw_dimensions_to_explain_%d_of_the_variance'%int(variance_to_explain*100))

    

def plot_fig1SI_D_prediction_vs_dimensionality(fig_num = None):
    
    if fig_num is None:
        fig_num = plt.gcf().number + 1
    
    ########## PARAMETERS ###########
    mlist = list(range(8))
    # mlist = [2,3,5,6]
    
    # slist = list(range(1,9))
    slist = list(range(9))

    # slist = [2,3,6]
    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mlist,
        'session_list':slist,
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':None, #will be varied
        }
    

    components_list = [0.1, 0.15, 0.25, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    # components_list = [0.1, 0.2, 0.8]

    components_list = ["%.3f"%ncomp for ncomp in components_list]
    components_list_len = len(components_list)
    
    print(components_list)
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    shuffle_reps = 10
    
    ## Plot params ##
    fs = 20
    
    
    error_by_dim = {(mtype, ncomp):[] for mtype in pparam.MOUSE_TYPE_LABELS for ncomp in components_list} # {("normal" or "shuffle", ncomp) : [list of errors]}
    error_by_dim_shuffle = {(mtype, ncomp):[] for mtype in pparam.MOUSE_TYPE_LABELS for ncomp in components_list} # {("normal" or "shuffle", ncomp) : [list of errors]}
    

    # error_by_dim_shuffle = {(mtype, ncomp):[] for mtype in pparam.MOUSE_TYPE_LABELS for ncomp in components_list} # {("normal" or "shuffle", ncomp) : [list of errors]}

    
    for ncomp_idx, ncomp in enumerate(components_list):
        
        preprocessing_param_dict['num_components'] = ncomp
        
        
        PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)
        PCA_dict = PCA_analysis_dict['PCA_dict']
        
        for midx, mnum in enumerate(mlist):
            mtype = pparam.MOUSE_TYPE_LABEL_BY_MOUSE[mnum]
            for sidx, snum in enumerate(slist):
                # #Get PCA
                position, pca = PCA_dict[mnum, snum]
                pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                        predictor_name=predictor_name, predictor_default=None, return_error=error_type)
                
                error_by_dim[mtype, ncomp].append(error)
                
                if shuffle_reps != 0:
                    timepoints = len(position)
                    idxs = list(range(timepoints))
                    error_shuffle_list = []
                    for rep in range(shuffle_reps):
                        shift = np.random.randint(-timepoints, timepoints)
                        idxs_shifted = idxs[shift:] + idxs[:shift]
                        idxs_shifted = np.random.choice(idxs_shifted, size=len(idxs_shifted), replace=False)
                        position_shuffled = position[idxs_shifted]
                        pca_shuffled = np.copy(pca)
                        
                        
                        
                        pos_pred_shuffle, error_shuffle, _ = pf.predict_position_CV(pca_shuffled, position_shuffled, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                                predictor_name=predictor_name, predictor_default=None, return_error=error_type)
                        error_shuffle_list.append(error_shuffle)
                    error_shuffle_avg = np.average(error_shuffle_list)
                    error_by_dim_shuffle[mtype, ncomp].append(error_shuffle_avg)

    #Compute averages to plot
    error_by_dim_avg = np.zeros((len(pparam.MOUSE_TYPE_LABELS), components_list_len))
    error_by_dim_std = np.zeros((len(pparam.MOUSE_TYPE_LABELS), components_list_len))
    error_by_dim_shuffle_avg = np.zeros(components_list_len)
    error_by_dim_shuffle_std = np.zeros(components_list_len)
    
    for ncomp_idx, ncomp in enumerate(components_list):

        for mtype_idx, mtype in enumerate(pparam.MOUSE_TYPE_LABELS):
            errors = error_by_dim[mtype, ncomp]
            avg = np.average(errors)
            std = np.std(errors)/np.sqrt(len(errors))
            error_by_dim_avg[mtype_idx, ncomp_idx] = avg
            error_by_dim_std[mtype_idx, ncomp_idx] = std
            
        if shuffle_reps != 0:
            errors = error_by_dim_shuffle[pparam.MOUSE_TYPE_LABELS[0], ncomp] + error_by_dim_shuffle[pparam.MOUSE_TYPE_LABELS[1], ncomp]
            avg = np.average(errors)
            std = np.std(errors)/np.sqrt(len(errors))
            error_by_dim_shuffle_avg[ncomp_idx] = avg
            error_by_dim_shuffle_std[ncomp_idx] = std

    ## Plot results ##
    fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
    ax = plt.gca()
    
    
    # xx = np.array(components_list).astype(float) * 100
    # ax.set_xscale('log')

    xx = np.arange(len(components_list))
    for mtype_idx, mtype in enumerate(pparam.MOUSE_TYPE_LABELS):
        avgs = error_by_dim_avg[mtype_idx]
        stds = error_by_dim_std[mtype_idx]
        
        color = pparam.MOUSE_TYPE_COLORS[mtype_idx]

        ax.plot(xx, avgs, lw=3, color=color, label=mtype)
        ax.fill_between(xx, avgs-stds, avgs+stds, color=color, alpha=0.5)
        
    if shuffle_reps != 0:

        ax.plot(xx, error_by_dim_shuffle_avg, lw=3, color=pparam.SHUFFLE_DEFAULT_COLOR, label='Shuffle')
        ax.fill_between(xx, error_by_dim_shuffle_avg-error_by_dim_shuffle_std, error_by_dim_shuffle_avg+error_by_dim_shuffle_std, 
                        color=pparam.SHUFFLE_DEFAULT_COLOR, alpha=0.5)
        
        
        
    #X axis
    ax.set_xlabel('Variance explained (%)', fontsize=fs)

    ax.set_xticks([], []) #Delete log ticks
    xlabels = [int(100*float(ncomp)) for ncomp in components_list]
    ax.set_xticks(xx, xlabels) #Add the ticks I want
    ax.tick_params(axis='x', labelsize=fs)
    ax.minorticks_off()
    
    #Y axis
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_ylabel('Error (cm)', fontsize=fs)

    #Both axis
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
                 
    #Legend
    ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
    
    #Figure params
    fig.tight_layout()
    
    
    fig_name = "fig1SI_prediction_by_dimensionality"    
    save_figure(fig,fig_name) 
        
    return
                

 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fig 2 CCA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  


def perform_mCCA_on_pca_dict_param_dict(pca_analysis_dict, cca_param_dict, force_recalculation=True):
    '''Given a PCA analysis dictionary (from "save_pca_data"), perform mCCA on it
    Input:
        PCA_analysis_dict: output of "save_pca_data"
        sessions_to_align: either 'all' or a list of sessions that need to be aligned
        pca_dim: int or percentage. Number of PCA dimensions to keep.
        return_warped_data: bool, if True, returns data with task-normalized bins
        return_trimmed_data: bool, if True, returns dataset trimmed so each session has the same number of laps. 
                        If False, some sessions may include laps not used in the alignment process
        verbose: bool, if True says which sessions it's aligning
        plot: bool, if True plots the result
        
    Returns:
        aligned_data_dict: dictionary with all relevant CCA data
        
    
    '''
    
    #First check if the cca parameters are the same
    analysis_name = pparam.ANALYSIS_NAME_LIST[1]
    if cca_param_dict['shuffle'] == True:
        custom_filename = pparam.default_param_dicts_names[analysis_name] + "_shuffle"
    else:
        custom_filename = analysis_name
    are_cca_params_equal, cca_param_dict, aligned_data_dict_previous = process_input_parameter_dict(cca_param_dict, analysis_name, custom_filename)
    
    #Then check if the pca parameters were the same as well
    preprocessing_param_dict = pca_analysis_dict['param_dict']
    try:
        preprocessing_param_dict_old = aligned_data_dict_previous[pparam.ANALYSIS_NAME_LIST[0] + '_param_dict']
        are_preprocessing_params_equal = compare_parameter_dictionaries(preprocessing_param_dict, preprocessing_param_dict_old)
    except KeyError:
        #No properly parsed previous analysis found
        are_preprocessing_params_equal = False
    
    # print('Are CCA parameters repeated?', are_preprocessing_params_equal, 
    #       'And PCA?', are_cca_params_equal)

    #If both the PCA and CCA analysis were the same, return previous result
    if force_recalculation == False and are_cca_params_equal == True and are_preprocessing_params_equal == True and cca_param_dict['shuffle'] == False: #Always recalculate for shuffle!
        return aligned_data_dict_previous
    
    #Path to save results
    analysis_dict_filename = get_analysis_dict_filename(custom_filename)

    
    #Warping parameters
    warping_bins = 150
    warp_based_on = 'position'
    # warp_based_on = 'position'

    #Prediction parameters
    error_type = 'sse'
    n_splits = 5
    predictor_name = 'Wiener'
    
    ## CCA params ##
    CCA_dim = cca_param_dict['CCA_dim']
    return_warped_data = cca_param_dict['return_warped_data']
    return_trimmed_data = cca_param_dict['return_trimmed_data']
    sessions_to_align = cca_param_dict['sessions_to_align']
    shuffle = cca_param_dict['shuffle']
    
    #Data parameters
    max_pos = pparam.MAX_POS
    mouse_list = pca_analysis_dict['mouse_list']
    pca_data_dict = pca_analysis_dict['PCA_dict']
    variance_explained_dict = pca_analysis_dict['variance_explained_dict']
    
    if sessions_to_align == 'all':
        session_list_original = pca_analysis_dict['session_list']
    else:
        session_list_original = sessions_to_align
        
    aligned_data_dict = {}

    for midx, mnum in enumerate(mouse_list):
        
        session_list = filter_repeated_sessions(mnum, session_list_original)
        pos_list, pca_list = pca_and_pos_from_dict_to_lists(pca_data_dict, mnum, session_list)
        M = len(session_list)
        
        #Set PCA dimension
        pca_list = mCCA_funs.set_dimension_of_pca_list(pca_list, CCA_dim, variance_explained_list = [variance_explained_dict[mnum, snum] for snum in session_list])
        #Perform mCCA
        pos_list_aligned, pca_dict_aligned, mCCA = mCCA_funs.perform_warped_mCCA(pos_list, pca_list, max_pos, warping_bins, warp_based_on, return_warped_data, return_trimmed_data, shuffle)

        #Normalize PCA after alignment changes
        pca_dict_aligned = mCCA_funs.normalize_pca_dict_aligned(pca_dict_aligned, mCCA)
        
        #Find space with best alignment
        best_space = mCCA_funs.return_best_mCCA_space(pos_list_aligned, pca_dict_aligned, max_pos=1500, plot=False, verbose=False)
        pca_list_aligned = pca_dict_aligned[best_space]
                    
        pca_list = [pca_dict_aligned[m][m] for m in range(M)]
        unaligned_error_array, aligned_error_array = mCCA_funs.get_cross_prediction_errors(pos_list_aligned, pca_list, pos_list_aligned, pca_dict_aligned, max_pos, n_splits, error_type, predictor_name)
        
        aligned_data_dict[mnum, 'pos'] = pos_list_aligned
        aligned_data_dict[mnum, 'pca'] = pca_list_aligned
        aligned_data_dict[mnum, 'pca_unaligned'] = pca_list
        aligned_data_dict[mnum, 'pca_dict_aligned'] = pca_dict_aligned
        
        aligned_data_dict[mnum, 'best_space'] = best_space
        aligned_data_dict[mnum, 'session_list'] = session_list
        aligned_data_dict[mnum, 'unaligned_error_array'] = unaligned_error_array
        aligned_data_dict[mnum, 'aligned_error_array'] = aligned_error_array  
        aligned_data_dict[mnum, 'mCCA_instance'] = mCCA
        
    aligned_data_dict['mouse_list'] = mouse_list
    aligned_data_dict['num_bins'] = warping_bins
    aligned_data_dict['param_dict'] = cca_param_dict
    aligned_data_dict[pparam.ANALYSIS_NAME_LIST[0] + '_param_dict'] = pca_analysis_dict['param_dict']
        

    # np.save(OUTPUT_PATH + pparam.cca_dict_name + ".npy", aligned_data_dict)
    np.save(analysis_dict_filename, aligned_data_dict)
            
    return aligned_data_dict 

def perform_mCCA_on_pca_dict(
        pca_analysis_dict,
        sessions_to_align = 'all', #[0,1,2,3,4,5,6,7,8] [0,1,2,3,4,5,7,8]
        pca_dim = 12, #6, '85%'
        return_warped_data = False,
        return_trimmed_data = False,
        plot=True,
        shuffle=False
        ):
    ''' 
    
    Given a PCA analysis dictionary (from "save_pca_data"), perform mCCA on it
    Input:
        PCA_analysis_dict: output of "save_pca_data"
        sessions_to_align: either 'all' or a list of sessions that need to be aligned
        pca_dim: int or percentage. Number of PCA dimensions to keep.
        return_warped_data: bool, if True, returns data with task-normalized bins
        return_trimmed_data: bool, if True, returns dataset trimmed so each session has the same number of laps. 
                        If False, some sessions may include laps not used in the alignment process
        verbose: bool, if True says which sessions it's aligning
        plot: bool, if True plots the result
    
    Returns:
        aligned_data_dict: dictionary with all relevant CCA data
        
    
    '''
    
    #Warping parameters
    warping_bins = 150
    warp_based_on = 'position'
    
    #Prediction parameters
    error_type = 'sse'
    n_splits = 5
    predictor_name = 'Wiener'
    
    #Data parameters
    max_pos = pparam.MAX_POS
    mouse_list = pca_analysis_dict['mouse_list']
    pca_data_dict = pca_analysis_dict['PCA_dict']
    variance_explained_dict = pca_analysis_dict['variance_explained_dict']
    
    if sessions_to_align == 'all':
        session_list_original = pca_analysis_dict['session_list']
    else:
        session_list_original = sessions_to_align
    

        
    aligned_data_dict = {}
    for midx, mnum in enumerate(mouse_list):
        
        session_list = filter_repeated_sessions(mnum, session_list_original)
        pos_list, pca_list = pca_and_pos_from_dict_to_lists(pca_data_dict, mnum, session_list)
        M = len(session_list)
                        
        #Set PCA dimension
        pca_list = mCCA_funs.set_dimension_of_pca_list(pca_list, pca_dim, variance_explained_list = [variance_explained_dict[mnum, snum] for snum in session_list])

        #Perform mCCA
        pos_list_aligned, pca_dict_aligned, mCCA = mCCA_funs.perform_warped_mCCA(pos_list, pca_list, max_pos, warping_bins, warp_based_on, return_warped_data, return_trimmed_data, shuffle)

        #Normalize PCA after alignment changes
        pca_dict_aligned = mCCA_funs.normalize_pca_dict_aligned(pca_dict_aligned, mCCA)
        
        #Find space with best alignment
        best_space = mCCA_funs.return_best_mCCA_space(pos_list_aligned, pca_dict_aligned, max_pos=1500, plot=False, verbose=False)

        pca_list_aligned = pca_dict_aligned[best_space]
                    
        pca_list = [pca_dict_aligned[m][m] for m in range(M)]
        unaligned_error_array, aligned_error_array = mCCA_funs.get_cross_prediction_errors(pos_list_aligned, pca_list, pos_list_aligned, pca_dict_aligned, max_pos, n_splits, error_type, predictor_name)
        
        aligned_data_dict[mnum, 'pos'] = pos_list_aligned
        aligned_data_dict[mnum, 'pca'] = pca_list_aligned
        aligned_data_dict[mnum, 'pca_unaligned'] = pca_list
        aligned_data_dict[mnum, 'pca_dict_aligned'] = pca_dict_aligned
        
        aligned_data_dict[mnum, 'best_space'] = best_space
        aligned_data_dict[mnum, 'session_list'] = session_list
        aligned_data_dict[mnum, 'unaligned_error_array'] = unaligned_error_array
        aligned_data_dict[mnum, 'aligned_error_array'] = aligned_error_array        
        aligned_data_dict['mouse_list'] = mouse_list
        aligned_data_dict['num_bins'] = warping_bins
            
    return aligned_data_dict    
            
     
def get_cca_filename_full_path(CCA_dim, return_trimmed_data, return_warped_data):
    cca_filename = 'aligned_data_dict'
    cca_filename += '_%s'%str(CCA_dim)
    cca_filename += ['_untrimmed','_trimmed'][return_trimmed_data]
    cca_filename += ['_unwarped','_warped'][return_warped_data]
    cca_filename_full_path = OUTPUT_PATH + cca_filename + '.npy'
    return cca_filename_full_path





def fig2_CCA_example():
    ''' Plots the result of alignment on a single '''
    
    mnum = 6
    snum1 = 1
    snum2 = 3
    mouse_list = [mnum]
    session_list = range(9)
    
    figsize = (8,6)

    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':session_list,
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':False,
        'num_components':'all',
        }

    
    ## CCA params ##    
    cca_param_dict = {
        'CCA_dim':12,
        'return_warped_data':False,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ## Plot params ##
    fig_num = 1
    fs = 17
    pca_plot_bins = 50
    max_pos = pparam.MAX_POS
    predictor_name_example_plot = 'XGBoost'

    #PCA
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)

    #CCA
    aligned_data_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)

    
    rows=3
    cols=2

    fig = plt.figure(fig_num, figsize=figsize); fig_num += 1
    
    
    pca_list_unaligned = aligned_data_dict[mnum, 'pca_unaligned']
    pca_list_aligned = aligned_data_dict[mnum, 'pca']          
    pos_list = aligned_data_dict[mnum, 'pos']
    
    #Force them to be the same size
    min_size = np.min([np.size(pos_list[snum]) for snum in [snum1, snum2]])
    pos_list = [pos[:min_size] for pos in pos_list]
    pca_list_unaligned = [pca[:,:min_size] for pca in pca_list_unaligned]
    pca_list_aligned = [pca[:,:min_size] for pca in pca_list_aligned]
    
    subplot_counter = 0
    for row in range(rows):
        
        if row==0:
            snums = [snum1]
        elif row ==1:
            snums = [snum2]
        elif row == 2:
            snums = [snum1, snum2]


        
        #Plot position prediction
        subplot_counter += 1
        ax = fig.add_subplot(rows, cols, subplot_counter, aspect=0.4)

        snum = snums[-1]

        pos = pos_list[snum]
        if row != 2:
            pca = pca_list_unaligned[snum]
        else:
            pca = pca_list_aligned[snum]



        if row == 0:
            pos_pred, error, _ = pf.predict_position_CV(pca, pos, n_splits=5, pmax=max_pos, predictor_name=predictor_name_example_plot)
            _,_, predictor = pf.predict_position_CV(pca, pos, n_splits=0, pmax=max_pos, predictor_name=predictor_name_example_plot)

        elif row == 1:
            pos_pred, error = pf.predict_position_from_predictor_object(pca, pos, predictor, periodic=True, pmax=max_pos)

        elif row == 2:
            _,_, predictor = pf.predict_position_CV(pca_list_aligned[snum1], pos_list[snum1], n_splits=0, pmax=max_pos, predictor_name=predictor_name_example_plot)
            pos_pred, error = pf.predict_position_from_predictor_object(pca_list_aligned[snum2],  pos_list[snum2], predictor, periodic=True, pmax=max_pos)

        time_steps = np.arange(pos.size)
        ax.scatter(time_steps, pos, s=4, alpha=0.6, lw=2, color=pparam.PREDICTION_COLORS[pparam.PREDICTION_LABELS[0]])
        ax.scatter(time_steps, pos_pred, alpha=1, s=4,lw=2, color=pparam.PREDICTION_COLORS[pparam.PREDICTION_LABELS[1]])
    
        
        
            
        #Ticks
        tickw, tickl = 4, 7
        ax.set_yticks([0, 750, 1500])
        ax.set_xticks(np.arange(0, np.max(time_steps), 400).astype(int))        
        ax.tick_params(axis='x', labelsize=fs, width=tickw, length=tickl)
        ax.tick_params(axis='y', labelsize=fs, width=tickw, length=tickl)
        ax.set_ylim([-0.1*pparam.MAX_POS, ax.get_ylim()[-1]])
        
        if row == 2:
            ax.set_xlabel('Time step', fontsize=fs)     
        
        # ax[1].tick_params(width=3, length=6)
        
        #Axis
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(3)
        
        
        if row == 0:

            ax.plot([0],[0], label=pparam.PREDICTION_LABELS[0], color=pparam.PREDICTION_COLORS[pparam.PREDICTION_LABELS[0]]) 
            ax.plot([0],[0], label=pparam.PREDICTION_LABELS[1], color=pparam.PREDICTION_COLORS[pparam.PREDICTION_LABELS[1]])
            ax.legend(fontsize=fs-5, loc='upper left', frameon=False)

        
        if row == 1:
            ax.set_ylabel('Position (mm)', fontsize=fs)

        
        
        
        #Plot PCA        
        subplot_counter += 1
        ax = fig.add_subplot(rows, cols, subplot_counter, projection='3d')
        for snum_idx, snum in enumerate(snums):
            pos = pos_list[snum]
            if row != 2:
                pca = pca_list_unaligned[snum]
            else:
                pca = pca_list_aligned[snum]
            # cbar = [None, True][row==0]
            cbar = None

            
            pos_bins, pca_avg, _ = pf.compute_average_data_by_position(pca, pos, position_bin_size=pca_plot_bins)
            pf.plot_pca_with_position(pca_avg, pos_bins, ax=ax, scatter=False, cbar=cbar) 
            

    fig.tight_layout()
    
    fig_name = "fig2_cca_example"    
    save_figure(fig,fig_name) 

    


def fig2_CCA_aligned_pcas(shuffle=False):
    
    mouse_list = np.arange(8)
    session_list = np.arange(9)
    mnum1 = 1
    mnum2 = 6

    # time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    # distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    # gaussian_size = 25  # Why not
    # data_used = 'amplitudes'
    # running = True
    # eliminate_v_zeros = False
    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':session_list,
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':False,
        'num_components':3,
        }

    
    
    ## CCA params ##
    return_warped_data = True
    return_trimmed_data = True
    CCA_dim = 12
    
    ## Plot params ##
    fig_num = 1
    pca_plot_bins = 50
    
    figsize1 = (5,5)
    angle_list1 = [40, 30, 80, 50]
    angle_azim_list1 = [-170, -25, -120, None]
    
    figsize2 = (8,6)
    angle_list2 = [80, -170, -90, 50, 50, -50]
    angle_azim_list2 = [-50, -90, 50, -150, None, -27]

    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)


    
    cca_filename_full_path = get_cca_filename_full_path(CCA_dim, return_trimmed_data, return_warped_data)
    
    #Can be commented after being run once
    aligned_data_dict = perform_mCCA_on_pca_dict(
            PCA_analysis_dict,
            sessions_to_align = 'all', #[0,1,2,3,4,5,6,7,8] [0,1,2,3,4,5,7,8]
            pca_dim = CCA_dim, #6, '85%'
            return_warped_data = return_warped_data,
            return_trimmed_data = return_trimmed_data,
            plot=False,
            shuffle=shuffle
            )
    np.save(cca_filename_full_path, aligned_data_dict, allow_pickle=True)
    #Can be commented after being run once

    aligned_data_dict = np.load(cca_filename_full_path, allow_pickle=True)[()]
    
    
    #FIG 2b: UNALIGNED VS ALIGNED

    fig_CCAcomp, axs_CCAcomp = plt.subplots(nrows=2, ncols = 2, squeeze=False, figsize=figsize1, num=fig_num, subplot_kw={'projection':'3d'})
    # fig_CCAcomp, axs_CCAcomp = plt.subplots(nrows=2, ncols = 2, squeeze=False, figsize=subfig_b_size, num=fig_num)

    fig_num += 1
    
    subplot_counter = 0
    for midx, mnum in enumerate([mnum1, mnum2]):
        for alignment_idx in range(2): #0 is unaligned, 1 is aligned
            
            ax = axs_CCAcomp[midx, alignment_idx]
            if alignment_idx == 0:
                pca_list = aligned_data_dict[mnum, 'pca_unaligned']
            elif alignment_idx == 1:
                pca_list = aligned_data_dict[mnum, 'pca']
                
            pos_list = aligned_data_dict[mnum, 'pos']
            num_sessions = len(pca_list)
            for sidx in range(num_sessions):
                pca = pca_list[sidx]
                pos = pos_list[sidx]
                pos_bins, pca_avg, _ = pf.compute_average_data_by_position(pca, pos, position_bin_size=pca_plot_bins)
                pf.plot_pca_with_position(pca_avg, pos_bins, ax=ax, scatter=False, cbar=None, 
                                          angle = angle_list1[subplot_counter], angle_azim = angle_azim_list1[subplot_counter])
            subplot_counter += 1    
                
    fig_CCAcomp.tight_layout()
    
    fig_name = "fig2_cca_aligned_pcas"    
    save_figure(fig_CCAcomp,fig_name)  
    # return
    #FIG 2c: ALL ALIGNED
    fig_CCAaligned, axs_CCAaligned = plt.subplots(nrows=2, ncols = 3, squeeze=False, figsize=figsize2, num=fig_num, subplot_kw={'projection':'3d'})
    fig_num += 1
    
    #Ignore plots from previous plot
    mlist_for_all_aligned_plot = [mnum for mnum in range(8) if mnum not in [mnum1,mnum2] ]
    
    subplot_counter = 0
    for midx, mnum in enumerate(mlist_for_all_aligned_plot):
        ax = axs_CCAaligned[midx//3, midx%3]
        
        if mnum in mouse_list:
            pca_list = aligned_data_dict[mnum, 'pca']
            pos_list = aligned_data_dict[mnum, 'pos']

            num_sessions = len(pca_list)
            for sidx in range(num_sessions):
                pca = pca_list[sidx]
                pos = pos_list[sidx]
                pos_bins, pca_avg, _ = pf.compute_average_data_by_position(pca, pos, position_bin_size=pca_plot_bins)
                pf.plot_pca_with_position(pca_avg, pos_bins, ax=ax, scatter=False, cbar=None, 
                                          angle = angle_list2[subplot_counter], angle_azim = angle_azim_list2[subplot_counter])
            # ax.set_title('M%d'%mnum, fontsize=25)
            
        subplot_counter += 1
    # fig_CCAaligned.subplots_adjust(wspace=-200, hspace=0)
    
    
    fig_CCAaligned.tight_layout()
    
    fig_name = "fig2_cca_aligned_pcas_remaining"    
    save_figure(fig_CCAaligned,fig_name)
    



def fig2_D_E_F_CCA_quantification():
    ''' Plot figures related to quantifying CCA alignemnt '''
    
    mouse_list = np.arange(8)
    # mouse_list = [2,7]
    session_list = np.arange(9)    
    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':session_list,
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all',
        }
    
    
    ## CCA params ##    
    cca_param_dict = {
        'CCA_dim':'.9',
        'return_warped_data':False,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
        
    
    
    
    ## CCA params ##
    add_shuffle = True
    
    ## Plot params ##
    fig_num = 1
    fs = 15
    
    CCAmap_figsize = (7,7)
    hist_figsize = (7,7)
    CCAavg_figsize = (5,4)
    figsize_sessions = (6,5)

    
    
    num_sessions = len(session_list)
    
    ########## STEP 1 - PCA ###########
    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)
    
    aligned_data_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)

    if add_shuffle:
        cca_param_dict_shuffle = {k:v for k,v in cca_param_dict.items()}
        cca_param_dict_shuffle['shuffle'] = True
        aligned_data_dict_shuffle = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict_shuffle)


    ### ADD SHUFFLE ###
       
    #Get prediction error by alignment kind
    error_by_alignment = {} #(mouse_type_label, CCA_label): list of errors
    for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
        #Gather together mice of same type
        mlist = pparam.MOUSE_TYPE_INDEXES[mouse_type_label]
        for label_idx, label in enumerate(pparam.CCA_LABELS[:4]):
            if label_idx == 3 and add_shuffle == False:
                continue
            error_list = []
            for mnum in mlist:
                if label_idx == 0:
                    errors = aligned_data_dict[mnum, 'unaligned_error_array'].diagonal()
                    
                else:
                
                    if label_idx == 1:
                        errors = aligned_data_dict[mnum, 'unaligned_error_array']
                    
                    elif label_idx == 2:
                        errors = aligned_data_dict[mnum, 'aligned_error_array']
                        
                    elif label_idx == 3:
                        errors = aligned_data_dict_shuffle[mnum, 'aligned_error_array']

                    errors = errors[np.where(~np.eye(errors.shape[0], dtype=bool))]


                error_list.extend(errors)
                
            error_by_alignment[mouse_type_label, label] = error_list
            
            
    #Get *relative* errors by alignment kind
    errordiff_by_alignment = {} #(mouse_type_label, CCA_label)
    errornorm_by_alignment = {} #(mouse_type_label, CCA_label)
    for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
        #Gather together mice of same type
        mlist = pparam.MOUSE_TYPE_INDEXES[mouse_type_label]
        for midx, mnum in enumerate(mlist):
            #Get reference error
            errors_self = aligned_data_dict[mnum, 'unaligned_error_array'].diagonal().astype(int)
            for label_idx, label in enumerate(pparam.CCA_LABELS):
                
                if label == pparam.CCA_LABELS[0]: #Self, skip
                    continue
                
                if label_idx == 3 and add_shuffle == False:
                    continue
                
                if label == pparam.CCA_LABELS[1]: #Unaligned
                    errors = aligned_data_dict[mnum, 'unaligned_error_array']

                elif label == pparam.CCA_LABELS[2]: #Aligned
                    errors = aligned_data_dict[mnum, 'aligned_error_array']
                    
                elif label == pparam.CCA_LABELS[3]:
                    errors = aligned_data_dict_shuffle[mnum, 'aligned_error_array']

                #Error difference
                error_difference = errors - errors_self[:, np.newaxis]
                error_difference = error_difference[np.where(~np.eye(error_difference.shape[0], dtype=bool))]
                
                if midx == 0:
                    errordiff_by_alignment[mouse_type_label, label] = []
                errordiff_by_alignment[mouse_type_label, label].extend(error_difference)   
                
                #Normalized error
                error_norm = errors / errors_self[:, np.newaxis]
                error_norm = error_norm[np.where(~np.eye(error_norm.shape[0], dtype=bool))]
                
                if midx == 0:
                    errornorm_by_alignment[mouse_type_label, label] = []
                errornorm_by_alignment[mouse_type_label, label].extend(error_norm)   
                

    #Fig 2d: CCA colormaps of cross prediction errors
    
    fig_CCAmap, axs_CCAmap = plt.subplots(2, 2, num=fig_num, figsize=CCAmap_figsize); fig_num += 1
    
    #Average them
    mouse_list = np.array(mouse_list)
    mouse_list_by_kind = [mouse_list[mouse_list< 4], mouse_list[mouse_list >=4]]
    
    overall_max_error = 0
    overall_min_error = np.inf
    
    
    for mlist_idx, mlist in enumerate(mouse_list_by_kind):
        
        num_sessions_max = len(session_list)
        alignment_names = ['unaligned_error_array', 'aligned_error_array']
        alignment_error_arrays = {name:np.zeros((num_sessions_max, num_sessions_max)) for name in alignment_names}
        alignment_counter = {name:np.zeros((num_sessions_max, num_sessions_max)) for name in alignment_names}
        for mnum in mlist:
            for name in alignment_names:
                error_array = aligned_data_dict[mnum, name]
                current_session_list = aligned_data_dict[mnum, 'session_list']
                for scounter1, snum1 in enumerate(current_session_list):
                    sidx1 = np.where(np.array(session_list) == snum1)[0]                
                    for scounter2, snum2 in enumerate(current_session_list):
                        sidx2 = np.where(np.array(session_list) == snum2)[0]                
                        alignment_error_arrays[name][sidx1, sidx2] += error_array[scounter1, scounter2]
                        alignment_counter[name][sidx1, sidx2] += 1
    
            
        alignment_error_arrays = {name:alignment_error_arrays[name]/alignment_counter[name] for name in alignment_names}

    
        min_error = np.min([np.min(alignment_error_arrays[name]) for name in alignment_names])
        max_error = np.max([np.max(alignment_error_arrays[name]) for name in alignment_names])
        overall_max_error = np.maximum(overall_max_error, max_error)
        overall_min_error = np.minimum(overall_min_error, min_error)
        unaligned_error_array = alignment_error_arrays['unaligned_error_array']
        aligned_error_array = alignment_error_arrays['aligned_error_array']
    
    
        plot_data_list = [unaligned_error_array, aligned_error_array]
        title_list = ['Unaligned', 'Aligned']
        fs = 15
        for alignment_idx in range(2):
            ax = axs_CCAmap[mlist_idx, alignment_idx]
            data = plot_data_list[alignment_idx]
            ax.imshow(data, cmap=pparam.ERROR_CMAP, interpolation='nearest', vmin=min_error, vmax=max_error)
            snames = [SESSION_NAMES[s] for s in session_list]
            ax.set_xticks(range(num_sessions), snames, fontsize=fs)
            ax.set_yticks(range(num_sessions), snames, fontsize=fs)
            ax.set_title(title_list[alignment_idx], fontsize = fs+6, pad=10)
            for m in range(num_sessions):
                ax.add_patch(Rectangle((m-0.5, m-0.5), 1, 1, fill=False, edgecolor='indianred', lw=3))
        
        
    # axs_CCAmap[0,0].annotate('CA3 i-D', xy=(0, 0.5), xytext=(170, 180), fontsize=20, xycoords='axes points')
    # axs_CCAmap[1,0].annotate('CA3 D-D', xy=(0, 0.5), xytext=(170, 175), fontsize=20, xycoords='axes points')            

        
    axs_CCAmap[0,0].set_ylabel('Trained on', fontsize=fs+4)
    axs_CCAmap[1,0].set_xlabel('Predicted on', fontsize=fs+4)
    
    
    fig_CCAmap.subplots_adjust(right=0.90)
    cbar_ax = fig_CCAmap.add_axes([1.05, 0.17, 0.035, 0.7])    
    norm = mpl.colors.Normalize(vmin=0, vmax=overall_max_error)    
    cbar = fig_CCAmap.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=pparam.ERROR_CMAP), cax=cbar_ax, orientation='vertical', fraction=0.01)
    cbar.ax.set_ylabel('Prediction error (cm)', fontsize=fs+7, rotation=270, labelpad=25)    
    cbar.ax.tick_params(axis='both', which='major', labelsize=fs+5)    
    
    fig_CCAmap.subplots_adjust(wspace=0, hspace=0)
    fig_CCAmap.tight_layout()
    
    #Create separate colorbar
    plt.figure(fig_num); fig_num += 1
    fig_CCAcbar = plt.gcf()
    cbar = pf.add_distance_cbar(fig_CCAcbar, pparam.ERROR_CMAP, vmin = 0, vmax = overall_max_error, fs=fs, 
                                cbar_label = '', 
                                cbar_kwargs = {'fraction':0.555, 'pad':0.04, 'aspect':15})
    
    # cbar_ticks = 5 * (np.linspace(0, 5 * (overall_max_error//5), num=4)//5) #Round up to the closest multiple of 5
    cbar_ticks = np.linspace(0, 5*(overall_max_error//5), num=4).astype(int)
    cbar.ax.set_yticks(cbar_ticks)
    cbar.ax.tick_params(axis='y', labelsize=25) 
    cbar.ax.set_ylabel('Prediction error (cm)', fontsize=fs+7, rotation=270, labelpad=25)

    
    
    #Get prediction error by alignment and session
    error_by_session = {} #(mouse_type_label, CCA_label, snum): list of errors
    error_by_session = {(mtype, CCA_label, snum):[] for mtype in pparam.MOUSE_TYPE_LABELS for CCA_label in pparam.CCA_LABELS[:3] for snum in session_list}
    error_across_sessions = {(mtype, CCA_label):[] for mtype in pparam.MOUSE_TYPE_LABELS for CCA_label in pparam.CCA_LABELS[:3]}
    for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
        #Gather together mice of same type
        mlist = mouse_list_by_kind[mtype_idx]

        for label_idx, cca_label in enumerate(pparam.CCA_LABELS[:3]):
            
            #To be plotted at the end           
            for mnum in mlist:
                
                current_session_list = aligned_data_dict[mnum, 'session_list']
                


                for sidx, snum in enumerate(current_session_list):
                     
                    if label_idx == 0:
                        errors = aligned_data_dict[mnum, 'unaligned_error_array'].diagonal()
                        errors = [errors[sidx]]
                        
                    elif label_idx == 1:
                        errors = aligned_data_dict[mnum, 'unaligned_error_array']
                        errors = errors[:, sidx] #Only when predicting on current session
                        non_current_session_idxs = np.where(~(np.arange(len(current_session_list))==sidx))
                        errors = errors[non_current_session_idxs] #Exclude self prediction
                        

                    
                    elif label_idx == 2:
                        errors = aligned_data_dict[mnum, 'aligned_error_array']
                        errors = errors[:, sidx] #Only when predicting on current session
                        non_current_session_idxs = np.where(~(np.arange(len(current_session_list))==sidx))
                        errors = errors[non_current_session_idxs] #Exclude self prediction
                        
                    error_by_session[mouse_type_label, cca_label, snum].extend(list(errors))
                    # error_across_sessions[mouse_type_label, cca_label].extend(list(errors))
                    error_across_sessions[mouse_type_label, cca_label].extend([np.average(errors)])
    
                    
    #Plot error across sessionf for id and dd
    
    def plot_error_across_sessions(fig_num, add_session_averages=False):
        ''' This function is NOT usable outside fig2_quantification! 
            Used to create small variations on the same data
        '''
            
        fig_across_sessions, axs_session = plt.subplots(2, 1, num=fig_num, figsize=figsize_sessions); fig_num += 1
        ax = plt.gca()
        
        # mice_types = pparam.MOUSE_TYPE_LABELS
        num_sessions = len(session_list)
        for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
            
            ax = axs_session[mtype_idx]
            
            for cca_label_idx, cca_label in enumerate(pparam.CCA_LABELS[:3]):
                color = pparam.CCA_COLORS[cca_label_idx]
    
                avg_list = []
                std_list = []
                
                for sidx, snum in enumerate(session_list):
                    errors = error_by_session[mouse_type_label, cca_label, snum]
                    avg_list.append(np.average(errors))
                    std_list.append(scipy.stats.sem(errors))
                    # ax.scatter([sidx]*len(errors), errors, color=color)
    
    
                        
                
                xx = np.arange(num_sessions)
                avg_list = np.array(avg_list)
                std_list = np.array(std_list)
                ax.plot(xx, avg_list, '-', lw=3, color=color, label=cca_label)
                ax.fill_between(xx, avg_list-std_list, avg_list+std_list, color=color, alpha=0.5)
                
                if add_session_averages == True:
                    #Plot the across-session avg and std
                    errors = error_across_sessions[mouse_type_label, cca_label]
                    last_xpos = xx[-1] + 1 + 0.5*cca_label_idx
                    
                    # #Errorbar
                    # err = np.std(errors)/np.sqrt(len(errors))
                    # # err = np.std(errors)
                    # ax.errorbar([last_xpos], np.average(errors), yerr = err, fmt='o', markersize=6, 
                    #                markeredgewidth=3, elinewidth = 3, zorder=2, color=color)
                    #Boxplot
                    bplot = ax.boxplot([errors], positions = [last_xpos], 
                                    showfliers=False,
                                    vert=True,
                                    patch_artist=True,
                                    widths = 0.3)
                    for box in bplot['boxes']:
                        box.set_facecolor(color)
                        box.set_alpha(0.5)
    
        
            
            #X axis
            if mtype_idx == 1:
                ax.set_xlabel('Session', fontsize=fs+4)
            ax.set_xticks(np.arange(num_sessions), [pparam.SESSION_NAMES[snum] for snum in session_list])
            ax.tick_params(axis='x', labelsize=fs+4)
    
            
            #Y axis
            ax.set_yticks(np.linspace(10 * (ax.get_ylim()[0]//10), 10 * (ax.get_ylim()[1]//10), num=3))
            ax.tick_params(axis='y', labelsize=fs+6)
            if mtype_idx == 0:
                ax.set_ylabel('Error (cm)', fontsize=fs+6)
        
            #Both axis
            ax.spines[['right', 'top']].set_visible(False)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                
            if mtype_idx == 0:
                #Legend
                ax.legend(fontsize=fs, loc='upper right', frameon=False)
    
        fig_across_sessions.subplots_adjust(hspace=35)
        #Figure params
        fig_across_sessions.tight_layout()
        return fig_num, fig_across_sessions
    
    fig_num, fig_across_sessions = plot_error_across_sessions(fig_num, add_session_averages=False)
    fig_num, fig_across_sessions_boxplot = plot_error_across_sessions(fig_num, add_session_averages=True)

    


    


    #Fig 2d&e: CCA histograms     
    fs=20

    ### Plot error results
    fig_hist, axs_hist = plt.subplots(2, 1, num=fig_num, figsize=hist_figsize); fig_num += 1
    
    num_bins_cca_hist = 15
    bin_min = overall_min_error
    bin_max = overall_max_error
    # bin_list = np.linspace(bin_min, bin_max, int((bin_max-bin_min)/2)) #Around 2 error units covered by each bin
    bin_list = np.linspace(bin_min, bin_max, num_bins_cca_hist) #Around 2 error units covered by each bin
    
    for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
        ax = axs_hist.ravel()[mtype_idx]
        
        #Plot bars
        for label_idx, label in enumerate(pparam.CCA_LABELS[:3]):
            errors = error_by_alignment[mouse_type_label, label]
            
            if label_idx == 0:
                errors = errors * (len(session_list)-1)
                
            ax.hist(errors, bins=bin_list, density=True, stacked=False, alpha=0.6,
                    label=label, color=pparam.CCA_COLORS[label_idx])
    
        #Plot average dots
        ymax = ax.get_ylim()[1] * 1.1
        for label_idx, label in enumerate(pparam.CCA_LABELS[:3]):
            errors = error_by_alignment[mouse_type_label, label]

            #Plot average
            ax.errorbar(np.average(errors), ymax, xerr = np.std(errors)/np.sqrt(len(errors)), fmt='o', markersize=6, 
                           markeredgewidth=3, elinewidth = 3, zorder=2, color=pparam.CCA_COLORS[label_idx])
    
    
        
        #Labels, fontsize
        if mtype_idx == 0:
            ax.legend(fontsize=fs-5, frameon=False)
            # ax.set_ylim([0, ymax+3])
            ax.set_ylabel('Normalized density', fontsize=fs)
            
        elif mtype_idx == 1:
            ax.set_xlabel('Error (cm)', fontsize=fs)
            
        ax.set_title('%s Mice' %mouse_type_label, fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        

            
    fig_hist.tight_layout()   
    
    

    

    #Plot averages
    
    def plot_cca_summary(fig_num, 
                         error_by_alignment_dict,
                         plot_type = 'boxplot'):
        ''' This function is NOT usable outside fig2_quantification! 
            Used to create small variations on the same data
            
            plot_type: 'bar' or 'boxplot'
        '''
        fig_CCAavg = plt.figure(num=fig_num, figsize=CCAavg_figsize); fig_num += 1
        ax  = plt.gca()
        all_bars_width = 0.5
        num_of_bars = 3
        barwidth = all_bars_width/(num_of_bars)
        xlabels_pos_list = [0, 1]
        
        
        
        # if not add_shuffle:
        #     cca_labels = pparam.CCA_LABELS[:3]
        # else:
        #     cca_labels = pparam.CCA_LABELS[:4]
        
        cca_labels_by_alignment = [k[1] for k in error_by_alignment_dict.keys()]
        cca_labels_in_dict = [l for l in pparam.CCA_LABELS if l in cca_labels_by_alignment]
        for mtype_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
            xpos_ref = xlabels_pos_list[mtype_idx]
            errors_list = []
            xpos_list = []
            for label_idx, label in enumerate(cca_labels_in_dict):
                errors = error_by_alignment_dict[mouse_type_label, label]
                errors_list.append(errors)
                
                
                if plot_type == 'bar':
                    xpos = xpos_ref - all_bars_width*0.5 + barwidth * (0.5+label_idx)
                    avg = np.average(errors)
                    # err = np.std(errors) ## Change also height in significance!
                    err = scipy.stats.sem(errors) ## Change also height in significance!
                    pltlabel = [None, label][mtype_idx==0]
                    ax.bar(xpos, avg, width=barwidth, alpha=0.7, edgecolor=None, color=pparam.CCA_COLORS[label_idx], label=pltlabel)
                    ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CCA_COLORS[label_idx], alpha=0.6)
                    
                elif plot_type == 'boxplot':
                    xpos = xpos_ref - all_bars_width*0.5 + barwidth * (0.5+label_idx)
                    bplot = ax.boxplot([errors], positions = [xpos], 
                                        showfliers=False,
                                        vert=True,
                                        patch_artist=True,
                                        widths = barwidth,
                                        labels=[pparam.CCA_COLORS[label_idx]])
                    for box in bplot['boxes']:
                        box.set_facecolor(pparam.CCA_COLORS[label_idx])
                        box.set_alpha(0.5)
                xpos_list.append(xpos)

        
            #Add significance with unaligned
            # errors_list_height = [np.average(e)+np.std(e) for e in errors_list]
            errors_list_height = [np.average(e)+np.std(e)/np.sqrt(len(e)) for e in errors_list]
            
            max_height = np.max(errors_list_height) + 3
            extra_height = 0
            d0 = 0.
            dp = -1
            label_padding = -0.025
            pval_fs = 20
            ref_labels = ['Unaligned', 'Aligned (shift)']
            ref_idxs = [cca_labels_in_dict.index(l) for l in cca_labels_in_dict if l in ref_labels]
            for ref_idx_idx, ref_idx in enumerate(ref_idxs):
                ref_label = ref_labels[ref_idx_idx]

                # _, pval_self = scipy.stats.ttest_ind(errors_list[0], errors_list[ref_idx], equal_var=False, permutations=None, alternative='two-sided')
                # _, pval_aligned = scipy.stats.ttest_rel(errors_list[2], errors_list[ref_idx], alternative='two-sided')
                
                tstat, pval_self = scipy.stats.mannwhitneyu(errors_list[0], errors_list[ref_idx], use_continuity=False, alternative='two-sided')
                tstat, pval_aligned = scipy.stats.wilcoxon(errors_list[2], errors_list[ref_idx], zero_method='wilcox', correction=False, alternative='two-sided')

                
                if ref_idx_idx == 0: #Ignore self to aligned (shift)
                    print('self to unaligned', pval_self)
                    pf.draw_significance(ax, pval_self, max_height + extra_height, xpos_list[0], xpos_list[ref_idx], d0, dp, orientation='top', thresholds = [0.01], fs=pval_fs, label_padding=label_padding)
                    extra_height += 3
                print('aligned to %s'%ref_label, pval_aligned)
                pf.draw_significance(ax, pval_aligned, max_height + extra_height, xpos_list[2], xpos_list[ref_idx], d0, dp, orientation='top', thresholds = [0.01], fs=pval_fs, label_padding=label_padding)
                extra_height += 3

            
            # #Significance self with aligned
            # _, pval_self_aligned = scipy.stats.ttest_ind(errors_list[0], errors_list[2], equal_var=True, permutations=None, alternative='two-sided')
            # pf.draw_significance(ax, pval_self_aligned, max_height*1.01 + extra_height, xpos_list[0], xpos_list[2], d0, dp, orientation='top', thresholds = [0.01], fs=pval_fs, label_padding=label_padding)




        xlabels = pparam.MOUSE_TYPE_LABELS
        ax.set_xticks(xlabels_pos_list, xlabels, fontsize=fs)
        ax.set_ylabel('Error (cm)', fontsize=fs)
        ax.tick_params(axis='y', which='major', labelsize=fs)
        ax.legend(fontsize=fs-7, frameon=False)
        
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        
        fig_CCAavg.tight_layout()
        
        return fig_num, fig_CCAavg, ax
    
    

    fig_num, fig_CCAavg, ax_CCAavg = plot_cca_summary(fig_num, 
                                            error_by_alignment, 
                                            plot_type='bar')
    
    


    
    fig_to_save = [(fig_across_sessions, "fig2_D_prediction_across_sessions"),
                   (fig_across_sessions_boxplot, "fig2_D_prediction_across_sessions_boxplot"),
                   (fig_CCAmap, 'fig2_C_cross_prediction_errors'),
                   (fig_CCAcbar, 'fig2_C_cross_prediction_errors_colorbar'),
                   (fig_hist, 'fig2_histogram'),
                   (fig_CCAavg, 'fig2_F_error_avg')
                   ]
    
    for fig, fig_name in fig_to_save:
        save_figure(fig, fig_name)

    return    
    



        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIG 3 - TCA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






def perform_APdecoding_on_cca_param_dict(CCA_analysis_dict, ap_decoding_param_dict):
    ''' Function that takes the output of CCA, takes the trial factors, and predicts the required label 
        See "ap_decoding_param_dict_default" in project_parameters for an explanation of each parameter
    
    '''
    
    LDA_components = 1
    max_tca_on_lda_attempts = 100 #TCA on LDA is repeated until the required number of repetitions is reached, but in case it never converges, this will stop it
    max_shuffle_LDA_attempts = 25 #Shuffle is less likely to converge when using LDA, so we might need many tries
    
    mouse_list = CCA_analysis_dict['mouse_list']
    TCA_dims_param = ap_decoding_param_dict['TCA_factors']
    # #LDA parameters
    LDA_imbalance_prop = ap_decoding_param_dict['LDA_imbalance_prop']
    LDA_imbalance_repetitions = ap_decoding_param_dict['LDA_imbalance_repetitions']
    LDA_trial_shuffles = ap_decoding_param_dict['LDA_trial_shuffles']
    LDA_session_shuffles = ap_decoding_param_dict['LDA_session_shuffles']
    
    
    num_bins = CCA_analysis_dict['num_bins']
    
    APdecoding_dict = {}
    for mnum in mouse_list:
        
        session_list = CCA_analysis_dict[mnum, 'session_list']
        pos_list = CCA_analysis_dict[mnum, 'pos']
        pca_list = CCA_analysis_dict[mnum, 'pca']
        # pca_list = CCA_analysis_dict[mnum, 'pca_unaligned']
                
        data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list)
        num_features, num_bins, total_trials = data_by_trial.shape
        num_CCA_dims = num_features
        if TCA_dims_param == 'max':
            num_TCA_dims = num_CCA_dims
        else:
            num_TCA_dims = int(TCA_dims_param)
        print('Performing TCA+LDA on M%d // CCA dim: %d, // TCA dim: %d' %(mnum, num_CCA_dims, num_TCA_dims))

        #Limit position (if indicated)
        if ap_decoding_param_dict['exclude_positions'] == True:
            positions = pos_by_trial[:,0] #Assumes all trials are binned using the same positions            
            pos_bool_filtered_out = pf.get_idxs_in_periodic_interval(positions, ap_decoding_param_dict['pos_to_exclude_from'],
                                           ap_decoding_param_dict['pos_to_exclude_to'], pparam.MAX_POS)
            pos_bool_selected = np.invert(pos_bool_filtered_out)
            data_by_trial = data_by_trial[:, pos_bool_selected, :]
            pos_by_trial = pos_by_trial[pos_bool_selected, :]
            
        num_bins_current = data_by_trial.shape[1]
            
        #Selecting trials to decode
        trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, ap_decoding_param_dict['session_comparisons'])
        num_trials_to_keep = len(label_by_trial)
        
        f1_best = -1
        
        f1_array = np.zeros((0, 2)) #1st axis is TCA repetition, 2nd is class 0 or 1
        # accuracy_array = []
        accuracy_array = np.zeros((0, 2)) #1st axis is TCA repetition, 2nd is class 0 or 1
        LDA_prob_array = np.zeros((0, num_trials_to_keep)) #1st axis is TCA repetition, 2nd is trial
        LDA_projection_array = np.zeros((0, num_trials_to_keep)) #1st axis is TCA repetition, 2nd is trial
        label_by_trial_predicted = np.zeros((0, num_trials_to_keep), dtype=int) #1st axis is TCA repetition, 2nd is trial
        
        feature_factors = np.zeros((0, num_CCA_dims, num_TCA_dims)) #1st axis is TCA repetition, 2nd is latent input dimension, 3rd is TCA dim
        time_factors = np.zeros((0, num_bins_current, num_TCA_dims)) #1st axis is TCA repetitions, 2nd is time bin, 3rd is TCA dim
        trial_factors = np.zeros((0, num_trials_to_keep, num_TCA_dims)) #1st axis is TCA repetition, 2nd is trial dim, 3rd is TCA dim

        APdecoding_weights = np.zeros((0, data_by_trial.shape[0])) #1st axis is TCA repetition, 2nd is input data feature. Gets the weight of each input dimension for AP decoding
        TCA_on_LDA_counter = 0
        while TCA_on_LDA_counter < ap_decoding_param_dict['TCA_on_LDA_repetitions'] and TCA_on_LDA_counter < max_tca_on_lda_attempts:
            # Step 4: TCA
            KTensor = APfuns.perform_TCA(data_by_trial, num_TCA_dims, ap_decoding_param_dict['TCA_replicates'], 
                                         ap_decoding_param_dict['TCA_method'], ap_decoding_param_dict['TCA_convergence_attempts'])
            feature_factors_temp, time_factors_temp, trial_factors_temp = KTensor
            LDA_input = trial_factors_temp[trials_to_keep]           

            #LDA on TCA factors
            try:                
                LDA_results_dict = APfuns.perform_LDA(LDA_input, label_by_trial, LDA_components, LDA_imbalance_prop, LDA_imbalance_repetitions)
            except np.linalg.LinAlgError:
                continue
            
            #Update arrays
            f1 = LDA_results_dict['f1']
            f1_array = np.vstack((f1_array, f1))
            # accuracy_array.append([LDA_results_dict['accuracy']])
            accuracy_array = np.vstack((accuracy_array, LDA_results_dict['accuracy']))
            LDA_prob_array = np.vstack((LDA_prob_array, LDA_results_dict['LDA_prob']))
            LDA_projection_array = np.vstack((LDA_projection_array, LDA_results_dict['LDA_projection'].squeeze()))
            label_by_trial_predicted = np.vstack((label_by_trial_predicted, LDA_results_dict['label_predicted']))
            
            feature_factors = np.vstack((feature_factors, feature_factors_temp[np.newaxis]))
            time_factors = np.vstack((time_factors, time_factors_temp[np.newaxis]))
            trial_factors = np.vstack((trial_factors, LDA_input[np.newaxis]))
            
            #Compute dimensional weights
            weights = np.abs(LDA_results_dict['weights'].reshape(num_TCA_dims, 1))
            # feature_factors = KTensor[0] #TCA weights for each data dimension
            APdecoding_weights_rep = np.dot(feature_factors_temp, weights)
            APdecoding_weights = np.vstack((APdecoding_weights, np.squeeze(APdecoding_weights_rep)))


            if np.min(f1) > np.min(f1_best): #Get the best one
                best_LDA_input = LDA_input #Used as reference for shuffle
                f1_best = f1
            
            TCA_on_LDA_counter += 1

        if f1_array.shape[0] == 0:
            print('WARNING: LDA ON TCA FAILED TO CONVERGE AFTER %d TRIES'%max_tca_on_lda_attempts)
        

        #Random shuffle (done on best TCA run)
        f1_array_shuffle = np.zeros((LDA_trial_shuffles,2))
        accuracy_array_shuffle = np.zeros((LDA_trial_shuffles, 2))
        LDA_prob_shuffle = np.zeros((LDA_trial_shuffles))
        LDA_projection_array_shuffle = np.zeros((LDA_trial_shuffles, num_trials_to_keep)) #1st axis is TCA repetition, 2nd is trial

        num_trials = len(label_by_trial)
        if LDA_trial_shuffles != 0 or LDA_session_shuffles != 0:
            print('Starting shuffle')
        for randidx in range(LDA_trial_shuffles):
            shuffle_attempt_counter = 0
            while shuffle_attempt_counter < max_shuffle_LDA_attempts: #We repeat until it converges
                shuffle_attempt_counter += 1
                shuffled_idxs = np.random.choice(np.arange(num_trials), size=num_trials, replace=False)
                label_by_trial_shuffled = label_by_trial[shuffled_idxs]
                try:
                    LDA_results_dict_shuffle = APfuns.perform_LDA(best_LDA_input, label_by_trial_shuffled, LDA_components, LDA_imbalance_prop, LDA_imbalance_repetitions)
                except np.linalg.LinAlgError:
                    continue

                f1_array_shuffle[randidx] = LDA_results_dict_shuffle['f1']
                accuracy_array_shuffle[randidx] = LDA_results_dict_shuffle['accuracy']
                LDA_prob_shuffle[randidx] = np.average(LDA_results_dict_shuffle['LDA_prob'])
                LDA_projection_array_shuffle = np.vstack((LDA_projection_array_shuffle, LDA_results_dict_shuffle['LDA_projection'].squeeze()))
                
                break
                
            else:
                print('we are in trouble! random shuffle max attempts reached!')
            
        #SESSION SHUFFLE (NOT USED)
        f1_array_session_shuffle = np.zeros((LDA_session_shuffles,2))
        LDA_prob_session_shuffle = np.zeros((LDA_session_shuffles))
        LDA_projection_array_session_shuffle = np.zeros((LDA_trial_shuffles, num_trials_to_keep)) #1st axis is TCA repetition, 2nd is trial

        #Step 1: which sessions were selected?
        # print(trials_to_keep)
        snum_by_trial_selected = snum_by_trial[trials_to_keep]
        selected_sessions = np.unique(snum_by_trial_selected)
        num_selected_sessions = len(selected_sessions)

        #Step 2: get label assigned to each session
        first_trial_per_snum = {snum:np.argmax(snum_by_trial_selected==snum) for snum in selected_sessions}
        label_list = np.array([label_by_trial[first_trial_per_snum[snum]] for snum in selected_sessions])
        
        for randidx in range(LDA_session_shuffles):
            session_shuffle_attempt_counter = 0
            while session_shuffle_attempt_counter < max_shuffle_LDA_attempts:
                session_shuffle_attempt_counter += 1
                
                # #Shuffle indexes at the session level, making sure they are different from original
                while True:
                    shuffled_session_idxs = np.random.choice(np.arange(num_selected_sessions), size=num_selected_sessions, replace=False)
                    label_list_shuffled = label_list[shuffled_session_idxs]
                    
                    label_by_trial_session_shuffled = np.array([label_list_shuffled[list(selected_sessions).index(snum)] for snum in snum_by_trial_selected])
                    
                    if np.allclose(label_list, label_list_shuffled) == False: #All labels must not be equal to the original
                        break
                    
                try:
                    LDA_results_dict_session_shuffle = APfuns.perform_LDA(best_LDA_input, label_by_trial_session_shuffled, LDA_components, LDA_imbalance_prop, LDA_imbalance_repetitions)
                except np.linalg.LinAlgError:
                    continue

                f1_array_session_shuffle[randidx] = LDA_results_dict_session_shuffle['f1']
                LDA_prob_session_shuffle[randidx] = np.average(LDA_results_dict_session_shuffle['LDA_prob'])
                LDA_projection_array_session_shuffle = np.vstack((LDA_projection_array_session_shuffle, LDA_results_dict_session_shuffle['LDA_projection'].squeeze()))
                
                break
                
            else:
                print('we are in trouble! random session shuffle max attempts reached!')

        
        snum_by_trial = snum_by_trial[trials_to_keep]

        
        #Objective dictionary
        APdecoding_dict.update({
            (mnum, 'snum_by_trial'):snum_by_trial,
            (mnum, 'selected_trials'):trials_to_keep,
            (mnum, 'label_by_trial'):label_by_trial,
            (mnum, 'label_by_trial_predicted'):label_by_trial_predicted,
            (mnum, 'LDA_projection_array'):LDA_projection_array,
            
            (mnum, 'feature_factors'):feature_factors,
            (mnum, 'time_factors'):time_factors,
            (mnum, 'trial_factors'):trial_factors,
            (mnum, 'APdecoding_weights'):APdecoding_weights,

            (mnum, 'LDA_prob_correct'):LDA_prob_array,
            (mnum, 'f1_array'):f1_array,
            (mnum, 'accuracy_array'):np.array(accuracy_array),

            (mnum, 'LDA_prob_correct_shuffle'):LDA_prob_shuffle,
            (mnum, 'f1_array_shuffle'):f1_array_shuffle,
            (mnum, 'accuracy_array_shuffle'):accuracy_array_shuffle,

            (mnum, 'LDA_prob_correct_session_shuffle'):LDA_prob_session_shuffle,
            (mnum, 'f1_array_session_shuffle'):f1_array_session_shuffle,

            })
        
    return APdecoding_dict
    
def APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict, force_recalculation=False):
    ''' Given a parameter dict, peform PCA, CCA, and TCA+LDA on it '''
    



        
    ########## STEP 1 - PCA ###########    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)

    ############## STEP 2: mCCA ############
    CCA_analysis_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)
        
    ############# STEP 3: TCA + LDA ############    
    APdecoding_dict = perform_APdecoding_on_cca_param_dict(CCA_analysis_dict, ap_decoding_param_dict)

            
    pipeline_output_dict = {
            'PCA_analysis_dict':PCA_analysis_dict,
            'CCA_analysis_dict':CCA_analysis_dict,
            'APdecoding_dict':APdecoding_dict,
            }
    
    
    np.save(OUTPUT_PATH + "pipeline_output_dict.npy", pipeline_output_dict, allow_pickle=True)
    
    return pipeline_output_dict


def fig3_A_and_fig3SI_A_TCA_factors():
    
    '''
    
    Plot LDA projection from TCA factors

    '''
    
    mouse_list = [6]


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(7),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':'.9', #11, '.9'
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':200,
        'pos_to_exclude_to':1300,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max', #int, or 'max' to get the maximum possible (determined by CCA)
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':2,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':0,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
    
    fig_num = plt.gcf().number + 1
    fs = 25
    fs_title = 35
    lw = 5
    trial_smoothing_size = 3 #trial smoothing in "trial" number



    ########## STEP 1 - PCA ###########    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)

    ############## STEP 2: mCCA ############
    CCA_analysis_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)
        
    ############# STEP 3: TCA + LDA ############
    mouse_list = CCA_analysis_dict['mouse_list']
    TCA_dims_param = ap_decoding_param_dict['TCA_factors']
    # #LDA parameters
    
    
    num_bins = CCA_analysis_dict['num_bins']
    
    for mnum in mouse_list:
        
        session_list = CCA_analysis_dict[mnum, 'session_list']
        pos_list = CCA_analysis_dict[mnum, 'pos']
        pca_list = CCA_analysis_dict[mnum, 'pca']
        # pca_list = CCA_analysis_dict[mnum, 'pca_unaligned']
                
        data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list)
        num_features, num_bins, total_trials = data_by_trial.shape
        num_CCA_dims = num_features
        if TCA_dims_param == 'max':
            num_TCA_dims = num_CCA_dims
        else:
            num_TCA_dims = int(TCA_dims_param)
        print('Performing TCA+LDA on M%d // CCA dim: %d, // TCA dim: %d' %(mnum, num_CCA_dims, num_TCA_dims))

        #Limit position (if indicated)
        if ap_decoding_param_dict['exclude_positions'] == True:
            positions = pos_by_trial[:,0] #Assumes all trials are binned using the same positions            
            pos_bool_filtered_out = pf.get_idxs_in_periodic_interval(positions, ap_decoding_param_dict['pos_to_exclude_from'],
                                           ap_decoding_param_dict['pos_to_exclude_to'], pparam.MAX_POS)
            pos_bool_selected = np.invert(pos_bool_filtered_out)
            data_by_trial = data_by_trial[:, pos_bool_selected, :]
            pos_by_trial = pos_by_trial[pos_bool_selected, :]
            
            
        #Selecting trials to decode
        trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, ap_decoding_param_dict['session_comparisons'])
        # snum_by_trial = snum_by_trial[trials_to_keep]
        # data_by_trial = data_by_trial[:, :, trials_to_keep]
        

        plotted_factors = False
        while plotted_factors == False:
            # Step 4: TCA
            KTensor, TCA_ensemble = APfuns.perform_TCA(data_by_trial, num_TCA_dims, ap_decoding_param_dict['TCA_replicates'], 
                                         ap_decoding_param_dict['TCA_method'], ap_decoding_param_dict['TCA_convergence_attempts'],
                                         return_ensemble = True)
            feature_factors, time_factors, trial_factors = KTensor
                        
            plotted_factors = True
            
            ''' Plots TCA results for trials of an aversive task following Negar's experimental design.
                Assumes the trials are concatenated across sessions, computed for PCA.
                TCA_ensemble: output from tensortools' TCA method
                session_list: list of session numbers used
                num_trials_by_snum: number of trials per session number
            '''

            TCA_dim = feature_factors.shape[1]
            
            
            tca_factors_figsize = (3 * 4, TCA_dim*2)
            ncols = 3
            nrows = feature_factors.shape[0]
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=tca_factors_figsize, num=fig_num); fig_num += 1  

            
            for factor in range(TCA_dim):
                for factor_type in range(3):
                    ax = axs[factor, factor_type]
            
                    vals = KTensor[factor_type][:, factor]
                    xx = np.arange(0, len(vals))
                    
                    if factor_type == 2:
                        vals = np.convolve(vals, np.ones(trial_smoothing_size), mode='same')/trial_smoothing_size
                    
                    color = 'black' #'tab:blue'
                    ax.plot(xx, vals, color=color, lw=lw)
                    minval, maxval = np.min(vals), np.max(vals)
                    ax.set_ylim([minval, maxval])
                    ax.set_yticks([minval, maxval], [np.around(minval, decimals=2), np.around(maxval, decimals=2)], fontsize=fs)
            
                    if factor_type == 0: #PCA visualizations
                        pca_dim_list = range(vals.shape[0])
                        
                        ax.set_xticks(pca_dim_list, pca_dim_list, fontsize=fs)
                        # ax.set_yticks([minval, maxval], [np.around(minval, decimals=2), np.around(maxval, decimals=2)], fontsize=fs)
                        for pca_dim_idx in pca_dim_list:
                            yy = np.linspace(minval, maxval)
                            xx = [pca_dim_idx] * len(yy)
                            ax.plot(xx, yy, '--', color='gray', alpha=0.5)
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                        if factor == TCA_dim-1:
                            ax.set_xlabel('PCA dim', fontsize=fs_title)
            
                        if factor == 0:
                            ax.set_title('TCA factors (AU)', fontsize=fs_title)
                            
                    if factor_type == 1: #Time/position factor visualization
                        yy = np.linspace(minval, maxval)
                        pos_landmarks = [0, 50, 100, 149]
                        pos_landmarks_names = [0, 500, 1000, 1500]
                        colors_landmarks = ['black', 'indianred', 'forestgreen', 'black']
                        ax.set_xticks(pos_landmarks, pos_landmarks_names, fontsize=fs)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                        for landmark in range(len(pos_landmarks)):
                            xx = [pos_landmarks[landmark]] * len(yy)
                            ax.plot(xx, yy, '--', color=colors_landmarks[landmark], lw=lw, alpha=0.6)
                        
                        if factor == TCA_dim-1:
                            ax.set_xlabel('Position (mm)', fontsize=fs_title)
                            
            
                            
                    if factor_type == 2: #Trial factors
                        APfuns.add_session_delimiters_to_trial_plot(ax, snum_by_trial, fs=fs)
                        if factor == TCA_dim-1:
                            ax.set_xlabel('Trial number', fontsize=fs_title)

                        
                    ax.tick_params(axis='x', which='major', labelsize=fs)
                    ax.tick_params(axis='y', which='major', labelsize=fs)
                    
                    ax.spines[['right', 'top']].set_visible(False)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(lw)
            fig.tight_layout()
            fig_name = 'fig3_A_TCA_factors_M%d'%mnum
            save_figure(fig, fig_name)

    return
    
    
    
def fig3_B_LDA_on_TCA():
    '''
    
    Plot LDA projection from TCA factors

    '''
    
    mouse_list = [2,6]


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(7),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':'.9', #11, '.9'
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':200,
        'pos_to_exclude_to':1300,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max', #int, or 'max' to get the maximum possible (determined by CCA)
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':20,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':0,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    




    ########## STEP 1 - PCA ###########    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)

    ############## STEP 2: mCCA ############
    CCA_analysis_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)
        
    ############# STEP 3: TCA + LDA ############    

    LDA_components = 1
    
    mouse_list = CCA_analysis_dict['mouse_list']
    TCA_dims_param = ap_decoding_param_dict['TCA_factors']
    # #LDA parameters
    
    
    num_bins = CCA_analysis_dict['num_bins']
    
    for mnum in mouse_list:
        
        session_list = CCA_analysis_dict[mnum, 'session_list']
        pos_list = CCA_analysis_dict[mnum, 'pos']
        pca_list = CCA_analysis_dict[mnum, 'pca']
        # pca_list = CCA_analysis_dict[mnum, 'pca_unaligned']
                
        data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list)
        num_features, num_bins, total_trials = data_by_trial.shape
        num_CCA_dims = num_features
        if TCA_dims_param == 'max':
            num_TCA_dims = num_CCA_dims
        else:
            num_TCA_dims = int(TCA_dims_param)
        print('Performing TCA+LDA on M%d // CCA dim: %d, // TCA dim: %d' %(mnum, num_CCA_dims, num_TCA_dims))

        #Limit position (if indicated)
        if ap_decoding_param_dict['exclude_positions'] == True:
            positions = pos_by_trial[:,0] #Assumes all trials are binned using the same positions            
            pos_bool_filtered_out = pf.get_idxs_in_periodic_interval(positions, ap_decoding_param_dict['pos_to_exclude_from'],
                                           ap_decoding_param_dict['pos_to_exclude_to'], pparam.MAX_POS)
            pos_bool_selected = np.invert(pos_bool_filtered_out)
            data_by_trial = data_by_trial[:, pos_bool_selected, :]
            pos_by_trial = pos_by_trial[pos_bool_selected, :]
            
            
        #Selecting trials to decode
        trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, ap_decoding_param_dict['session_comparisons'])
        num_trials_to_keep = len(label_by_trial)
        
        
        label_by_trial_predicted = np.zeros((0, num_trials_to_keep), dtype=int) #1st axis is TCA repetition, 2nd is trial
        
        plotted_factors = False
        while plotted_factors == False:
            # Step 4: TCA
            KTensor = APfuns.perform_TCA(data_by_trial, num_TCA_dims, ap_decoding_param_dict['TCA_replicates'], 
                                         ap_decoding_param_dict['TCA_method'], ap_decoding_param_dict['TCA_convergence_attempts'])
            feature_factors_temp, time_factors_temp, trial_factors_temp = KTensor
            LDA_input = trial_factors_temp[trials_to_keep]           

            #LDA on TCA factors
            # LDA_results_dict = APfuns.perform_LDA(LDA_input, label_by_trial, LDA_components, LDA_imbalance_prop, LDA_imbalance_repetitions)
            
            
            
            LDA = LinearDiscriminantAnalysis(solver="eigen", #svd, lsqr, eigen
                                              shrinkage=0,  #None, "auto", float 0-1
                                              n_components=LDA_components, #Dimensionality reduction
                                              store_covariance=False #Only useful for svd, which doesn't automatically calculate it
                                              )
            try:
                LDA.fit(LDA_input, label_by_trial)
            except np.linalg.LinAlgError:
                continue
            
            label_by_trial_predicted = (LDA.predict(LDA_input))
            f1 = np.average(pf.multiclass_f1(label_by_trial, label_by_trial_predicted))
        
            trial_factors_LDA_projection = np.squeeze(LDA.transform(LDA_input))
            fig, ax = APfuns.plot_LDA_projection(trial_factors_LDA_projection, label_by_trial, snum_by_trial=snum_by_trial, plot_legend=True, ax=None)
            
            ax.set_title('M%d, F1 = %.1f'%(mnum, f1), fontsize=15)
            fig_name = 'fig3_B_LDA_example_M%d'%mnum
            save_figure(fig, fig_name)
            plotted_factors = True

    return

def fig3_C_D_f1_plots_and_fig3SI_B_C_accuracy_plots():
    
    ''' Performs TCA across sessions for an animal.
        Step 1: perform PCA, limit to minimum possible of dimensions
        Step 2: align through CCA, so every dimension represents something similar about the data
        Step 3: split into trials through warping
        Step 4: TCA!
    '''
    
    mouse_list = np.arange(8)
    # mouse_list = [6]
    # mouse_list = [2,6]


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':'.9', #11, '.9'
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':200,
        'pos_to_exclude_to':1300,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max', #int, or 'max' to get the maximum possible (determined by CCA)
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':25,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':25,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
    CCA_random_shifts = 25    #Number of random shifts WARNING: LDA PROB NOT CALCULATED FOR IT!!!
    
    
    print('STARTING')
    print('Sessions: %d'%len(preprocessing_param_dict['session_list']))
    print('V zeros: %s'%(str(preprocessing_param_dict['eliminate_v_zeros'])))
    print('CCA dim: %s'%str(cca_param_dict['CCA_dim']))
    print('TCA dim: %s'%str(ap_decoding_param_dict['TCA_factors']))


    #Do analysis
    pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict)
    
    if CCA_random_shifts > 0:
        cca_param_dict_shift = {k:v for k,v in cca_param_dict.items()}
        ap_decoding_param_dict_shift = {k:v for k,v in ap_decoding_param_dict.items()}
        cca_param_dict_shift['shuffle'] = True
        ap_decoding_param_dict_shift['TCA_on_LDA_repetitions'] = 5
        ap_decoding_param_dict_shift['LDA_trial_shuffles'] = 0
        ap_decoding_param_dict_shift['LDA_session_shuffles'] = 0
        
        f1_array_shift = np.zeros((len(mouse_list), CCA_random_shifts, 2))
        accuracy_array_shift = np.zeros((len(mouse_list), CCA_random_shifts, 2))
        for shift_run in range(CCA_random_shifts):
            pipeline_output_dict_shift = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict_shift, ap_decoding_param_dict_shift)
            AP_decoding_shift_current = pipeline_output_dict_shift['APdecoding_dict']
            for midx, mnum in enumerate(mouse_list):
                f1 = AP_decoding_shift_current[mnum, 'f1_array']
                f1_array_shift[midx, shift_run] = np.average(f1, axis=0)
                acc = AP_decoding_shift_current[mnum, 'accuracy_array']
                accuracy_array_shift[midx, shift_run] = np.average(acc, axis=0)
                
                
    
    

    #PUTTING DATA IN ARRAYS
    mtype_by_mouse = np.array([pparam.MOUSE_TYPE_LABEL_BY_MOUSE[mnum] for mnum in mouse_list])
    num_mice = len(mouse_list)
    LDA_trial_shuffles = ap_decoding_param_dict['LDA_trial_shuffles']
    LDA_session_shuffles = ap_decoding_param_dict['LDA_session_shuffles']
    
    APdecoding_dict = pipeline_output_dict['APdecoding_dict']
    
    best_lda_by_mouse = np.zeros(len(mouse_list), dtype=int)
    best_f1_by_mouse = np.zeros((len(mouse_list), 2))
    
    f1_avg_by_mouse = np.zeros((len(mouse_list), 2))
    f1_std_by_mouse = np.zeros((len(mouse_list), 2))
    
    accuracy_avg_by_mouse = np.zeros((len(mouse_list), 2))
    accuracy_std_by_mouse = np.zeros((len(mouse_list), 2))

    
    for midx, mnum in enumerate(mouse_list):
        f1_array = APdecoding_dict[mnum, 'f1_array']
        min_f1 = np.min(f1_array, axis=1)
        best_lda_idx = np.argmax(min_f1)
        best_lda_by_mouse[midx] = best_lda_idx
        best_f1 = f1_array[best_lda_idx]
        best_f1_by_mouse[midx] = best_f1
        
        #Avg F1
        f1_avg_by_mouse[midx] = np.average(f1_array, axis=0)
        f1_std_by_mouse[midx] = np.std(f1_array, axis=0)
        # f1_std_by_mouse[midx] = np.std(f1_array, axis=0)/np.sqrt(f1_array.shape[0])

        #Accuracy
        acc = APdecoding_dict[mnum, 'accuracy_array']
        accuracy_avg_by_mouse[midx] = np.average(acc, axis=0)
        accuracy_std_by_mouse[midx] = np.std(acc, axis=0)
                    
    #Take the indicated shuffles
    control_names = []
    control_label_idxs = []
    if LDA_trial_shuffles > 0:
        control_names.append('shuffle')
        control_label_idxs.append(2)
    if LDA_session_shuffles > 0:
        control_names.append('session_shuffle')
        control_label_idxs.append(3)
    if CCA_random_shifts > 0:
        control_names.append('CCA_shift')
        control_label_idxs.append(4)

        
    control_data = {}    
    for control in control_names:       
        f1_avg_control_by_mouse = np.zeros((len(mouse_list), 2))
        f1_std_control_by_mouse = np.zeros((len(mouse_list), 2))
        
        accuracy_avg_control_by_mouse = np.zeros((len(mouse_list), 2))
        accuracy_std_control_by_mouse = np.zeros((len(mouse_list), 2))
        
        for midx, mnum in enumerate(mouse_list):
            if 'shuffle' in control:
                f1_control = APdecoding_dict[mnum, 'f1_array_' + control]
                accuracy_control = APdecoding_dict[mnum, 'accuracy_array_' + control]
            
            elif 'CCA' in control:
                f1_control = f1_array_shift[midx]
                accuracy_control = accuracy_array_shift[midx]

            
            f1_avg_control_by_mouse[midx] = np.average(f1_control, axis=0)
            f1_std_control_by_mouse[midx] = np.std(f1_control, axis=0)
            # f1_std_control_by_mouse[midx] = np.std(f1_control, axis=0)/np.sqrt(f1_control.shape[0])
            # f1_std_control_by_mouse[midx] = np.average(np.std(f1_control, axis=0))

            accuracy_avg_control_by_mouse[midx] = np.average(accuracy_control, axis=0)
            accuracy_std_control_by_mouse[midx] = np.std(accuracy_control, axis=0)
            
        control_data[control, 'f1_avg'] = f1_avg_control_by_mouse
        control_data[control, 'f1_std'] = f1_std_control_by_mouse
        control_data[control, 'accuracy_avg'] = accuracy_avg_control_by_mouse
        control_data[control, 'accuracy_std'] = accuracy_std_control_by_mouse
    
    #Plot parameters
    fig_num = 1
    pca_plot_bins = 50

    nrows = 2
    ncols = num_mice // 2 + num_mice % 2
    if num_mice == 1:
        nrows=1
        
    #### PLOT PCAS ####
    fig_pca, axs_pca = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(4 * ncols, 3 * nrows), num=fig_num, subplot_kw={'projection':'3d'}); fig_num += 1  
    for midx, mnum in enumerate(mouse_list): 

        CCA_analysis_dict = pipeline_output_dict['CCA_analysis_dict']
        pca_list, pos_list = CCA_analysis_dict[mnum, 'pca'], CCA_analysis_dict[mnum, 'pos']
        ax = axs_pca.ravel()[midx]; ax.set_title("M%d" %mnum, fontsize=15)
        pf.plot_pca_overlapped(pca_list, pos_list, average_by_session=True, pca_plot_bins=pca_plot_bins, ax=ax)
        
        



    #### DECODING PERFORMANCE PLOTS ####
    fs=15
    ymin = 0.2
    figsize_decoding_by_mouse = (6,3)
    figsize_decoding_summary = (4,4)
    decoding_avg_std_pairs = [(f1_avg_by_mouse, f1_std_by_mouse), (accuracy_avg_by_mouse, accuracy_std_by_mouse)]
    decoding_key_list = ['f1', 'accuracy']
    decoding_key_list = ['f1', 'accuracy']

    for decoding_pair_idx, (avg_by_mouse, std_by_mouse) in enumerate(decoding_avg_std_pairs):
        decoding_key = decoding_key_list[decoding_pair_idx]
        decoding_label = decoding_key[0].upper() + decoding_key[1:] #Make first letter uppercase
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=figsize_decoding_by_mouse, num=fig_num)
        ax = ax[0,0]; fig_num += 1
        xpos_list = np.arange(num_mice)
        shift = 0.15
        for class_idx, class_label in enumerate(range(2)):
            label=pparam.AP_DECODING_LABELS[class_idx]
            color=pparam.AP_DECODING_COLORS[class_idx]
            perf = avg_by_mouse[:, class_idx]
            perfstd = std_by_mouse[:, class_idx]
            xx = xpos_list
            xx = xpos_list - shift * (1-2*class_idx)
            ax.errorbar(xx, perf, yerr=perfstd, fmt='_', color=color, alpha=0.9, label=label, 
                              markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)

       
        for idx, control in enumerate(control_names):
            control_idx = control_label_idxs[idx]

            for class_idx, class_label in enumerate(range(2)):
                if class_idx == 0:
                    label=pparam.AP_DECODING_LABELS[control_idx]
                else:
                    label=None
                color=pparam.AP_DECODING_COLORS[control_idx]
                perf = control_data[control, decoding_key + '_avg'][:, class_idx]
                perfstd = control_data[control, decoding_key + '_std'][:, class_idx]
                xx = xpos_list
                xx = xpos_list - shift * (1-2*class_idx)
                ax.errorbar(xx, perf, yerr=perfstd, fmt='_', color=color, alpha=0.9, label=label, 
                                  markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
        
        
        
                    
        ## Plot significances with controls
        ymax = ax.get_ylim()[1]
        if len(control_names) > 0:
            for midx, mnum in enumerate(mouse_list):
                
                ## Pval for each individually. Done for each mouse and class against each control
                significance = True
                for class_idx, class_label in enumerate(range(2)):
                
                    if decoding_key == 'f1':
                        perf = APdecoding_dict[mnum, 'f1_array'][:, class_idx]
                    elif decoding_key == 'accuracy':
                        perf = APdecoding_dict[mnum, 'accuracy_array'][:, class_idx]
                                
                    for control_idx, control in enumerate(control_names):
                        if 'shuffle' in control:
                            perf_control = APdecoding_dict[mnum, decoding_key + '_array_' + control][:, class_idx]
                        
                        elif 'CCA' in control:
                            if decoding_key == 'f1':
                                perf_control = f1_array_shift[midx][:, class_idx]
                            elif decoding_key == 'accuracy':
                                perf_control = accuracy_array_shift[midx][:, class_idx]
                            
                        # tstat, pval = scipy.stats.ttest_ind(perf, perf_control, equal_var=True, permutations=None, alternative='greater')
                        tstat, pval = scipy.stats.mannwhitneyu(perf, perf_control, use_continuity=False, alternative='greater')
                        #If even a single class vs control test is non-significant, mark it as non-significant
                        if pval > 0.05:
                            significance = False
                        
                if significance == True:
                    maxval = np.max(avg_by_mouse[midx])
                    xp = xpos_list[midx]-0.05
                    yp = maxval + 0.05
                    ax.text(xp, yp, '*', fontsize = 25, style='italic')
                    ymax = np.maximum(ymax, yp)
                    
                ## Two way anova, using variables "class type" (AP or No AP) and "analysis type" (shuffle, normal)
                class_type_label = []
                analysis_type_label = []
                value_list = [] #f1 or acc
                
                for class_idx, class_label in enumerate(range(2)):
                    
                    if decoding_key == 'f1':
                        perf = APdecoding_dict[mnum, 'f1_array'][:, class_idx]
                    elif decoding_key == 'accuracy':
                        perf = APdecoding_dict[mnum, 'accuracy_array'][:, class_idx]
                        
                    class_type_label.extend([class_label]*len(perf))
                    analysis_type_label.extend(["normal"]*len(perf))
                    value_list.extend(perf)
                                                    
                    for control_idx, control in enumerate(control_names):
                        if 'shuffle' in control:
                            perf_control = APdecoding_dict[mnum, decoding_key + '_array_' + control][:, class_idx]
                        
                        elif 'CCA' in control:
                            if decoding_key == 'f1':
                                perf_control = f1_array_shift[midx][:, class_idx]
                            elif decoding_key == 'accuracy':
                                perf_control = accuracy_array_shift[midx][:, class_idx]

                        class_type_label.extend([class_label]*len(perf_control))
                        analysis_type_label.extend([control]*len(perf_control))
                        value_list.extend(perf_control)
                        
                # all_data = np.hstack((shuffle_id, shuffle_dd, shuffle2_id, shuffle2_dd, f1_id, f1_dd))

                df = pd.DataFrame({'ctype':class_type_label, 
                                   'atype':analysis_type_label,
                                   "value":value_list})
                
                
                model = ols('value ~ C(ctype) + C(atype) + C(ctype):C(atype)', data=df).fit() 
                
                result = sm.stats.anova_lm(model, typ=2) 
                
                print(result)
                print("M%d"%mnum, " // Controls pval: ", result.loc[["C(atype)"]]['PR(>F)'].values[0], 
                      " // Class pval: ", result.loc[["C(ctype)"]]['PR(>F)'].values[0])
                



        
        ax.legend(fontsize=15, loc = 'upper right', frameon=False)
        ax.set_xlim([-shift-0.1, num_mice-(1-shift-0.1)])
        # ax.set_ylim([0, ax.get_ylim()[1]])
        yminlim = np.minimum(ax.get_ylim()[0], ymin)
        ymaxlim = np.maximum(ax.get_ylim()[1], ymax)
        ax.set_ylim([yminlim, ymaxlim])

        xpos_list = np.arange(num_mice)
        xlabels = ["M%d"%mnum for mnum in mouse_list]
        ax.set_xticks(xpos_list, xlabels, fontsize=fs)
        ax.set_ylabel("%s score"%decoding_label, fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)

        xlims = ax.get_xlim()
        ax.plot(xlims, [0.5, 0.5], '--k', alpha=0.5)
        
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        
        
        fig.tight_layout()
        fig_name = 'fig3_%s_by_mouse'%decoding_label
        save_figure(fig, fig_name)
        
        
        
        #VD vs DD summary plot
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=figsize_decoding_summary, num=fig_num)
        ax = ax[0,0]; fig_num += 1
        
        xpos_list = np.arange(2)
        shift = 0.1
        ymax = 0.5
        for mouse_type_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
            m_idxs = mtype_by_mouse == mouse_type_label
            if np.sum(m_idxs) == 0:
                continue
            for class_idx, class_label in enumerate(range(2)):
                perf_avg = np.average(avg_by_mouse[m_idxs, class_idx])
                perf_std = np.average(std_by_mouse[m_idxs, class_idx])
                ymax = np.maximum(ymax, perf_avg)

                xx = xpos_list[mouse_type_idx] - shift * (1-2*class_idx)
                label=pparam.AP_DECODING_LABELS[class_idx]
                if mouse_type_idx == 1:
                    label=None
                color = pparam.AP_DECODING_COLORS[class_idx]
                ax.errorbar(xx, perf_avg, yerr=perf_std, fmt='_', color=color, alpha=0.9, label=label, 
                                  markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
                
                if decoding_pair_idx == 0:
                    print(mouse_type_label, class_label, perf_avg)
                    

            


            
            for idx, control in enumerate(control_names):
                
                for class_idx, class_label in enumerate(range(2)):

                    perf_avg = np.average(control_data[control, decoding_key + '_avg'][m_idxs, class_idx])
                    perf_std = np.average(control_data[control, decoding_key + '_std'][m_idxs, class_idx])
                    
                    # print(mouse_type_label, control, class_label, perf_avg)

                    control_idx = control_label_idxs[idx]
                    label =pparam.AP_DECODING_LABELS[control_idx]
                    if mouse_type_idx != 0 or class_idx != 0:
                        label=None
                    color = pparam.AP_DECODING_COLORS[control_idx]
                    xx = xpos_list[mouse_type_idx] - shift * (1-2*class_idx)

                    ax.errorbar(xx, perf_avg, yerr=perf_std, fmt='_', color=color, alpha=0.9, label=label, 
                                      markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
                    
                    if decoding_pair_idx == 0:
                        print(mouse_type_label, class_label, perf_avg)

        #Plot significances with controls. Done for each mouse class against each control
        if len(control_names) > 0 and decoding_key == 'f1':
            for mouse_type_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
                m_idxs = mtype_by_mouse == mouse_type_label
                if np.sum(m_idxs) == 0:
                    continue
    
                significance = True
                for class_idx, class_label in enumerate(range(2)):
                
                    perf = avg_by_mouse[m_idxs, class_idx]
    
                    for control_idx, control in enumerate(control_names):
                        
                        perf_control = control_data[control, decoding_key + '_avg'][m_idxs, class_idx]
                        
                        # tstat, pval = scipy.stats.ttest_ind(perf, perf_control, equal_var=True, permutations=None, alternative='greater')
                        tstat, pval = scipy.stats.mannwhitneyu(perf, perf_control, use_continuity=False, alternative='greater')

                        
                        #If even a single class vs control test is non-significant, mark it as non-significant
                        if pval > 0.05:
                            significance = False
                        
                if significance == True:
                    maxval = np.max([np.average(avg_by_mouse[m_idxs, class_idx]) for class_idx in range(2)])
                    xp = xpos_list[mouse_type_idx]-0.05
                    yp = maxval + 0.05
                    ax.text(xp, yp, '*', fontsize = 25, style='italic')
                    ymax = np.maximum(ymax, yp)

        #Plot ID vs DD significances (merge both class labels together)       
        perf_by_mtype = []
        for mouse_type_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
            m_idxs = mtype_by_mouse == mouse_type_label
            perf_by_mtype.append(avg_by_mouse[m_idxs, :].ravel())
            
        # tstat, pval = scipy.stats.ttest_ind(perf_by_mtype[0], perf_by_mtype[1], equal_var=True, permutations=None, alternative='two-sided')
        tstat, pval = scipy.stats.wilcoxon(perf_by_mtype[0], perf_by_mtype[1], zero_method='wilcox', correction=False, alternative='two-sided')
        print("ID vs DD, Wilcoxon pval: %.4f"%pval)
        p00 = ymax + 0.05
        p11 = xpos_list[0]
        p10 = xpos_list[1]
        d0 = 0
        dp = 0.05
        ymax = np.maximum(ymax, p00)
        pf.draw_significance(ax, pval, p00, p10, p11, d0, dp, orientation='top', label_padding=0, thresholds = [0.05], fs=20)


            
        ## Two way anova, using variables "class type" (AP or No AP) and "analysis type" (shuffle, normal)
        if len(control_names) > 0 and decoding_key == 'f1':
            axon_type_label = []
            analysis_type_label = []
            value_list = []
            for mouse_type_idx, mouse_type_label in enumerate(pparam.MOUSE_TYPE_LABELS):
                m_idxs = mtype_by_mouse == mouse_type_label
                if np.sum(m_idxs) == 0:
                    continue
    
                for class_idx, class_label in enumerate(range(2)):
                
                    perf = avg_by_mouse[m_idxs, class_idx]
                    axon_type_label.extend([mouse_type_label]*len(perf))
                    analysis_type_label.extend(["normal"]*len(perf))
                    value_list.extend(perf)
                    
                    for control_idx, control in enumerate(control_names):
                        
                        perf_control = control_data[control, decoding_key + '_avg'][m_idxs, class_idx]
                        axon_type_label.extend([mouse_type_label]*len(perf_control))
                        analysis_type_label.extend([control]*len(perf_control))
                        value_list.extend(perf_control)
               
            df = pd.DataFrame({'axontype':axon_type_label, 
                               'atype':analysis_type_label,
                               "value":value_list})
            model = ols('value ~ C(axontype) + C(atype) + C(axontype):C(atype)', data=df).fit() 
            result = sm.stats.anova_lm(model, typ=2)
            print(result)
            print("ID vs DD ANOVA // Controls pval: ", result.loc[["C(atype)"]]['PR(>F)'].values[0], 
                  " // Class pval: ", result.loc[["C(axontype)"]]['PR(>F)'].values[0])


        
        ax.legend(fontsize=15, frameon=False)
        # ax.set_xlim([-shift-0.1, num_mice-(1-shift-0.1)])
        yminlim = np.minimum(ax.get_ylim()[0], ymin)
        ymaxlim = np.maximum(ax.get_ylim()[1], ymax)
        ax.set_ylim([yminlim, ymaxlim])
            
        ax.legend(fontsize=15, loc = 'upper right', frameon=False)

        xlabels = pparam.MOUSE_TYPE_LABELS
        ax.set_xticks(xpos_list, xlabels, fontsize=fs)
        ax.set_ylabel("%s score"%decoding_label, fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)   

        xlims = ax.get_xlim()
        ax.plot(xlims, [0.5, 0.5], '--k', alpha=0.5)
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        fig.tight_layout()
        fig_name = 'fig3_%s_summary'%decoding_label
        save_figure(fig, fig_name)    
    
    return





def fig3_F_and_3SI_A_B_belt_restriction_plots():
    exclusion_center_list = [250, 750, 1250]
    exclusion_interval_size = 500
    fig_name = 'fig3_belt_restriction'
    belt_restriction_plots(exclusion_center_list, exclusion_interval_size, fig_name)

def fig3SI_C():
    exclusion_center_list = [0, 500, 1000]
    exclusion_interval_size = 500
    fig_name = 'fig3SI_C'
    belt_restriction_plots(exclusion_center_list, exclusion_interval_size, fig_name)
    
    
def fig3SI_D():
    exclusion_center_list = [0, 500, 1000]
    exclusion_interval_size = 1000
    fig_name = 'fig3SI_D'
    belt_restriction_plots(exclusion_center_list, exclusion_interval_size, fig_name)

def belt_restriction_plots(
        exclusion_center_list = [250, 750, 1250],
        exclusion_interval_size = 500,
        fig_name = 'fig3_belt_restriction'
        ):
    
    ''' 
        Performs air puff decoding when part of the belt is excluded from the TCA analysis.
        exclusion_center_list: centers of the segment that is going to be excluded.
        exclusion_interval_size: total length of the excluded segment.
            e.g. if 250 is the center and 500 the interval size, a section from x=0 to x=500 will be excluded.
    '''
    

    
    mouse_list = np.arange(8)
    # mouse_list = [2,6]
    # mouse_list = [0,2,6,7]
    # mouse_list = [0,3,6,7]

    # mouse_list = [6]
    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':False,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':'.9',
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max',
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':20,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':20,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
    fig_num = 1
    fs=15
    plot_cut_pca = True
    


    exclusion_left_list = [(exclusion_center_list[i]-exclusion_interval_size/2)%pparam.MAX_POS for i in range(len(exclusion_center_list))]
    exclusion_right_list = [(exclusion_center_list[i]+exclusion_interval_size/2)%pparam.MAX_POS for i in range(len(exclusion_center_list))]   
    
    
    #### ANALYZE FOR EACH EXCLUSION INTERVAL ####
    shuffle_num = ap_decoding_param_dict['LDA_trial_shuffles']
    f1_dict_by_center_and_mouse = {}
    f1_shuffle_dict_by_center_and_mouse = {}

    for center_idx, center in enumerate(exclusion_center_list):
        start, end = exclusion_left_list[center_idx], exclusion_right_list[center_idx]
        
        ap_decoding_param_dict['exclude_positions'] = True
        ap_decoding_param_dict['pos_to_exclude_from'] = start
        ap_decoding_param_dict['pos_to_exclude_to'] = end
        
        
        # #Do analysis
        pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict)

        # #Put data in arrays
        APdecoding_dict = pipeline_output_dict['APdecoding_dict']
        for midx, mnum in enumerate(mouse_list): 
            f1_array = APdecoding_dict[mnum, 'f1_array']
            f1_dict_by_center_and_mouse[center, mnum] = f1_array
            f1_array = APdecoding_dict[mnum, 'f1_array_shuffle']
            f1_shuffle_dict_by_center_and_mouse[center, mnum] = f1_array
            
            
            
        ## OPTIONAL: PLOT CUT DATA (only for the first center and mouse) ##
        if plot_cut_pca == True:
            CCA_analysis_dict = pipeline_output_dict['CCA_analysis_dict']
            num_bins = CCA_analysis_dict['num_bins']
            
            APdecoding_dict = {}
            for midx,mnum in enumerate(mouse_list):
                if midx != 0:
                    continue
                fig = plt.figure(fig_num, figsize=(7,7)); fig_num += 1
                ax = plt.subplot(projection='3d')
                
                session_list = CCA_analysis_dict[mnum, 'session_list']
                pos_list = CCA_analysis_dict[mnum, 'pos']
                pca_list = CCA_analysis_dict[mnum, 'pca']
                        
                data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list)
                num_features, num_bins, total_trials = data_by_trial.shape
    
                #Limit position (if indicated)
                if ap_decoding_param_dict['exclude_positions'] == True:
                    positions = pos_by_trial[:,0] #Assumes all trials are binned using the same positions            
                    pos_bool_filtered_out = pf.get_idxs_in_periodic_interval(positions, ap_decoding_param_dict['pos_to_exclude_from'],
                                                   ap_decoding_param_dict['pos_to_exclude_to'], pparam.MAX_POS)
                    pos_bool_selected = np.invert(pos_bool_filtered_out)
                    data_by_trial = data_by_trial[:, pos_bool_selected, :]
                    pos_by_trial = pos_by_trial[pos_bool_selected, :]
                    
                    for sidx, snum in enumerate(np.unique(snum_by_trial)):
                        # print(data_by_trial.shape)
                        trials = snum_by_trial == snum                
                        pca = pf.flatten_warped_data(data_by_trial[:, :, trials])
                        pos = pf.flatten_warped_data(pos_by_trial[:, trials])
                        pos, pca, _ = pf.compute_average_data_by_position(pca, pos, position_bin_size=5)
                        cbar = [None, True][midx == 0 and sidx == 0]
                        pf.plot_pca_with_position(pca, pos, ax=ax, scatter=True, cbar=cbar, ms=100)
            ## OPTIONAL: PLOT CUT DATA ##

            
    #Get mouse-averaged results
    f1_avg_list = []; f1_std_list = []
    f1_shuffle_avg_list = []; f1_shuffle_std_list = []
    for center_idx, center in enumerate(exclusion_center_list):
        f1_all = np.array([f1_dict_by_center_and_mouse[center, mnum].ravel() for mnum in mouse_list]).ravel()
        f1_avg_list.append(np.average(f1_all))
        f1_all_std = np.array([np.std(f1_dict_by_center_and_mouse[center, mnum].ravel()) for mnum in mouse_list]).ravel()
        f1_std_list.append(np.average(f1_all_std))
        
        f1_all = np.array([f1_shuffle_dict_by_center_and_mouse[center, mnum].ravel() for mnum in mouse_list]).ravel()
        f1_shuffle_avg_list.append(np.average(f1_all))
        # f1_shuffle_std_list.append(np.std(f1_all))
        f1_shuffle_all_std = np.array([np.std(f1_shuffle_dict_by_center_and_mouse[center, mnum].ravel()) for mnum in mouse_list]).ravel()
        f1_shuffle_std_list.append(np.average(f1_shuffle_all_std))
        
    #### ANALYZE WITHOUT EXCLUSION, AS COMPARISON ####    
    ap_decoding_param_dict['exclude_positions'] = False
    pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict)
    
    #Store reference results
    #The standard deviation is the average of the standard deviation of each mouse
    APdecoding_dict_ref = pipeline_output_dict['APdecoding_dict']
    f1_ref_list = []
    f1_ref_std_list = []
    for midx, mnum in enumerate(mouse_list): 
        f1_array = APdecoding_dict_ref[mnum, 'f1_array']
        f1_ref_list.extend(f1_array.ravel())
        f1_ref_std_list.append(np.std(f1_array.ravel()))
    f1_ref_avg = np.average(f1_ref_list)
    f1_ref_std = np.average(f1_ref_std_list)
    
    ###### PLOT RESULTS ######
    
    #If the exclusion interval is larger than half the belt, plot the center of the inclusion interval instead
    #EG: if interval size is 1400 and period is 1500, the result only takes a small 100mm window into account. Use that as reference instead.
    
    if exclusion_interval_size < 750:
        center_list = exclusion_center_list
        plot_interval_size = exclusion_interval_size
        center_type = 'Exclusion'

    else:
        center_list = ((np.array(exclusion_center_list)+pparam.MAX_POS/2)%pparam.MAX_POS).astype(int)
        plot_interval_size = int(pparam.MAX_POS - exclusion_interval_size)
        center_type = 'Inclusion'
        
        
        
    #SI: results for each mouse separately
    fig, axs = plt.subplots(2, 4, figsize=(4 * 4, 3 * 2)); fig_num += 1
    
    for midx, mnum in enumerate(mouse_list):
        ax = axs.ravel()[mnum]

        #Plot main results
        f1_avg_mouse = [np.average(f1_dict_by_center_and_mouse[center, mnum]) for center in exclusion_center_list]
        f1_std_mouse = [np.std(f1_dict_by_center_and_mouse[center, mnum]) for center in exclusion_center_list]
        ax.errorbar(center_list, f1_avg_mouse, yerr=f1_std_mouse, fmt='_', color='black', alpha=0.9, label='Restricted', 
                              markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
        
        
        #Plot shuffle
        if shuffle_num > 0:
            f1_shuffle_avg_mouse = [np.average(f1_shuffle_dict_by_center_and_mouse[center, mnum]) for center in exclusion_center_list]
            f1_shuffle_std_mouse = [np.std(f1_shuffle_dict_by_center_and_mouse[center, mnum]) for center in exclusion_center_list]
            ax.errorbar(center_list, f1_shuffle_avg_mouse, yerr=f1_shuffle_std_mouse, fmt='_', color=pparam.AP_DECODING_COLORS[2], alpha=0.9, 
                        label=pparam.AP_DECODING_LABELS[2], markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
        
        #Plot reference
        f1_avg_ref_mouse = np.average(APdecoding_dict_ref[mnum, 'f1_array'])
        f1_std_ref_mouse = np.std(APdecoding_dict_ref[mnum, 'f1_array'])
        
        #Draw F1 reference average
        xlims = ax.get_xlim()
        xx = np.linspace(xlims[0], xlims[1])
        ax.plot(xx, [f1_avg_ref_mouse]*len(xx), '--', color='black', lw=3, alpha=0.5, label='Full')
        ax.fill_between(xx, [f1_avg_ref_mouse-f1_std_ref_mouse]*len(xx), [f1_avg_ref_mouse+f1_std_ref_mouse]*len(xx), color='gray', alpha=0.3)
        ax.set_xlim(xlims)

        
        #Plot significance with shuffle for each segment center
        ymax = ax.get_ylim()[1]
        if shuffle_num > 0:
            for center_idx, center in enumerate(exclusion_center_list):
                f1_all = np.array(f1_dict_by_center_and_mouse[center, mnum]).ravel()
                f1_all_shuffle = np.array(f1_shuffle_dict_by_center_and_mouse[center, mnum]).ravel()

                # tstat, pval = scipy.stats.ttest_ind(f1_all, f1_all_shuffle, equal_var=True, permutations=None, alternative='two-sided')                
                tstat, pval = scipy.stats.mannwhitneyu(f1_all, f1_all_shuffle, use_continuity=False, alternative='two-sided')

                if 0.05 > pval:
                    xp = center_list[center_idx]-50
                    yp = np.max(f1_avg_ref_mouse + f1_std_ref_mouse) + 0.05
                    ax.text(xp, yp, '*', fontsize = 25, style='italic')
                    ymax = np.maximum(ymax, yp)
                    
        
        #X axis
        ax.set_xticks(center_list, center_list)
        ax.tick_params(axis='x', labelsize=fs)
        ax.set_xlim([np.min(center_list)-150, np.max(center_list)+150])

        #Y axis
        ax.tick_params(axis='y', labelsize=fs+4)
        ax.set_ylim([0.2, ymax+0.1])


        #Both axis
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
    
            
        if midx == 0:
            ax.set_xlabel('%s segment center'%center_type, fontsize=fs+4)
            ax.set_ylabel('Avg F1 score', fontsize=fs+4)
            ax.legend(fontsize=fs-6, frameon=False)


        ax.set_title('M%d'%mnum, fontsize=fs+2)
        
    fig.suptitle(r'%s range:$\pm$ %d (mm)'%(center_type, plot_interval_size/2), fontsize=fs+3)

    fig.tight_layout()
    
    fig_name_by_mouse = '%s_by_mouse'%fig_name
    save_figure(fig, fig_name_by_mouse)
       
    
    
    
    
    
    #Fig4 subplot: mice-averaged results
    fig = plt.figure(fig_num, figsize=(5,5)); fig_num += 1
    ax = plt.gca()
    fs = 18


    #Plot exclusion results
    ax.errorbar(center_list, f1_avg_list, yerr=f1_std_list, fmt='_', color='black', alpha=0.9, label='Restricted', 
                              markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
    
    #Plot shuffle results
    if shuffle_num > 0:
        ax.errorbar(center_list, f1_shuffle_avg_list, yerr=f1_shuffle_std_list, fmt='_', color=pparam.AP_DECODING_COLORS[2], alpha=0.9, 
                    label=pparam.AP_DECODING_LABELS[2], markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
    
    #Plot reference
    xlims = ax.get_xlim()
    xx = np.linspace(xlims[0], xlims[1])
    ax.plot(xx, [f1_ref_avg]*len(xx), '--', color='black', lw=3, alpha=0.5, label='Full')
    ax.fill_between(xx, [f1_ref_avg-f1_ref_std]*len(xx), [f1_ref_avg+f1_ref_std]*len(xx), color='gray', alpha=0.3)
    ax.set_xlim(xlims)
    
    #Plot significance with shuffle for each segment center
    ymax = ax.get_ylim()[1]
    if shuffle_num > 0:
        for center_idx, center in enumerate(exclusion_center_list):
            f1_all = np.array([f1_dict_by_center_and_mouse[center, mnum].ravel() for mnum in mouse_list]).ravel()
            f1_all_shuffle = np.array([f1_shuffle_dict_by_center_and_mouse[center, mnum].ravel() for mnum in mouse_list]).ravel()

            # tstat, pval = scipy.stats.ttest_ind(f1_all, f1_all_shuffle, equal_var=True, permutations=None, alternative='two-sided')                
            tstat, pval = scipy.stats.mannwhitneyu(f1_all, f1_all_shuffle, use_continuity=False, alternative='two-sided')

            if 0.05 > pval:
                xp = center_list[center_idx]-25
                yp = np.max(f1_avg_ref_mouse + f1_std_ref_mouse) + 0.02
                ax.text(xp, yp, '*', fontsize = 25, style='italic')
                ymax = np.maximum(ymax, yp)
                    
    #X axis
    ax.set_xlabel('%s segment center'%center_type, fontsize=fs)
    ax.set_xticks(center_list, center_list)
    ax.tick_params(axis='x', labelsize=fs)
    ax.set_xlim([np.min(center_list)-150, np.max(center_list)+150])

    #Y axis
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_ylabel('Avg F1 score', fontsize=fs)
    ax.set_ylim([0.2, ymax+0.05])


    #Both axis
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    #Title
    ax.set_title(r'%s range:$\pm$ %d (mm)'%(center_type, plot_interval_size/2), fontsize=fs)
    ax.legend(fontsize=fs, frameon=False)

    
    #Figure params
    fig.tight_layout()
    fig_name_summary = '%s_summary'%fig_name
    save_figure(fig, fig_name_summary)

    


def fig4_A_session_comparisons():
    
    ''' Performs TCA across sessions for an animal.
        Step 1: perform PCA, limit to minimum possible of dimensions
        Step 2: align through CCA, so every dimension represents something similar about the data
        Step 3: split into trials through warping
        Step 4: TCA!
    '''
    
    mouse_list = np.arange(8)
    # mouse_list = [6]
    # mouse_list = [0,2,6,7]
    # mouse_list = [0,3,6,7]

    # mouse_list = [2,6]


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':'.9', #'.9'
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':1000,
        'pos_to_exclude_to':1500,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max',
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':20,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':20,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
        
    

    
    
    
    
    
    
    
    
    fig_num = 1
    fs = 15
    # # figsize_f1 = (6,3)
    # # figsize_f1_summary = (3,4)


    #Do analysis
    session_comparisons_list = ['airpuff', 'BT', 'BP', 'TP']
    # session_comparisons_list = ['BT', 'TP']

    session_comparisons_label_dict = {'airpuff':'BP-T', 'BT':'B-T', 'BP': 'B-P', 'TP':'T-P'}


    f1_array_list = []
    f1_array_shuffle_list = []
    
    for comp_idx, session_comparison in enumerate(session_comparisons_list):
        ap_decoding_param_dict['session_comparisons'] = session_comparison
    
        pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict)
        

        #Put data in arrays
        num_mice = len(mouse_list)
        LDA_trial_shuffles = ap_decoding_param_dict['LDA_trial_shuffles']
        APdecoding_dict = pipeline_output_dict['APdecoding_dict']
    
        f1_array = np.zeros((num_mice, 2)) #num mice X num classes
        # f1_array_std = np.zeros((num_mice, 2)) #num mice X num classes

        f1_array_shuffle = np.zeros((num_mice, LDA_trial_shuffles, 2))
        # f1_array_shuffle_std = np.zeros((num_mice, LDA_trial_shuffles, 2))
        
        for midx, mnum in enumerate(mouse_list): 
            f1_array[midx, :] = np.average(APdecoding_dict[mnum, 'f1_array'], axis=0)
            f1_array_shuffle[midx, :] = np.average(APdecoding_dict[mnum, 'f1_array_shuffle'], axis=0)


        f1_array_list.append(f1_array)
        f1_array_shuffle_list.append(f1_array_shuffle)
    
        
    #Mouse types
    mtype_by_mouse = np.array([pparam.MOUSE_TYPE_LABEL_BY_MOUSE[mnum] for mnum in mouse_list])
    mouse_types = pparam.MOUSE_TYPE_LABELS
    
    labels = [session_comparisons_label_dict[scomp] for scomp in session_comparisons_list]
    fig = plt.figure(fig_num, figsize=(5,3)); fig_num += 1
    ax = plt.gca()
    xpos_list = np.arange(len(session_comparisons_list))
    xgap = 0.2
    for mtype_idx, mtype in enumerate(mouse_types):
        midxs = np.where(mtype_by_mouse == mtype)[0]
        f1avg = [np.average(f1[midxs]) for f1 in f1_array_list]
        f1std = [np.std(f1[midxs]) for f1 in f1_array_list]
        ax.errorbar(xpos_list + xgap * (2*mtype_idx-1), f1avg, yerr=f1std, label=mtype, color=pparam.MOUSE_TYPE_COLORS[mtype_idx],
                    fmt='_', markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
        
        if LDA_trial_shuffles > 0:
        # for mtype_idx, mtype in enumerate(mouse_types):
            midxs = np.where(mtype_by_mouse == mtype)[0]
            f1avg = [np.average(f1[midxs]) for f1 in f1_array_shuffle_list]
            f1std = [np.std(f1[midxs]) for f1 in f1_array_shuffle_list]
            label = [None, pparam.AP_DECODING_LABELS[2]][mtype_idx == 0]
            ax.errorbar(xpos_list + xgap * (2*mtype_idx-1), f1avg, yerr=f1std, label=label, color=pparam.SHUFFLE_DEFAULT_COLOR,
                        fmt='_', markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
            
            
    ## Plot significances with controls. Done for each mouse type against their respective shuffle control
    overall_max_val = ax.get_ylim()[1]
    for session_comparison_idx, session_comparison in enumerate(session_comparisons_list):
        #If one of the two axon types is non-significant, don't plot the asterisk!
        significance = True 
        session_comparison_maxval = 0
        for mtype_idx, mtype in enumerate(mouse_types):
            midxs = np.where(mtype_by_mouse == mtype)[0]
            f1 = f1_array_list[session_comparison_idx][midxs].ravel()
            f1_control = f1_array_shuffle_list[session_comparison_idx][midxs]
            f1_control = np.average(f1_control, axis=1).ravel()
            # tstat, pval = scipy.stats.ttest_ind(f1, f1_control, equal_var=True, permutations=None, alternative='greater')
            tstat, pval = scipy.stats.mannwhitneyu(f1, f1_control, use_continuity=False, alternative='greater')

            if pval > 0.05:
                significance = False
            session_comparison_maxval = np.maximum(session_comparison_maxval, np.average(f1) + np.std(f1))

                
        if significance == True:
            xp = xpos_list[session_comparison_idx]-0.1
            yp = session_comparison_maxval + 0.01
            ax.text(xp, yp, '*', fontsize = 25, style='italic')
            overall_max_val = np.maximum(overall_max_val, yp)    
            
            
    ## Two way anova, using variables "class type" (AP or No AP) and "analysis type" (shuffle, normal)
    axon_type_label = []
    analysis_type_label = []
    value_list = []
    
    for session_comparison_idx, session_comparison in enumerate(session_comparisons_list):
        #If one of the two axon types is non-significant, don't plot the asterisk!
        significance = True 
        session_comparison_maxval = 0
        for mtype_idx, mtype in enumerate(mouse_types):
            midxs = np.where(mtype_by_mouse == mtype)[0]

            f1 = f1_array_list[session_comparison_idx][midxs].ravel()
            axon_type_label.extend([mtype]*f1.size)
            analysis_type_label.extend(['normal']*f1.size)
            value_list.extend(f1)
            
            
            f1_control = f1_array_shuffle_list[session_comparison_idx][midxs].ravel()
            axon_type_label.extend([mtype]*f1_control.size)
            analysis_type_label.extend(['shuffle']*f1_control.size)
            value_list.extend(f1_control)
    

        df = pd.DataFrame({'axontype':axon_type_label, 
                           'atype':analysis_type_label,
                           "value":value_list})
        model = ols('value ~ C(axontype) + C(atype) + C(axontype):C(atype)', data=df).fit() 
        result = sm.stats.anova_lm(model, typ=2)
        print(result)
        print("%s ANOVA // Controls pval: "%session_comparison, result.loc[["C(atype)"]]['PR(>F)'].values[0], 
              " // Class pval: ", result.loc[["C(axontype)"]]['PR(>F)'].values[0])

    
    xlabels = labels
    ax.set_xticks(xpos_list, xlabels, fontsize=fs)
    ax.set_ylabel("Avg F1 score", fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)  
    ax.tick_params(axis='x', labelsize=fs)  

    ymin = 0.2
    ymin = np.minimum(ymin, ax.get_ylim()[0])
    ax.set_ylim([ymin, overall_max_val])

    xlims = ax.get_xlim()
    ax.plot(xlims, [0.5, 0.5], '--k', alpha=0.5)
    
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.legend(fontsize=15, loc='lower right', frameon=False)
    
    
    save_figure(fig, 'fig4_session_comparisons')
    return




def fig4_C_D_E_distance_measures():
    compute_and_plot_trial_factor_distances(distance_comparison = 'session_type')
    
    
    
def fig4SI_G_H_I_distance_measures():
    compute_and_plot_trial_factor_distances(distance_comparison = 'airpuff')

    
def compute_and_plot_trial_factor_distances(distance_comparison = 'session_type'):
    ''' Calculate distances between trial factors '''

    mouse_list = np.arange(8)
    # mouse_list = [2,6]
    # mouse_list = [2,3,5,6]
    # mouse_list = [0,1,2,3]  
    # mouse_list = [4,5,6,7]

    # mouse_list = [0,6]

    #Distance parameters
    distance_comparison = 'session_type' #'session_type', 'airpuff', 'single_session'
    distance_shuffles = 10

    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':False,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':11,
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':1000,
        'pos_to_exclude_to':1500,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max',
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':10,
        
        ## LDA params ##
        'LDA_imbalance_prop':.6,
        'LDA_imbalance_repetitions':1,
        'LDA_trial_shuffles':0,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT'
        }
    
    mouse_list = preprocessing_param_dict['mouse_list']
    session_list = preprocessing_param_dict['session_list']

    ## TCA params ##
    

    # TCA parameters
    TCA_factors = ap_decoding_param_dict['TCA_factors']


    ########## STEP 1 - PCA ###########    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)

    ############## STEP 2: mCCA ############
    CCA_analysis_dict = perform_mCCA_on_pca_dict_param_dict(PCA_analysis_dict, cca_param_dict)
    
    
    num_bins = CCA_analysis_dict['num_bins']
    

    mouse_list = CCA_analysis_dict['mouse_list']

    num_bins = CCA_analysis_dict['num_bins']
    
    #Prepare distance matrices
    if distance_comparison == 'session_type':
        distance_type_names = pparam.SESSION_TYPE_LABELS
        distance_type_colors = pparam.SESSION_TYPE_COLORS
        distance_type_num = len(distance_type_names)
        distance_type_names_short = [n[0] for n in distance_type_names]
        
    elif distance_comparison == 'airpuff':
        distance_type_names = ['AP', 'No AP']
        distance_type_colors = pparam.SESSION_TYPE_COLORS[:2]
        distance_type_num = len(distance_type_names)
        distance_type_names_short = distance_type_names

    elif distance_comparison == 'single_session':
        distance_type_names = pparam.SESSION_NAMES
        cmap = mpl.cm.get_cmap('Set2')
        distance_type_colors = [cmap(num) for num in np.linspace(0,1,9)]
        distance_type_num = len(distance_type_names)
        distance_type_names_short = distance_type_names

       
    distance_dict_by_mouse_and_type = {(mnum, type1_idx, type2_idx):[] for mnum in mouse_list for type1_idx in range(distance_type_num) for type2_idx in range(distance_type_num)}
    distance_dict_by_mouse_and_type_shuffle = {(mnum, type1_idx, type2_idx):[] for mnum in mouse_list for type1_idx in range(distance_type_num) for type2_idx in range(distance_type_num)}

    for midx, mnum in enumerate(mouse_list):
        
        print('Performing TCA+LDA on M%d'%mnum)
        
        session_list = CCA_analysis_dict[mnum, 'session_list']
        pos_list = CCA_analysis_dict[mnum, 'pos']
        pca_list = CCA_analysis_dict[mnum, 'pca']
        # pca_list = CCA_analysis_dict[mnum, 'pca_unaligned']

        # session_list_to_decode = [snum for snum in session_list if snum in ap_decoding_param_dict['sessions_to_decode']]
                
        data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list)

        print(mnum, data_by_trial.shape)
        #Limit position (if indicated)
        if ap_decoding_param_dict['exclude_positions'] == True:
            positions = pos_by_trial[:,0] #Assumes all trials are binned using the same positions
            pos_bool = np.invert(np.bitwise_and(positions > ap_decoding_param_dict['pos_to_exclude_from'], positions < ap_decoding_param_dict['pos_to_exclude_to']))
            data_by_trial = data_by_trial[:, pos_bool, :]
            pos_by_trial = pos_by_trial[pos_bool, :]
            
        #Selecting trials to decode
        # ap_decoding_param_dict['session_comparisons'] = 'airpuff'
        trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, ap_decoding_param_dict['session_comparisons'])


        if TCA_factors == 'max':
            TCA_dimensions = data_by_trial.shape[0]

        TCA_counter_total = 0
        for TCA_on_LDA_counter in range(ap_decoding_param_dict['TCA_on_LDA_repetitions']):
            # #Step 4: TCA 
            KTensor = APfuns.perform_TCA(data_by_trial, TCA_dimensions, ap_decoding_param_dict['TCA_replicates'], 
                                         ap_decoding_param_dict['TCA_method'], ap_decoding_param_dict['TCA_convergence_attempts'])
            trial_factors = KTensor[2]
            
            TCA_counter_total += 1
            ############## STARTING THE DISTANCE MEASUREMENTS ###################   
            
            fig_num = 1
            fs = 15
            trial_type_by_trial = np.array([pparam.SESSION_TYPE_LABELS.index(pparam.SESSION_LABEL_BY_SNUM[snum]) for snum in snum_by_trial])
            AP_by_trial = pparam.get_AP_labels_from_snum_by_trial(snum_by_trial)
            # stype_by_trial = stype_by_trial[trials_to_keep]

            if distance_comparison == 'session_type':
                label_by_trial = trial_type_by_trial
            
            elif distance_comparison == 'airpuff':
                label_by_trial = AP_by_trial
            
            elif distance_comparison == 'single_session':
                label_by_trial = snum_by_trial

            #Center trial factors
            trial_factors, _, _ = pf.normalize_data(trial_factors, axis=0)
            
            # trial_factors_clipped = np.zeros(trial_factors.shape)
            
            
            #Get distance from each trial to cluster average
            trial_type_unique = np.unique(label_by_trial)
            trial_type_num = len(trial_type_unique)
            num_trials = len(label_by_trial)
            for i in range(1+distance_shuffles):
                
                if i==0:
                    label_by_trial_current = np.copy(label_by_trial)
                else:
                    shuffle_idxs = np.random.choice(range(len(label_by_trial)), size=len(label_by_trial), replace=False)
                    label_by_trial_current = label_by_trial[shuffle_idxs]
                    label_by_trial_current = np.random.choice(np.unique(label_by_trial), size=len(label_by_trial), replace=True)
                    
                    avg_samples_per_class = int(np.floor(len(label_by_trial)/len(np.unique(label_by_trial))))
                    label_by_trial_new = np.zeros(len(label_by_trial), dtype=type(label_by_trial[0]))
                    for dtype_idx,dtype in enumerate(np.unique(label_by_trial)):
                        if dtype != np.unique(label_by_trial)[-1]:
                            label_by_trial_new[((dtype_idx)*avg_samples_per_class):((dtype_idx+1)*avg_samples_per_class)] = dtype
                        else:
                            label_by_trial_new[((dtype_idx)*avg_samples_per_class):] = dtype
                    shuffle_idxs = np.random.choice(range(len(label_by_trial_new)), size=len(label_by_trial_new), replace=False)
                    label_by_trial_current = label_by_trial_new[shuffle_idxs]

                    

                #Method 1: compare to centroid   
                distance_array = np.zeros((num_trials, trial_type_num))
                class_centers = np.zeros((trial_type_num, TCA_dimensions))
                for trial_type_idx, trial_type in enumerate(trial_type_unique):
                    trial_bool = label_by_trial_current == trial_type
                    trial_factor_type = trial_factors[trial_bool]
                    trial_type_center = np.average(trial_factor_type, axis=0)
                    class_centers[trial_type_idx] = trial_type_center
    
                for trial in range(num_trials):
                    for trial_type_index, trial_type in enumerate(trial_type_unique):
                        tf = trial_factors[trial]
                        center = class_centers[trial_type_index]
                        d = np.linalg.norm(tf-center)
                        distance_array[trial, trial_type_index] = d
                    
                # #Take out outliers (IMPROVE!)
                distances_ordered = np.sort(distance_array.ravel())
                max_clip_distance = distances_ordered[int(distances_ordered.size * 0.95)] #Get 95th percentile
                distance_array = np.clip(distance_array, 0, max_clip_distance)
                
                # #Z-score
                # if i == 0:
                #     dmean = np.mean(distance_array)
                # dmean = np.mean(distance_array)
                # distance_array = distance_array/dmean #+ 1
                distance_array = 1 + (distance_array - np.average(distance_array))/np.std(distance_array)
            
                #Get distances of each trial to each 
                for center_type_idx, center_type in enumerate(trial_type_unique):
                    for trial in range(num_trials):
                        trial_type = label_by_trial_current[trial]
                        d = distance_array[trial, center_type_idx]
                        if i == 0:
                            distance_dict_by_mouse_and_type[mnum, trial_type, center_type_idx].append(d)
                        else:
                            distance_dict_by_mouse_and_type_shuffle[mnum, trial_type, center_type_idx].append(d)                     

        
    mtype_by_mouse = np.array([pparam.MOUSE_TYPE_LABEL_BY_MOUSE[mnum] for mnum in mouse_list])
    mtype_unique = np.sort(np.unique(mtype_by_mouse))[::-1]
    midxs_by_type = []
    for mtype in mtype_unique:
        midxs_by_type.append([midx for midx in range(len(mouse_list)) if mtype_by_mouse[midx] == mtype])
        
    #Get within-between array
    distance_dict_by_mouse_dtype_and_belonging = {(mnum, dtype, belonging_idx):[] for mnum in mouse_list
                                                 for dtype in range(distance_type_num)
                                                 for belonging_idx in range(2)}
    distance_dict_by_mouse_dtype_and_belonging_shuffle = {(mnum, dtype, belonging_idx):[] for mnum in mouse_list
                                                 for dtype in range(distance_type_num)
                                                 for belonging_idx in range(2)}
    
    #Get this information by mouse type
    distance_dict_by_mtype_dtype_and_belonging = {(mtype, dtype, belonging_idx):[] for mtype in mtype_unique
                                                     for dtype in range(distance_type_num)
                                                     for belonging_idx in range(2)}
    distance_dict_by_mtype_dtype_and_belonging_shuffle = {(mtype, dtype, belonging_idx):[] for mtype in mtype_unique
                                                 for dtype in range(distance_type_num)
                                                 for belonging_idx in range(2)}
    
    
    #Difference between outside and inside
    distance_diff_dict_by_mtype_dtype = {}
    distance_diff_dict_by_mtype_dtype_shuffle = {}

    for midx, mnum in enumerate(mouse_list):
        mtype = mtype_by_mouse[midx]
        for dtype1 in range(distance_type_num):
            ddiff_array = np.zeros(len(distance_dict_by_mouse_and_type[mnum, dtype1, dtype1]))
            ddiff_array_shuffle = np.zeros(len(distance_dict_by_mouse_and_type_shuffle[mnum, dtype1, dtype1]))

            for dtype2 in range(distance_type_num):
                belonging_idx = 1 - (dtype1 == dtype2) #0: within, 1:between

                dlist = distance_dict_by_mouse_and_type[mnum, dtype1, dtype2]
                distance_dict_by_mouse_dtype_and_belonging[mnum, dtype1, belonging_idx].extend(list(dlist))
                distance_dict_by_mtype_dtype_and_belonging[mtype, dtype1, belonging_idx].extend(list(dlist))
                
                dlist_shuffle = distance_dict_by_mouse_and_type_shuffle[mnum, dtype1, dtype2]
                distance_dict_by_mouse_dtype_and_belonging_shuffle[mnum, dtype1, belonging_idx].extend(list(dlist_shuffle))
                distance_dict_by_mtype_dtype_and_belonging_shuffle[mtype, dtype1, belonging_idx].extend(list(dlist_shuffle))
                
                if belonging_idx != 0:
                    ddiff_array += (np.array(dlist) - np.array(distance_dict_by_mouse_and_type[mnum, dtype1, dtype1]))
                    ddiff_array_shuffle += (np.array(dlist_shuffle) - np.array(distance_dict_by_mouse_and_type_shuffle[mnum, dtype1, dtype1]))

            ddiff_array /= (distance_type_num-1)
            ddiff_array_shuffle /= (distance_type_num-1)
            
            if (mtype, dtype1) not in distance_diff_dict_by_mtype_dtype.keys():
                distance_diff_dict_by_mtype_dtype[mtype, dtype1] = (list(ddiff_array))
                distance_diff_dict_by_mtype_dtype_shuffle[mtype, dtype1] = (list(ddiff_array_shuffle))
            else:
                distance_diff_dict_by_mtype_dtype[mtype, dtype1].extend(list(ddiff_array))
                distance_diff_dict_by_mtype_dtype_shuffle[mtype, dtype1].extend(list(ddiff_array_shuffle))

    #Dist diffs overall
    distance_diff_dict_by_mtype = {}
    distance_diff_dict_by_mtype_shuffle = {}
    for mtype_idx, mtype in enumerate(mtype_unique):
        distance_diff_dict_by_mtype[mtype] = []
        distance_diff_dict_by_mtype_shuffle[mtype] = []
        for dtype in range(distance_type_num):
            distance_diff_dict_by_mtype[mtype].extend(distance_diff_dict_by_mtype_dtype[mtype, dtype])
            distance_diff_dict_by_mtype_shuffle[mtype].extend(distance_diff_dict_by_mtype_dtype_shuffle[mtype, dtype])

    #Get colormap array
    distance_matrix_by_mouse_avg = np.zeros((len(mouse_list), distance_type_num, distance_type_num))
    distance_matrix_by_mouse_std = np.zeros((len(mouse_list), distance_type_num, distance_type_num))
    distance_matrix_by_mouse_avg_shuffle = np.zeros((len(mouse_list), distance_type_num, distance_type_num))

    for midx, mnum in enumerate(mouse_list):
        for dtype1 in range(distance_type_num):
            for dtype2 in range(distance_type_num):
                dlist = distance_dict_by_mouse_and_type[mnum, dtype1, dtype2]
                distance_matrix_by_mouse_avg[midx, dtype1, dtype2] = np.average(dlist)
                distance_matrix_by_mouse_std[midx, dtype1, dtype2] = np.std(dlist)/np.sqrt(len(dlist))

                dlist = distance_dict_by_mouse_and_type_shuffle[mnum, dtype1, dtype2]
                distance_matrix_by_mouse_avg_shuffle[midx, dtype1, dtype2] = np.average(dlist)                

    #Plot 1 - colormap by mouse
    fig_cmap, axs_cmap = plt.subplots(2, 4, num=fig_num, figsize=(7,4)); fig_num += 1
    fs = 12
    vmin = np.min(distance_matrix_by_mouse_avg)
    vmax = np.max(distance_matrix_by_mouse_avg)
    
    for midx, mnum in enumerate(mouse_list):
        ax = axs_cmap.ravel()[midx]
        darray = distance_matrix_by_mouse_avg[midx]
        ax.imshow(darray, cmap=pparam.DISTANCE_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_yticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_title('M%d'%mnum, fontsize = fs+3, pad=0)
    
    axs_cmap[0,0].set_ylabel('From', fontsize=fs+4)
    axs_cmap[1,0].set_xlabel('To', fontsize=fs+4)
    
    ### Add colorbar in the plot
    # fig_cmap.subplots_adjust(right=0.90)
    # cbar_ax = fig_cmap.add_axes([1.05, 0.17, 0.035, 0.7])    
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    # cbar = fig_cmap.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=pparam.DISTANCE_CMAP), cax=cbar_ax, orientation='vertical', fraction=0.01)
    # cbar.ax.set_ylabel('Trial distance (norm.)', fontsize=fs+7, rotation=270, labelpad=25)    
    # cbar.ax.tick_params(axis='both', which='major', labelsize=fs+5)    
    
    fig_cmap.subplots_adjust(wspace=0, hspace=0)
    fig_cmap.tight_layout()
    fig_name = 'FigSI_distances_colormap_by_mouse_%s'%distance_comparison
    save_figure(fig_cmap, fig_name)
   
    #Plot 2 - colormap by mouse type
    fig_cmap, axs_cmap = plt.subplots(1, 2, num=fig_num, figsize=(5,3)); fig_num += 1
    fs = 16
    
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        ax = axs_cmap.ravel()[midx_list_idx]
        darray = distance_matrix_by_mouse_avg[midx_list]
        darray = np.average(darray, axis=0)
        ax.imshow(darray, cmap=pparam.DISTANCE_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_yticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_title(r"$\bf{%s}$"%mtype_unique[midx_list_idx], fontsize = fs+3, pad=10)
        for dtype_idx in range(darray.shape[0]):
            ax.add_patch(Rectangle((dtype_idx-0.5, dtype_idx-0.5), 1, 1, fill=False, edgecolor='black', lw=1))
    

    axs_cmap[0].set_ylabel(r"$\bf{}$From", fontsize=fs+4)
    axs_cmap[0].set_xlabel(r"$\bf{}$To", fontsize=fs+4)  
    
    fig_cmap.subplots_adjust(wspace=0, hspace=0)
    fig_cmap.tight_layout()
    
    fig_name = 'Fig4_distances_colormap_by_mtype_%s'%distance_comparison
    save_figure(fig_cmap, fig_name)
    
    ### Make colorbar separately
    plt.figure(fig_num); fig_num += 1
    fig_cbar = plt.gcf()    

    vmin_cbar, vmax_cbar = np.around([vmin, vmax], decimals=1)
    cbar_ticks = np.linspace(vmin_cbar, vmax_cbar, num=4)
    cbar_ticks = np.around(cbar_ticks, decimals=2)
    vmin_cbar = cbar_ticks[0]; vmax_cbar = cbar_ticks[-1]
    
    cbar = pf.add_distance_cbar(fig_cbar, pparam.DISTANCE_CMAP, vmin = vmin_cbar, vmax = vmax_cbar, fs=fs, 
                                cbar_label = '', 
                                cbar_kwargs = {'fraction':0.555, 'pad':0.04, 'aspect':15}
                                )
    cbar.ax.set_yticks(cbar_ticks)
    cbar.ax.tick_params(axis='y', labelsize=25) 
    cbar.ax.set_ylabel('Trial distance (norm.)', fontsize=fs+7, rotation=270, labelpad=25)
    
    fig_name = 'Fig4_distances_colormap_colorbar_%s'%distance_comparison
    save_figure(fig_cbar, fig_name)
    
    
    #Plot 2.2 - Within distance time evolution
    fs = 20
    fig = plt.figure(fig_num); fig_num+=1
    ax = plt.gca()
    xx = range(len(np.unique(label_by_trial_current)))
    
    delta_vals = {} #{(mtype, dtype): [delta of avg]}
    delta_ypos_vals = {dtype:[] for dtype in trial_type_unique[:-1]} #{(dtype): [delta of avg]}
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        mtype = mtype_unique[midx_list_idx]
        color = pparam.MOUSE_TYPE_COLORS_BY_MTYPE[mtype]
        
        avg_list = np.zeros(len(trial_type_unique))
        std_list = np.zeros(len(trial_type_unique))
        for trial_type_idx, trial_type in enumerate(trial_type_unique):
            dds = distance_dict_by_mtype_dtype_and_belonging[mtype, trial_type_idx, 0]
            avg_list[trial_type_idx] = np.average(dds)
            std_list[trial_type_idx] = np.std(dds)/np.sqrt(len(dds))
        ax.plot(xx, avg_list, 'o-', lw=3, color=color, label=mtype)
        ax.fill_between(xx, avg_list-std_list, avg_list+std_list, color=color, alpha=0.5)
        
        
         
            
        for dtype_idx, dtype in enumerate(trial_type_unique[:-1]):

            #Delta stat significances (compare the differences with probe)
            delta_of_avg_list = []
            for midx in midx_list:
                mnum = mouse_list[midx]
                d = distance_dict_by_mouse_dtype_and_belonging[mnum, dtype, 0]
                dprobe = distance_dict_by_mouse_dtype_and_belonging[mnum, trial_type_unique[-1], 0]
                delta_of_avg = np.average(d) - np.average(dprobe)
                # deltas = [d1-d2 for d1 in d for d2 in dprobe]
                # delta_of_avg = np.average(deltas)
                delta_of_avg_list.append(delta_of_avg)
            delta_vals[mtype,dtype] = delta_of_avg_list
       
    #Plot delta stat significances (compare the differences with probe)

    for dtype_idx, dtype in enumerate(trial_type_unique[:-1]):
        mtype = mtype_unique[midx_list_idx]
        delta_list_id = delta_vals[MOUSE_TYPE_LABELS[0], dtype]
        delta_list_dd = delta_vals[MOUSE_TYPE_LABELS[1], dtype]
        # tstat, pval = scipy.stats.ttest_ind(delta_list_id, delta_list_dd, equal_var=True, permutations=None, alternative='greater')
        tstat, pval = scipy.stats.mannwhitneyu(delta_list_id, delta_list_dd, use_continuity=False, alternative='greater')

        p00 = np.max(delta_ypos_vals[dtype])
        p10 = xx[dtype_idx]; p11 = p10
        d0 = 0; dp = 0; label_padding = 0
        pf.draw_significance(ax, pval, p00, p10, p10, d0, dp, orientation='top', label_padding=label_padding, thresholds = [0.05], fs=fs+15)
     
            
         
        print(delta_list_id, delta_list_dd)
        print("pval", dtype, pval)

    xlabels = [distance_type_names[i] for i in trial_type_unique]
    ax.set_xticks(xx, xlabels, fontsize=fs)
    ax.set_ylabel('Within class distances (norm.)', fontsize=fs-5)
    ax.tick_params(axis='y', which='major', labelsize=fs)
    ax.legend(fontsize=fs, frameon=False)
    # ax.set_ylim([0.5, ax.get_ylim()[1]])
    # ax.set_title("", fontsize=fs)
    
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    fig.tight_layout()     
    fig_name = 'FigSI_within_distances_by_dtype_%s'%(distance_comparison)
    save_figure(fig, fig_name)
    
    
    #Plot 3: Shuffle colormap
    fig_cmap, axs_cmap = plt.subplots(1, 2, num=fig_num, figsize=(5,3)); fig_num += 1
    fs = 18
    
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        ax = axs_cmap.ravel()[midx_list_idx]
        darray = distance_matrix_by_mouse_avg_shuffle[midx_list]
        darray = np.average(darray, axis=0)
        ax.imshow(darray, cmap=pparam.DISTANCE_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_yticks(range(distance_type_num), distance_type_names_short, fontsize=fs)
        ax.set_title(r"$\bf{%s}$"%mtype_unique[midx_list_idx], fontsize = fs+3, pad=10)
        for dtype_idx in range(darray.shape[0]):
            ax.add_patch(Rectangle((dtype_idx-0.5, dtype_idx-0.5), 1, 1, fill=False, edgecolor='black', lw=1))
    
    axs_cmap[0].set_ylabel('From', fontsize=fs+4)
    axs_cmap[0].set_xlabel('To', fontsize=fs+4)
    
    fig_cmap.subplots_adjust(wspace=0, hspace=0)
    fig_cmap.tight_layout()
    
    fig_name = 'Fig4_distances_colormap_by_mtype_shuffle_%s'%distance_comparison
    save_figure(fig_cmap, fig_name)
    
    
    #Plot 4: Barplots, distance by type, includes shuffle
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        mtype = mtype_unique[midx_list_idx]
        ## Within-without (by class and with shuffle)
        fig = plt.figure(fig_num, figsize=(5,4)); fig_num+=1
        ax = plt.gca()
        all_bars_width = 0.75
        num_of_bars = 4
        barwidth = all_bars_width/(num_of_bars)
        xpos_list = range(len(np.unique(label_by_trial_current)))
        for trial_type_idx, trial_type in enumerate(trial_type_unique):
            for belonging_idx, belonging in enumerate(pparam.DISTANCE_LABELS):
                for analysis_type_idx, analysis_type in enumerate(['Normal', 'Shuffle']):
                    if analysis_type_idx == 0:
                        dlist = distance_dict_by_mtype_dtype_and_belonging[mtype_unique[midx_list_idx], trial_type_idx,  belonging_idx]
                        color = distance_type_colors[trial_type_idx]
                        pltlabel = [None, belonging][trial_type_idx==0]
                    else:
                        dlist = distance_dict_by_mtype_dtype_and_belonging_shuffle[mtype_unique[midx_list_idx], trial_type_idx,  belonging_idx]
                        color = pparam.SHUFFLE_DEFAULT_COLOR
                        pltlabel = [None, 'Shuffle'][trial_type_idx==0 and analysis_type_idx == 0]
                    
                    xpos = xpos_list[trial_type_idx] - all_bars_width/2 + barwidth/2 + barwidth * belonging_idx + 2 * barwidth * analysis_type_idx
                    
                    avg = np.average(dlist)
                    err = np.std(dlist)/np.sqrt(len(dlist))
                    # err = np.std(dlist)
                    alpha = [1, 0.5][belonging_idx == 1]
                    ax.bar(xpos, avg, width=barwidth, alpha=alpha, edgecolor=None, color=color, label=pltlabel)
                    ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, 
                                color=color, alpha=alpha)
                    
                    
                
        xlabels = [distance_type_names_short[i] for i in trial_type_unique]
        ax.set_xticks(xpos_list, xlabels, fontsize=fs)
        ax.set_ylabel('Trial factor distances', fontsize=fs)
        ax.tick_params(axis='y', which='major', labelsize=fs)
        ax.legend(fontsize=fs-3, frameon=False)
        # ax.set_ylim([0.5, ax.get_ylim()[1]])
        ax.set_title(r"$\bf{%s}$"%mtype_unique[midx_list_idx], fontsize=fs)
        

        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        
        fig.tight_layout() 
        
        fig_name = 'FigSI_distances_by_dtype_%s_%s'%(mtype, distance_comparison)
        save_figure(fig, fig_name)
    
    #Plot 5: Distance difference barplots by dtype
    fs = 20
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        mtype = mtype_unique[midx_list_idx]
        ## Within-without (by class and with shuffle)
        fig = plt.figure(fig_num, figsize=(4,5)); fig_num+=1
        ax = plt.gca()
        all_bars_width = .8
        num_of_bars = 2
        barwidth = all_bars_width/(num_of_bars)
        xpos_list = range(len(np.unique(label_by_trial_current)))
        for trial_type_idx, trial_type in enumerate(trial_type_unique):
            for analysis_type_idx, analysis_type in enumerate(['Data', 'Shuffle']):
                if analysis_type_idx == 0:
                    dlist = distance_diff_dict_by_mtype_dtype[mtype, trial_type_idx]
                else:
                    dlist = distance_diff_dict_by_mtype_dtype_shuffle[mtype, trial_type_idx]
                    
                color = [distance_type_colors[trial_type_idx], pparam.SHUFFLE_DEFAULT_COLOR][analysis_type_idx]
                pltlabel = [None, analysis_type][trial_type_idx==0]
                xpos = xpos_list[trial_type_idx] - all_bars_width/2 + barwidth/2 + barwidth * analysis_type_idx
                avg = np.average(dlist)
                err = np.std(dlist)/np.sqrt(len(dlist))
                # err = np.std(dlist)
                alpha = [1, 0.5][analysis_type_idx]
                alpha = 0.9
                ax.bar(xpos, avg, width=barwidth, alpha=alpha, edgecolor=None, color=color, label=pltlabel)
                ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, 
                            color=color, alpha=alpha)
                
                
            #Statistical significances
            d1 = distance_diff_dict_by_mtype_dtype[mtype, trial_type_idx]
            d2 = distance_diff_dict_by_mtype_dtype_shuffle[mtype, trial_type_idx]
            # tstat, pval = scipy.stats.ttest_ind(d1, d2, equal_var=True, permutations=None, alternative='two-sided')
            tstat, pval = scipy.stats.mannwhitneyu(d1, d2, use_continuity=False, alternative='two-sided')

            xtext = xpos_list[trial_type_idx] - all_bars_width/2 + barwidth/2
            ytext = np.average(d1)+np.std(d1)/np.sqrt(len(d1)) + 0.025*(ax.get_ylim()[1] - ax.get_ylim()[0])
            p00 = ytext; p10 = xtext; p11 = xtext + barwidth
            d0 = 0; dp = 0; label_padding = -0.075
            pf.draw_significance(ax, pval, p00, p10, p11, d0, dp, orientation='top', label_padding=label_padding, thresholds = [0.05], fs=fs+15)
         
        xlabels = [distance_type_names_short[i] for i in trial_type_unique]
        ax.set_xticks(xpos_list, xlabels, fontsize=fs)
        ax.set_ylabel('Within-between distance difference', fontsize=fs-4)
        ax.tick_params(axis='y', which='major', labelsize=fs)
        ax.legend(fontsize=fs-3, frameon=False)
        # ax.set_ylim([0.5, ax.get_ylim()[1]])
        ax.set_title(r"$\bf{%s}$"%mtype, fontsize=fs)

        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        
        fig.tight_layout()
        
        fig_name = 'FigSI_distances_within_between_diff_by_dtype_%s_%s'%(mtype, distance_comparison)
        save_figure(fig, fig_name)
        
        print(mtype, midx_list)

    
    #Plot 6: Distance difference barplots overall
    fs = 25
    fig = plt.figure(fig_num, figsize=(4,6)); fig_num+=1
    ax = plt.gca()
    all_bars_width = .8
    num_of_bars = 2
    barwidth = all_bars_width/(num_of_bars)
    xpos_list = range(len(midxs_by_type))
    for midx_list_idx, midx_list in enumerate(midxs_by_type):
        mtype = mtype_unique[midx_list_idx]
        for analysis_type_idx, analysis_type in enumerate(['Data', 'Shuffle']):
            if analysis_type_idx == 0:
                dlist = distance_diff_dict_by_mtype[mtype]
            else:
                dlist = distance_diff_dict_by_mtype_shuffle[mtype]
                
            color = [pparam.MOUSE_TYPE_COLORS_BY_MTYPE[mtype], pparam.SHUFFLE_DEFAULT_COLOR][analysis_type_idx]
            pltlabel = [None, analysis_type][midx_list_idx==0]
            xpos = xpos_list[midx_list_idx] - all_bars_width/2 + barwidth/2 + barwidth * analysis_type_idx
            avg = np.average(dlist)
            err = np.std(dlist)/np.sqrt(len(dlist))
            # err = np.std(dlist)
            alpha = [1, 0.5][analysis_type_idx]
            alpha = 0.9
            ax.bar(xpos, avg, width=barwidth, alpha=alpha, edgecolor=None, color=color, label=pltlabel)
            ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, 
                        color=color, alpha=alpha)
            
            
        #Statistical significances
        d1 = distance_diff_dict_by_mtype[mtype]
        d2 = distance_diff_dict_by_mtype_shuffle[mtype]
        # tstat, pval = scipy.stats.ttest_ind(d1, d2, equal_var=True, permutations=None, alternative='two-sided')
        tstat, pval = scipy.stats.mannwhitneyu(d1, d2, use_continuity=False, alternative='two-sided')
        xtext = xpos_list[midx_list_idx] - all_bars_width/2 + barwidth/2
        ytext = np.average(d1)+np.std(d1)/np.sqrt(len(d1)) + 0.025*(ax.get_ylim()[1] - ax.get_ylim()[0])
        p00 = ytext; p10 = xtext; p11 = xtext + barwidth
        d0 = 0; dp = 0; label_padding = -0.03
        pf.draw_significance(ax, pval, p00, p10, p11, d0, dp, orientation='top', label_padding=label_padding, thresholds = [0.05], fs=fs+15)
     
     
    xlabels = [r"$\bf{}$%s"%mtype for mtype in mtype_unique]
    ax.set_xticks(xpos_list, xlabels, fontsize=fs)
    ax.set_ylabel('Within-between distance difference', fontsize=fs-3)
    ax.tick_params(axis='y', which='major', labelsize=fs)
    ax.legend(fontsize=fs-3, frameon=False)
    # ax.set_ylim([0.5, ax.get_ylim()[1]])
    # ax.set_title(r"$\bf{%s}$"%mtype_unique[midx_list_idx], fontsize=fs)

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    fig.tight_layout()
    fig_name = 'Fig4_within_between_diff_by_mtype_%s'%distance_comparison
    save_figure(fig, fig_name)
    
    return





def fig5_A_F1_by_TCA_dimension():
    
    dimension_list = list(np.arange(3,16))
    # dimension_list = [5,6]
    vary_cca_and_tca_together = False
    vary_only_tca = True
    compute_f1_by_cca_and_tca_dimension(dimension_list, vary_cca_and_tca_together, vary_only_tca)
    
    
def fig3SI_F1_by_CCA_and_TCA_dimension_and_mouse():
    dimension_list = list(np.arange(3,16))
    vary_cca_and_tca_together = True
    vary_only_tca = False
    compute_f1_by_cca_and_tca_dimension(dimension_list, vary_cca_and_tca_together, vary_only_tca)
    

def compute_f1_by_cca_and_tca_dimension(
        dimension_list = list(np.arange(3,16)),
        vary_cca_and_tca_together = False,
        vary_only_tca = False,
        ):
    ''' 
        Plots F1 results based on dimensionality 
            dimension_list: list of dimensions to be analyzed
            vary_cca_and_tca_together: if true, TCA dim is always set to the same as CCA
            vary_only_tca: if True, CCA is fixed to the max dimension and only TCA is modified

    '''
    
    mouse_list = np.arange(8)
    # mouse_list = [6]
    # mouse_list = [0,2,6,7]
    # mouse_list = [0,3,6,7]

    # mouse_list = [4, 6]


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':False,
        'num_components':'all'
        }
    
    cca_param_dict = {
        'CCA_dim':None,
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':1000,
        'pos_to_exclude_to':1500,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':None,
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':20,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':20,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
    

    num_dims = len(dimension_list)
    num_mice = len(mouse_list)
    f1_array_by_dim_by_mouse = np.zeros((num_mice, num_dims, num_dims)) - 1 #mouse_idx, CCA dim, TCA dim
    f1_array_by_dim_by_mouse_std = np.zeros((num_mice, num_dims, num_dims)) - 1
    f1_shuffle_array_by_dim_by_mouse = np.zeros((num_mice, num_dims, num_dims)) - 1#mouse_idx, CCA dim, TCA dim
    f1_shuffle_array_by_dim_by_mouse_std = np.zeros((num_mice, num_dims, num_dims)) - 1
    
    f1_overall_min = 0.5
    f1_overall_max = 0.5
    for CCA_dim_idx, CCA_dim in enumerate(dimension_list):
        for TCA_dim_idx, TCA_dim in enumerate(dimension_list):
            
            if CCA_dim < TCA_dim: #TCA dimension should always be equal or larger than CCA
                continue
            elif vary_cca_and_tca_together == True and CCA_dim != TCA_dim: #When only varying CCA, ignore all TCA dimensions different than CCA dimensions
                continue
            elif vary_only_tca == True and CCA_dim != dimension_list[-1]: #CCA is fixed to the highest dimension that TCA can reach
                continue
                
            print('CCA dim: %d, TCA dim: %d' %(CCA_dim, TCA_dim))
            cca_param_dict['CCA_dim'] = CCA_dim
            ap_decoding_param_dict['TCA_factors'] = TCA_dim

            #Do analysis
            pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict)
            APdecoding_dict = pipeline_output_dict['APdecoding_dict']
            for midx, mnum in enumerate(mouse_list):
                f1_array = np.average(APdecoding_dict[mnum, 'f1_array'])
                f1_array_by_dim_by_mouse[midx, CCA_dim_idx, TCA_dim_idx] = f1_array
                # f1_array_by_dim_by_mouse_std[midx, CCA_dim_idx, TCA_dim_idx] = np.std(APdecoding_dict[mnum, 'f1_array'])
                f1_array_by_dim_by_mouse_std[midx, CCA_dim_idx, TCA_dim_idx] = np.std(APdecoding_dict[mnum, 'f1_array'])/np.sqrt(APdecoding_dict[mnum, 'f1_array'].size)

                f1_overall_min = np.minimum(f1_overall_min, np.min(f1_array))
                f1_overall_max = np.maximum(f1_overall_max, np.max(f1_array))
                
                f1_shuffle_array_by_dim_by_mouse[midx, CCA_dim_idx, TCA_dim_idx] = np.average(APdecoding_dict[mnum, 'f1_array_shuffle'])
                # f1_shuffle_array_by_dim_by_mouse_std[midx, CCA_dim_idx, TCA_dim_idx] = np.std(APdecoding_dict[mnum, 'f1_array_shuffle'])
                f1_shuffle_array_by_dim_by_mouse_std[midx, CCA_dim_idx, TCA_dim_idx] = np.std(APdecoding_dict[mnum, 'f1_array_shuffle'])/np.sqrt(APdecoding_dict[mnum, 'f1_array_shuffle'].size)

                print(f1_array_by_dim_by_mouse[midx, CCA_dim_idx, TCA_dim_idx])
                
    f1_array_by_dim_by_mouse[f1_array_by_dim_by_mouse < 0] = np.nan
    f1_array_by_dim_by_mouse_std[f1_array_by_dim_by_mouse_std < 0] = np.nan
    f1_shuffle_array_by_dim_by_mouse[f1_shuffle_array_by_dim_by_mouse < 0] = np.nan
    f1_shuffle_array_by_dim_by_mouse_std[f1_shuffle_array_by_dim_by_mouse_std < 0] = np.nan
    
    f1_by_dim_dict = {
        'mouse_list':mouse_list,
        'dimension_list':dimension_list,
        'f1_array_by_dim_by_mouse': f1_array_by_dim_by_mouse,
        'f1_array_by_dim_by_mouse_std': f1_array_by_dim_by_mouse_std,
        'f1_shuffle_array_by_dim_by_mouse': f1_shuffle_array_by_dim_by_mouse,
        'f1_shuffle_array_by_dim_by_mouse_std': f1_shuffle_array_by_dim_by_mouse_std,
        'vary_cca_and_tca_together':vary_cca_and_tca_together,
        'vary_only_tca':vary_only_tca
        }
    
    np.save(OUTPUT_PATH + "f1_by_dim_dict.npy", f1_by_dim_dict)
    
    plot_f1_by_dimension("f1_by_dim_dict.npy")
    
def plot_f1_by_dimension(f1_by_dim_dict=None):
    ''' Plots results from "compute_f1_by_cca_and_tca_dimension".
        f1_by_dim_dict: if None, the default one is loaded. If a string, load a dict with that name from "OUTPUT" folder. Otherwise, it's assumed to be the correct dictionary
    '''
    
    if f1_by_dim_dict is None:
        f1_by_dim_dict = np.load(OUTPUT_PATH + "f1_by_dim_dict.npy", allow_pickle=True)[()]
    elif type(f1_by_dim_dict) is str:
        f1_by_dim_dict = np.load(OUTPUT_PATH + f1_by_dim_dict, allow_pickle=True)[()]

    mouse_list = f1_by_dim_dict['mouse_list']; num_mice = len(mouse_list)
    dimension_list = f1_by_dim_dict['dimension_list']; num_dims = len(dimension_list)
    f1_array_by_dim_by_mouse = f1_by_dim_dict['f1_array_by_dim_by_mouse']
    f1_array_by_dim_by_mouse_std = f1_by_dim_dict['f1_array_by_dim_by_mouse_std']
    f1_shuffle_array_by_dim_by_mouse = f1_by_dim_dict['f1_shuffle_array_by_dim_by_mouse']
    f1_shuffle_array_by_dim_by_mouse_std = f1_by_dim_dict['f1_shuffle_array_by_dim_by_mouse_std']
    vary_cca_and_tca_together = f1_by_dim_dict['vary_cca_and_tca_together']
    vary_only_tca = f1_by_dim_dict['vary_only_tca']
    vary_only_tca = True
    
    f1_overall_min = np.minimum(np.nanmin(f1_array_by_dim_by_mouse), np.nanmin(f1_shuffle_array_by_dim_by_mouse))
    f1_overall_max = np.maximum(np.nanmax(f1_array_by_dim_by_mouse), np.nanmax(f1_shuffle_array_by_dim_by_mouse))
    are_there_shuffles = not np.isnan(f1_shuffle_array_by_dim_by_mouse).all()
    

    ## Select linear data per mouse, taking the cases where TCA_dim=CCA_dim.
    ## EXCEPTION: when "vary_only_tca" is True, take the case of increasing TCA dimension instead
    f1_array_by_mouse = np.zeros((num_mice, num_dims))
    f1_array_by_mouse_std = np.zeros((num_mice, num_dims))
    f1_shuffle_array_by_mouse = np.zeros((num_mice, num_dims))
    f1_shuffle_array_by_mouse_std = np.zeros((num_mice, num_dims))
    for midx, mnum in enumerate(mouse_list):
        if vary_only_tca == False:
            f1_array_by_mouse[midx] = f1_array_by_dim_by_mouse[midx].diagonal()
            f1_array_by_mouse_std[midx] = f1_array_by_dim_by_mouse_std[midx].diagonal()
            f1_shuffle_array_by_mouse[midx] = f1_shuffle_array_by_dim_by_mouse[midx].diagonal()
            f1_shuffle_array_by_mouse_std[midx] = f1_shuffle_array_by_dim_by_mouse_std[midx].diagonal()
        else:
            f1_array_by_mouse[midx] = f1_array_by_dim_by_mouse[midx, -1]
            f1_array_by_mouse_std[midx] = f1_array_by_dim_by_mouse_std[midx, -1]
            f1_shuffle_array_by_mouse[midx] = f1_shuffle_array_by_dim_by_mouse[midx, -1]
            f1_shuffle_array_by_mouse_std[midx] = f1_shuffle_array_by_dim_by_mouse_std[midx, -1]

    #PLOTS
    fig_num = 1
    fs = 15
    
    
    #Plot 1: full TCA and CCA hyperparameter search

    if vary_cca_and_tca_together == False and vary_only_tca == False:
        

        # fig = plt.figure(fig_num); fig_num += 1
        fig, axs = plt.subplots(2, 4, figsize=(4 * 4, 3 * 2)); fig_num += 1
        
        for midx, mnum in enumerate(mouse_list):
            
            ax = axs.ravel()[mnum]
            f1_array = f1_array_by_dim_by_mouse[midx]
            # ax.imshow(f1.T)
            cmap = mpl.cm.get_cmap('viridis') # jet doesn't have white color
            cmap.set_bad('w') # default value is 'k'
            img = ax.imshow(f1_array.T, cmap=cmap, interpolation='nearest', vmin=f1_overall_min, vmax=f1_overall_max)
    
            ax.set_xticks(range(num_dims), dimension_list, fontsize=fs)
            ax.set_xlabel('CCA dimension', fontsize=fs+4)
            ax.xaxis.tick_top()            
            ax.xaxis.set_label_position('top')             
            ax.set_yticks(range(num_dims), dimension_list, fontsize=fs-3)
            ax.yaxis.tick_right()            
            ax.set_ylabel('TCA factors', fontsize=fs+4)
            ax.set_title("M%d"%mouse_list[midx], fontsize = fs+4)
    
            if midx == len(mouse_list)-1:
                cbar = fig.colorbar(img, shrink = 0.96)
                cbar_ticks = np.around(np.linspace(f1_overall_min, f1_overall_max, num=6, dtype=float), decimals=2)
                cbar.ax.set_yticks(cbar_ticks, cbar_ticks, fontsize=fs)
                cbar.ax.set_ylabel('Average F1', fontsize=fs)             
                
    
        fig.tight_layout()
    
    
        save_figure(fig, 'fig3SI_f1_by_cca_and_tca_dimension')
        
        
    #####################################################
    
    #Plot 2: plot the relationship with dimension keeping CCa and TCA equal.
    #If "only vary TCA" option is enabled, then plot evolution with TCA for every mouse
    # fig = plt.figure(fig_num, figsize=(8,6)); fig_num += 1
    fig, axs = plt.subplots(2, 4, figsize=(3 * 4, 2 * 2)); fig_num += 1
    


    
    for midx, mnum in enumerate(mouse_list):
        ax = axs.ravel()[mnum]
        f1_list = f1_array_by_mouse[midx]
        f1_std_list = f1_array_by_mouse_std[midx]
        xpos = dimension_list
        ax.plot(xpos, f1_list, lw=3)
        ax.fill_between(xpos, f1_list-f1_std_list, f1_list+f1_std_list, alpha=0.3)

        
        if are_there_shuffles == True:
            f1_list_shuffle = f1_shuffle_array_by_mouse[midx]
            f1_std_list_shuffle = f1_shuffle_array_by_mouse_std[midx]
            
            ax.plot(xpos, f1_list_shuffle, color=pparam.SHUFFLE_DEFAULT_COLOR, alpha=0.6)
            ax.fill_between(xpos, f1_list_shuffle-f1_std_list_shuffle, f1_list_shuffle+f1_std_list_shuffle, color=pparam.SHUFFLE_DEFAULT_COLOR, alpha=0.3)

        # ax.set_ylim([f1_overall_min-0.1, f1_overall_max+0.1])
        ax.set_xticks(xpos, xpos, fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        ax.set_title("M%d"%mouse_list[midx], fontsize = fs+4)

        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        if midx == 0:
            ax.set_ylabel("Average F1", fontsize=fs)
        if midx == 4:
            ax.set_xlabel("Dimension", fontsize=fs)
            
        
    if vary_only_tca == False:
        plt.suptitle('Varying CCA and TCA dimension together', fontsize=fs+4)
    else:
        plt.suptitle('Varying TCA dimension with $N_{CCA} =$%d'%dimension_list[-1], fontsize=fs+4)

    fig.tight_layout()

    save_figure(fig, 'fig5SI_f1_by_tca_dimension_and_mouse')

    #####################################################
    
    #Plot 3: keeping TCA and CCA at the same dimension, averaged across mice   
    fig = plt.figure(fig_num, figsize=(6,4)); fig_num += 1
    ax = plt.gca()
    fs = 20
    
    mtype_by_mouse = np.array([pparam.MOUSE_TYPE_LABEL_BY_MOUSE[mnum] for mnum in mouse_list])
    mtype_unique = np.unique(mtype_by_mouse)
    for mouse_type_idx, mouse_type_label in enumerate(mtype_unique):
        m_idxs = mtype_by_mouse == mouse_type_label
        if np.sum(m_idxs) == 0:
            continue
        
        f1_avg = np.average(f1_array_by_mouse[m_idxs], axis=0)
        # f1_std = np.std(f1_array_by_mouse[m_idxs], axis=0)
        f1_std = np.std(f1_array_by_mouse[m_idxs], axis=0)/np.sqrt(f1_array_by_mouse[m_idxs].shape[0])

        xpos = dimension_list
        label=mouse_type_label
        color = pparam.MOUSE_TYPE_COLORS_BY_MTYPE[mouse_type_label]
        ax.plot(xpos, f1_avg, label=label, color=color, lw=3)
        ax.fill_between(xpos, f1_avg-f1_std, f1_avg+f1_std, color=color, alpha=0.3)

        
        if are_there_shuffles == True and mouse_type_idx == 0:

            f1_list_shuffle = np.average(f1_shuffle_array_by_mouse, axis=0)
            # f1_std_list_shuffle = np.std(f1_shuffle_array_by_mouse, axis=0)
            f1_std_list_shuffle = np.std(f1_shuffle_array_by_mouse, axis=0)/np.sqrt(f1_shuffle_array_by_mouse.shape[0])

            ax.plot(xpos, f1_list_shuffle, color=pparam.SHUFFLE_DEFAULT_COLOR, lw=3, alpha=1., label = pparam.AP_DECODING_LABELS[2])
            ax.fill_between(xpos, f1_list_shuffle-f1_std_list_shuffle, f1_list_shuffle+f1_std_list_shuffle, color=pparam.SHUFFLE_DEFAULT_COLOR, alpha=0.3)


        # ax.set_ylim([f1_overall_min-0.1, f1_overall_max+0.1])
        ax.set_xticks(xpos, xpos, fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        ax.legend(fontsize=fs+3, frameon=False)

        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        if mouse_type_idx == 0:
            ax.set_ylabel("Average F1", fontsize=fs)
            ax.set_xlabel("Dimension", fontsize=fs)


        
    if vary_only_tca == False:
        ax.set_title('Varying CCA and TCA dimension together', fontsize=fs-3)
    else:
        ax.set_title('Varying TCA dimension with $N_{CCA} =$%d'%dimension_list[-1], fontsize=fs)
    fig.tight_layout()

    save_figure(fig, 'fig5A_f1_by_tca_dimension')







    return

def get_position_decoding_weights_from_pca_dict(PCA_analysis_dict, predictor_name = 'Wiener'):
    ''' Input: 
            PCA_analysis_dict from "perform_pca_on_multiple_mice_param_dict" function
        Output: 
            pca_position_weights_dict: keys are (mnum, snum), values of size "number of pca features" with the position weight of each PCA dimension
            cell_position_weights_dict: keys are (mnum, snum), values of size "number of cells" with the position weight of each cell
    '''
    error_type = 'sse' #Not used, so doesn't mater
    mouse_list = PCA_analysis_dict['mouse_list']
    session_list = PCA_analysis_dict['session_list']
    PCA_dict = PCA_analysis_dict['PCA_dict']
    components_dict = PCA_analysis_dict['components_dict']
    
    #Get position prediction coefficients
    # pca_position_weights = np.zeros((num_mice, num_pca_plot_dims))
    pca_position_weight_dict = {} #(mnum, snum) : [pca1_w, pca2_w, ...]
    cell_position_weight_dict = {} #(mnum, snum) : [cell1_w, cell2_w, ...]
    
    for midx, mnum in enumerate(mouse_list):
        for sidx, snum in enumerate(session_list): 
            position, pca = PCA_dict[mnum, snum]
            pca_components = components_dict[mnum, snum]
            # print(pca_components.shape)
            # print(np.around(np.dot(pca_components, pca_components.T), decimals=0))
            # raise TypeError

            pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=0, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                predictor_name=predictor_name, predictor_default=None, return_error=error_type)
            position_coef = np.abs(predictor.model.coef_)

            pca_position_w = np.average(position_coef, axis=0) #Average over the two predictive dimensions
            # pca_position_w, _, _ = pf.normalize_data(pca_position_w, axis=0)

            #Get PCA contribution
            pca_position_weight_dict[mnum, snum] = pca_position_w
            
            # #Get cell contribution           
            pca_components = components_dict[mnum, snum] # (n_components, n_features)
            cell_components = scipy.linalg.inv(pca_components)#  (n_features, n_components)
            cell_components = np.abs(cell_components)
            cell_position_w = np.dot(cell_components, pca_position_w.T).squeeze()
            cell_position_weight_dict[mnum, snum] = cell_position_w
            
            
            
            
            
            
            
            
    return pca_position_weight_dict, cell_position_weight_dict
            
            
    
def get_airpuff_decoding_weights_from_analysis_dicts(PCA_analysis_dict, CCA_analysis_dict, APdecoding_dict):
    
    full_session_list = PCA_analysis_dict['param_dict']['session_list']
    mouse_list = PCA_analysis_dict['mouse_list']
    components_dict = PCA_analysis_dict['components_dict']
    
    # cca_airpuff_weight_array = np.zeros((num_mice, num_features))
    cca_airpuff_weight_dict = {} #(mnum) : [pca1_weight, pca2_weight, ...]
    pca_airpuff_weight_dict = {} #(mnum, snum) : [pca1_weight, pca2_weight, ...]
    cell_airpuff_weight_dict = {} #(mnum, snum) : [cell1_weight, cell2_weight, ...]
    for midx, mnum in enumerate(mouse_list):
        
        msession_list = CCA_analysis_dict[mnum, 'session_list']
        num_features = CCA_analysis_dict[mnum, 'pca'][0].shape[0]

        weights = APdecoding_dict[mnum, 'APdecoding_weights'] #size (tca reps) X (TCA dimensions)
        weights = np.average(weights, axis=0) #Average over TCA reps

        # weights = weights/np.sum(weights)
        # weights, _, _ = pf.normalize_data(weights, axis=0)

        # cca_airpuff_weight_array[midx] = weights
        cca_airpuff_weight_dict[mnum] = weights

        mCCA = CCA_analysis_dict[mnum, 'mCCA_instance']
        best_space = CCA_analysis_dict[mnum, 'best_space']
        # pca_weights_by_session = np.zeros((num_sessions, num_features))
        skipped_sessions = []
        for sidx, snum in enumerate(full_session_list):
            if snum not in msession_list:
                skipped_sessions.append(snum)
                continue
            sidx_for_this_mouse = msession_list.index(snum)
            
            
            pca_weights = mCCA.align(weights[:, np.newaxis], best_space, sidx_for_this_mouse)
            # pca_weights_by_session[sidx,:] = np.squeeze(pca_weights)
            pca_weights, _, _ = pf.normalize_data(pca_weights, axis=0)
            pca_airpuff_weight_dict[mnum, snum] = pca_weights
            
            pca_components = components_dict[mnum, snum][:num_features]
            pca_components = np.abs(pca_components)
            pca_weights = pca_weights.reshape((1,pca_weights.size))
                        
            cell_weights = np.dot(pca_weights, pca_components).squeeze()
            cell_airpuff_weight_dict[mnum, snum] = cell_weights
            
        #Deal with repeated sessions easy way !!!! ATTENTION: HACK SOLUTION TO REPEATED SESSIONS CHANGE !!!!!
        for snum in skipped_sessions:
            reps = pparam.SESSION_REPEATS[mnum]
            for rep_tuple in reps:
                if snum in rep_tuple:
                    repeated_session = rep_tuple[1 - rep_tuple.index(snum)] #Choose the other session which has been done
                    pca_airpuff_weight_dict[mnum, snum] = pca_airpuff_weight_dict[mnum, repeated_session]
                    cell_airpuff_weight_dict[mnum, snum] = cell_airpuff_weight_dict[mnum, repeated_session]

    return cca_airpuff_weight_dict, pca_airpuff_weight_dict, cell_airpuff_weight_dict


def normalize_weight_dict(weight_dict):
    for k in weight_dict.keys():
        weights = weight_dict[k]
        weights = np.absolute(weights)
        weights, _, _ = pf.normalize_data(weights, axis=0)
        weight_dict[k] = weights
    return weight_dict

def fig5_B_C_D_and_fig5SI_A_B_C_D_E_decoding_weight_plots():
    ''' Compare the contribution of each cell to position and airpuff decoding '''
    
    mouse_list = np.arange(8)
    # mouse_list = [6]
    # mouse_list = [0,2,6,7]
    # mouse_list = [0,3,6,7]

    # mouse_list = [0, 6]
    # mouse_list = np.arange(4,8)


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    cca_param_dict = {\
        'CCA_dim':'.9', #'.9'
        'return_warped_data':True,
        'return_trimmed_data':False,
        'sessions_to_align':'all',
        'shuffle':False
        }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':1000,
        'pos_to_exclude_to':1500,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max',
        'TCA_replicates':11,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':20,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':0,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
    #Prediction parameters
    predictor_name = 'Wiener'
    
    fig_num = 1;
    fs = 15
    fs_title = fs+4
    
    num_mice = len(mouse_list)
    session_list = preprocessing_param_dict['session_list']
    weight_types = pparam.WEIGHT_TYPES_LABELS
    weight_types_colors = pparam.WEIGHT_TYPES_COLORS

    #Do analysis
    pipeline_output_dict = APdecoding_pipeline(preprocessing_param_dict, cca_param_dict, ap_decoding_param_dict, force_recalculation=True)
    
    #Load relevant data
    PCA_analysis_dict = pipeline_output_dict['PCA_analysis_dict']
    APdecoding_dict = pipeline_output_dict['APdecoding_dict']
    CCA_analysis_dict = pipeline_output_dict['CCA_analysis_dict']

    PCA_dict = PCA_analysis_dict['PCA_dict']
    all_session_dims = [PCA_dict[mnum, snum][1].shape[0] for mnum in mouse_list for snum in session_list]
    max_dims = np.max(all_session_dims)
    
    #Get weights
    pca_position_weight_dict, cell_position_weight_dict = get_position_decoding_weights_from_pca_dict(PCA_analysis_dict, predictor_name)
    cca_airpuff_weight_dict, pca_airpuff_weight_dict, cell_airpuff_weight_dict = get_airpuff_decoding_weights_from_analysis_dicts(
        PCA_analysis_dict, CCA_analysis_dict, APdecoding_dict)
    
    #Normalize and store
    pca_position_weight_dict = normalize_weight_dict(pca_position_weight_dict)
    cell_position_weight_dict = normalize_weight_dict(cell_position_weight_dict)
    pca_airpuff_weight_dict = normalize_weight_dict(pca_airpuff_weight_dict)
    cell_airpuff_weight_dict = normalize_weight_dict(cell_airpuff_weight_dict)

    
    pca_weight_dicts = [pca_position_weight_dict, pca_airpuff_weight_dict]
    cell_weight_dicts = [cell_position_weight_dict, cell_airpuff_weight_dict]

    # # Get number of features by mouse and session:
    dimension_by_mouse_list = [CCA_analysis_dict[mnum, 'pca'][0].shape[0] for mnum in mouse_list]

    
    #Get position prediction coefficients by cell type
    place_cell_position_weight_dict = {0:[], 1:[]}
    place_cell_airpuff_weight_dict = {0:[], 1:[]}
    place_cell_weight_dicts = [place_cell_position_weight_dict, place_cell_airpuff_weight_dict]
    
    significance_counters_array = np.array([[0,0],[0,0]], dtype=float) #First axis is position/airpuff, second is non-place cell/place cell
    
    session_counter = 0.
    for weight_type_idx, place_cell_weight_dict in enumerate(weight_types):
        weight_dict = cell_weight_dicts[weight_type_idx]
        weight_dict_by_type = place_cell_weight_dicts[weight_type_idx]
        
        for midx, mnum in enumerate(mouse_list):
            
            for sidx, snum in enumerate(session_list): 
            
                

                cell_weights = weight_dict[mnum, snum]
                # cell_weights = np.absolute(cell_weights)
                # cell_weights, _, _ = pf.normalize_data(cell_weights, axis=0)
    
                place_cell_bool = pf.load_place_cell_boolean(mnum, snum, criteria='dombeck').astype(bool)
                if len(np.unique(place_cell_bool)) == 1:
                    continue
                cell_type_weight_list = []
                for cell_type in np.unique(place_cell_bool):
                    weights_by_cell_type = cell_weights[place_cell_bool == cell_type]
                    cell_type_weight_list.append(list(weights_by_cell_type))
                    
                    # weight_dict_by_type[cell_type].extend(list(weights_by_cell_type)) #Add all of them
                    weight_dict_by_type[cell_type].append(np.average((weights_by_cell_type))) #Only add session average
                    
                for cell_type in np.unique(place_cell_bool):
                    sig_alternative = ['less', 'greater'][1-cell_type]
                    # tstat, pval = scipy.stats.ttest_ind(cell_type_weight_list[0], cell_type_weight_list[1], 
                                                        # equal_var=True, permutations=None, alternative=sig_alternative)
                    tstat, pval = scipy.stats.mannwhitneyu(cell_type_weight_list[0], cell_type_weight_list[1], use_continuity=False, alternative=sig_alternative)

                    pval_label = pf.get_significance_label(pval, thresholds = [0.05, 0.005, 0.0005], asterisk=True, ns=True)

                    if pval_label != 'ns':
                        significance_counters_array[weight_type_idx, int(cell_type)] += 1

                session_counter += 1
                
    session_counter = session_counter/2            
    for weight_type_idx, weight_type in enumerate(weight_types):
        significance_counter_pair = significance_counters_array[weight_type_idx]
        nplace_cell_prop = significance_counter_pair[0]/session_counter
        place_cell_prop = significance_counter_pair[1]/session_counter
        print('%d%% of sessions had significant non-place cell contribution to %s' %(100*nplace_cell_prop, weight_types[weight_type_idx]))  
        print('%d%% of sessions had significant place cell contribution to %s' %(100*place_cell_prop, weight_types[weight_type_idx]))            



    ### ~~~~~~~~~~~~ PLOTS ~~~~~~~~~~~~~~~~



    ########## PLACE CELL DECODING STATISTICS PLOTS ##############
    fig = plt.figure(fig_num, figsize=(5,5)); fig_num += 1
    ax = plt.gca()
    xpos_list = [0, 1, 2.5,3.5]
    box_width = .6
    for weight_type_idx, place_cell_weight_dict in enumerate(weight_types):
        weight_dict_by_type = place_cell_weight_dicts[weight_type_idx]
        
        
        
        # tstat, pval = scipy.stats.ttest_ind(weight_dict_by_type[0], weight_dict_by_type[1], 
                                        # equal_var=True, permutations=None, alternative='less')
        
        bp_list = []
        for cell_type in np.arange(2):
            xpos = xpos_list[cell_type + 2*weight_type_idx]
            avg = np.average(weight_dict_by_type[cell_type])
            pltlabel = [None, cell_type][weight_type_idx == 0]
            # err = np.std(weight_dict_by_type[cell_type])
            # ax.errorbar(xpos, avg, yerr=None, fmt='_', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CELL_TYPE_COLORS[cell_type], alpha=1)
            # ax.errorbar(xpos, avg, yerr=err, fmt='_', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CELL_TYPE_COLORS[cell_type], alpha=0.8)
            bplot = ax.boxplot([weight_dict_by_type[cell_type]], positions = [xpos], 
                                showfliers=False,
                                vert=True,
                                patch_artist=True,
                                widths = box_width,
                                labels=[pltlabel])  
            for box in bplot['boxes']:
                box.set_facecolor(pparam.CELL_TYPE_COLORS[cell_type])
                # box.set_facecolor(weight_types_colors[weight_type_idx])
                box.set_alpha(0.5)
            bp_list.append(bplot)
                
         
        #Draw significance
        pval = pf.get_pval_greater_or_lesser(weight_dict_by_type[0], weight_dict_by_type[1])
        pval = np.abs(pval)
        x1 = xpos_list[0 + 2*weight_type_idx]
        x2 = xpos_list[1 + 2*weight_type_idx]
        yy = ax.get_ylim()[1]
        pf.draw_significance(ax, pval, yy, x1, x2, 0.05, 0.05, orientation='top', thresholds = [0.05], fs=25)

                            
    pos_ticks = [np.average(xpos_list[0:2]), np.average(xpos_list[2:4])]
    ax.set_xticks(pos_ticks, weight_types, fontsize=fs_title)
    ax.tick_params(axis='y', labelsize=fs_title)
    ax.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0]], pparam.CELL_TYPE_LABELS, fontsize=fs, loc='upper right', frameon=False)

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    ax.set_ylabel("Prediction weights", fontsize=fs_title)
    ax.set_xlabel("Prediction type", fontsize=fs_title)

    # ax.set_title("Decoding %s weights by cell type"%weight_type_label, fontsize=fs+2)
    fig.tight_layout()
    # ax.set_xlabel("PCA dimension", fontsize=fs_title)
    save_figure(fig, 'fig5_decoding_weights_by_cell_type')


    
    
    
    
    ########## CCA DIMENSION DECODING PLOTS ##############
    
    #Plot CCA weights per mouse
    fig, axs = plt.subplots(2, 4, figsize=(3 * 4, 2 * 2)); fig_num += 1

    for midx, mnum in enumerate(mouse_list):           
        ax = axs.ravel()[mnum]
        
        ws_by_dim = cca_airpuff_weight_dict[mnum]
        xx = np.arange(1, len(ws_by_dim)+1)

        ax.plot(xx, ws_by_dim, 'o')
        
        
        # ax.set_xticks(xx, xx, fontsize=fs)
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.legend(fontsize=fs-3, frameon=False)
    
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        if midx == 0:
            ax.set_ylabel("Decoding weights", fontsize=fs_title)
            ax.set_xlabel("Latent dimension", fontsize=fs_title)
    
    
            
        ax.set_title('M%d'%mnum, fontsize=fs_title)
    fig.suptitle('Air puff decoding weights by CCA dimension', fontsize=fs)
    fig.tight_layout()

    save_figure(fig, 'fig5SI_cca_airpuff_decoding_weights_by_mouse')
    
    
    
    
    
    #Plot cca airpuff weights per dimension
    all_cca_dims = [len(cca_airpuff_weight_dict[mnum]) for mnum in mouse_list]
    min_cca_dim = np.min(all_cca_dims)    
    
    fig = plt.figure(fig_num, figsize=(6,4)); fig_num += 1
    ax = plt.gca()
    avg_array = np.zeros(min_cca_dim)
    std_array = np.zeros(min_cca_dim)
    for ccafeature in range(min_cca_dim):
        weights = [cca_airpuff_weight_dict[mnum][ccafeature] for mnum in mouse_list]
        avg_array[ccafeature] = np.average(weights)
        std_array[ccafeature] = np.std(weights)/np.sqrt(len(weights))
        
    xx = np.arange(1, min_cca_dim+1)
    ax.plot(xx, avg_array, lw=3)
    ax.fill_between(xx, avg_array - std_array, avg_array + std_array, alpha=0.3)

    # ax.set_xticks(xx, xx, fontsize=fs)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.legend(fontsize=fs-3, frameon=False)

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    ax.set_ylabel("Prediction weights", fontsize=fs_title)
    ax.set_xlabel("CCA dimension", fontsize=fs_title)
    ax.set_title('Air puff weights by CCA dimension, mice average', fontsize=fs_title)

    fig.tight_layout()
    save_figure(fig, 'fig5SI_cca_airpuff_decoding_weights')
    
    
    
    
    
    ########## PCA DECODING PLOTS ##############
    
    #Plot pca weights (airpuff and position)    
    for weight_dict_idx, weight_dict in enumerate(pca_weight_dicts):
        
        fig, axs = plt.subplots(2, 4, figsize=(3 * 4, 2 * 2)); fig_num += 1
    
        for midx, mnum in enumerate(mouse_list):
            
            
            msession_list = CCA_analysis_dict[mnum, 'session_list']
            num_features = dimension_by_mouse_list[midx]
            ws_all = np.zeros((len(msession_list), num_features))
            for sidx, snum in enumerate(msession_list):
                weights = weight_dict[mnum, snum].squeeze()
                # weights =  pca_position_weights[midx, :, :min_dims]

                ws_all[sidx] = weights[:num_features]
                
            
            ws_avg = np.average(ws_all, axis=0)
            ws_std = np.std(ws_all, axis=0)
            ws_std = np.std(ws_all, axis=0)/np.sqrt(ws_all.shape[0])
            ax = axs.ravel()[mnum]
            
            xx = np.arange(1, num_features+1)
            ax.plot(xx, ws_avg, '-', lw=3)
            ax.fill_between(xx, ws_avg-ws_std, ws_avg+ws_std, alpha=0.3)
            
            # ax.set_xticks(xx, xx, fontsize=fs)
            ax.tick_params(axis='x', labelsize=fs)
            ax.tick_params(axis='y', labelsize=fs)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.legend(fontsize=fs, frameon=False)
        
            ax.spines[['right', 'top']].set_visible(False)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                
            if midx == 0:
                ax.set_ylabel("Decoding weights", fontsize=fs_title)
                ax.set_xlabel("Latent dimension", fontsize=fs_title)
        
        
                
            ax.set_title('M%d'%mnum, fontsize=fs_title)
            
        fig.suptitle('Decoding %s'%weight_types[weight_dict_idx], fontsize=fs_title)
        fig.tight_layout()
        
        fig_name = 'fig5SI_pca_%s_weights'%weight_types[weight_dict_idx]
        save_figure(fig, fig_name)
        
        
        
        
    fig = plt.figure(fig_num, figsize=(6,4)); fig_num += 1
    ax = plt.gca()
    
    
    min_pca_dim = np.min(dimension_by_mouse_list) 
    xx = np.arange(1, min_pca_dim+1)
    pca_weights_masked_list = []
    max_weights_by_dim = np.zeros(min_cca_dim)
    for weight_dict_idx, weight_dict in enumerate(pca_weight_dicts):

        #Get weights in a single matrix
        pca_weights_array = np.zeros((num_mice, len(session_list), max_dims))
        pca_dimension_counter = np.zeros(pca_weights_array.shape, dtype=bool) #1 if that dimension for that mouse and session is used

        for midx, mnum in enumerate(mouse_list):
            for sidx, snum in enumerate(session_list): 
                session_pca_position_weights= weight_dict[mnum, snum].squeeze()
                pca_weights_array[midx, sidx, :len(session_pca_position_weights)] = session_pca_position_weights
                pca_dimension_counter[midx, sidx, :len(session_pca_position_weights)] = 1
                
        #Plot overall position weights per dimension
        pca_weights_masked = np.ma.array(pca_weights_array, mask=np.invert(pca_dimension_counter))
        pca_weights_masked_list.append(pca_weights_masked)

        weights_per_dimension_avg = np.ma.average(pca_weights_masked, axis=(0,1))
        # weights_per_dimension_std = np.ma.std(pca_weights_masked, axis=(0,1)) #STD
        weights_per_dimension_std = np.ma.std(pca_weights_masked, axis=(0,1))/np.sqrt(np.ma.count(pca_weights_masked, axis=(0,1))) #SEM
        avg = weights_per_dimension_avg[:min_cca_dim]
        std = weights_per_dimension_std[:min_cca_dim]
        ax.plot(xx, avg, lw=3, color = weight_types_colors[weight_dict_idx], label = weight_types[weight_dict_idx])
        ax.fill_between(xx, avg-std, avg+std,alpha=0.3)
        
        max_weights_by_dim = np.maximum(max_weights_by_dim, avg+std)
    
        

            
            

    # ax.set_xticks(xx, xx, fontsize=fs_title)
    ax.tick_params(axis='x', labelsize=fs_title)
    ax.tick_params(axis='y', labelsize=fs_title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=fs, frameon=False)

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    ax.set_ylabel("Prediction weights", fontsize=fs_title)
    ax.set_xlabel("PCA dimension", fontsize=fs_title)
    # ax.set_title('Decoding %s, mice average'%weight_types[weight_dict_idx], fontsize=fs_title)

    fig.tight_layout()
    fig_name = 'fig5_pca_decoding_%s_weights_by_dim'%weight_types[weight_dict_idx]
    save_figure(fig, fig_name)
    
    
    
    ########## AIRPUFF VS POSITION DECODING PLOTS, FOR PCA AND CELL ##############

    
    weight_dict_pairs = [(pca_position_weight_dict, pca_airpuff_weight_dict), 
                         (cell_position_weight_dict, cell_airpuff_weight_dict)]
    
    weight_dict_pairs_labels = ['PCA', 'Cell']
    for weight_dict_pair_idx, (position_weight_dict, airpuff_weight_dict) in enumerate(weight_dict_pairs):
        fig = plt.figure(fig_num, figsize=(5,5)); fig_num += 1
        ax = plt.gca()
        
        xvals = []
        yvals = []
        for midx, mnum in enumerate(mouse_list):        
            msession_list = CCA_analysis_dict[mnum, 'session_list']
            num_features = CCA_analysis_dict[mnum, 'pca'][0].shape[0]
            for sidx, snum in enumerate(msession_list):
                position_weights = position_weight_dict[mnum, snum].squeeze()
                if weight_dict_pair_idx == 0:
                    position_weights = position_weights[:num_features] #Filter latter PCA dimensions
                
                airpuff_weights = airpuff_weight_dict[mnum, snum].squeeze()
                
                ax.scatter(position_weights, airpuff_weights, c='lightgray', edgecolor='black', alpha=0.9)
                xvals.extend(position_weights)
                yvals.extend(airpuff_weights)
            
        rval, pval = pf.add_linear_regression(xvals, yvals, ax, 'black')
        pval_label = pf.get_significance_label(pval, thresholds=[0.05, 0.005, 0.0005], asterisk=False, ns=True)
        ax.tick_params(axis='x', labelsize=fs_title)
        ax.tick_params(axis='y', labelsize=fs_title)
        # ax.legend(fontsize=fs-3, frameon=False)
    
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        ax.set_ylabel("Air puff weights", fontsize=fs_title)
        ax.set_xlabel("Position weights", fontsize=fs_title)
        
        

        ax.set_title('%s decoding weights, r=%.3f'%(weight_dict_pairs_labels[weight_dict_pair_idx], rval), fontsize=fs_title)
        # fig.suptitle('Airpuff decoding weights by PCA dimension', fontsize=fs+2)
        fig.tight_layout()
    
        save_figure(fig, 'fig5_%s_airpuff_vs_position_weights'%weight_dict_pairs_labels[weight_dict_pair_idx])
    
    
    
    
    

    
    
    return
    


def fig5SI_F_decoding_weight_control():
    ''' Compare the contribution of each cell to position and airpuff decoding '''
    
    mouse_list = np.arange(8)
    # mouse_list = [4]
    # mouse_list = [0,2,6,7]
    # mouse_list = [0,3,6,7]

    # mouse_list = [0, 6]
    # mouse_list = np.arange(7)


    
    preprocessing_param_dict = {
        #session params
        'mouse_list':mouse_list,
        'session_list':np.arange(9),
        
        #Preprocessing parameters
        'time_bin_size':1,
        'distance_bin_size':1,
        'gaussian_size':25,
        'data_used':'amplitudes',
        'running':True,
        'eliminate_v_zeros':True,
        'num_components':'all'
        }
    
    # dataset_fraction = 1/2.
    dataset_fraction = 2/3.
    # dataset_fraction = 3/5.

    #Prediction parameters
    predictor_name = 'Wiener'
    
    fig_num = 1;
    fs = 15
    
    session_list = preprocessing_param_dict['session_list']


    #Do analysis    
    PCA_analysis_dict = perform_pca_on_multiple_mice_param_dict(preprocessing_param_dict)
    PCA_dict = PCA_analysis_dict['PCA_dict']    
    
    #Split data into two
    PCA_dict1 = {}
    PCA_dict2 = {}

    for midx, mnum in enumerate(mouse_list):
        for sidx, snum in enumerate(session_list):
            position, pca = PCA_dict[mnum, snum]
            datapoints = position.size

            fraction = int(dataset_fraction * datapoints)
            PCA_dict1[mnum, snum] = (position[:fraction], pca[:, :fraction])
            PCA_dict2[mnum, snum] = (position[(datapoints-fraction):], pca[:, (datapoints-fraction):])
            
    PCA_analysis_dict1 = {k:v for k,v in PCA_analysis_dict.items()}    
    PCA_analysis_dict2 = {k:v for k,v in PCA_analysis_dict.items()}    

    PCA_analysis_dict1['PCA_dict'] = PCA_dict1
    PCA_analysis_dict2['PCA_dict'] = PCA_dict2

    
    #Get position weights from pca
    
    pca_position_weight_dict1, cell_position_weight_dict1 = get_position_decoding_weights_from_pca_dict(PCA_analysis_dict1, predictor_name)
    pca_position_weight_dict2, cell_position_weight_dict2 = get_position_decoding_weights_from_pca_dict(PCA_analysis_dict2, predictor_name)

    pca_position_weight_dict1 = normalize_weight_dict(pca_position_weight_dict1)
    pca_position_weight_dict2 = normalize_weight_dict(pca_position_weight_dict2)
    cell_position_weight_dict1 = normalize_weight_dict(cell_position_weight_dict1)
    cell_position_weight_dict2 = normalize_weight_dict(cell_position_weight_dict2)

    weight_dict_pairs = [(pca_position_weight_dict1, pca_position_weight_dict2), 
                         (cell_position_weight_dict1, cell_position_weight_dict2)]
    
    weight_dict_pairs_labels = ['PCA', 'Cell']
    for weight_dict_pair_idx, (weight_dict1, weight_dict2) in enumerate(weight_dict_pairs):
        fig = plt.figure(fig_num, figsize=(5,5)); fig_num += 1
        ax = plt.gca()
        
        xvals = []
        yvals = []
        for midx, mnum in enumerate(mouse_list):        
            
            for sidx, snum in enumerate(session_list):
                position_weights1 = weight_dict1[mnum, snum].squeeze()
                xvals.extend(position_weights1)
                position_weights2 = weight_dict2[mnum, snum].squeeze()
                yvals.extend(position_weights2)
                ax.scatter(position_weights1, position_weights2, c='lightgray', edgecolor='black', alpha=0.9)
            
        rval, pval = pf.add_linear_regression(xvals, yvals, ax, 'black')
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        # ax.legend(fontsize=fs-3, frameon=False)
    
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            
        ax.set_ylabel("Position weights", fontsize=fs+5)
        ax.set_xlabel("Position weights", fontsize=fs+5)
        ax.set_title('%s decoding weights, r=%.3f'%(weight_dict_pairs_labels[weight_dict_pair_idx], rval), fontsize=fs+2)
        fig.tight_layout()
    
        save_figure(fig, 'fig5SI_%s_airpuff_vs_airpuff_weights_control'%weight_dict_pairs_labels[weight_dict_pair_idx])











    




with h5py.File(FAT_CLUSTER_PATH, 'r') as fat_cluster:

    if __name__ == '__main__':
        tt = time.time()
        main()
        # np.random.seed(9)
        print('Time Ellapsed: %.1f' % (time.time() - tt))
        plt.show()