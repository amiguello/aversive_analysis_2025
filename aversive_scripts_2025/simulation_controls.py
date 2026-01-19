# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 16:31:58 2025

@author: admin
"""
import time
import h5py
import bisect
from functools import partial

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Rectangle

#Global project variables
import project_parameters as pparam
from project_parameters import (FAT_CLUSTER_PATH, OUTPUT_PATH, SESSION_NAMES, SESSION_REPEATS, MOUSE_TYPE_LABELS)
import main_figures as mf

#Project scripts
import processing_functions as pf
import mCCA as mCCA_funs
import APdecoding_funs as APfuns

def main():
    # compute_and_store_velocity_data()
    # example_generated_data_pca_and_position_prediction()
    # compute_position_prediction_error_on_generated_data()
    # quantify_generated_data_cca_single_pair()
    # simulated_mCCA_quantification()
    simulated_airpuff_quantification()
    # test_shapes()
    # <>
    
    # return
    
    
def test_shapes():
    
    # ## Periodic spiral shape
    # max_pos = pparam.MAX_POS
    # pp = np.arange(max_pos)
    # x = np.cos(2*np.pi*pp/max_pos)
    # y = np.sin(2*np.pi*pp/max_pos)
    # # z =[0]*len(pp)
    # z = np.cos(2*np.pi*pp/max_pos)
    
    
    
    
    # fig = plt.figure().add_subplot(projection='3d')
    # ax = plt.gca()
    # ax.plot(latent_array[0], latent_array[1], latent_array[2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # return
    
    ## Eight shape
    max_pos = pparam.MAX_POS
    pp = np.arange(max_pos)
    x = np.cos(2*np.pi*pp/max_pos)
    y = np.sin(2*np.pi*pp/max_pos)
    # z =[0]*len(pp)
    z = np.cos(2*np.pi*pp/max_pos)

    latent_array = np.vstack((x,y,z))
    print(latent_array.shape)
    
    xang = np.pi/3
    
    
                
    for p in pp:
        ang = 2*np.pi*p/max_pos
        Rx = np.array([[1,0,0],[0, np.cos(ang), -np.sin(ang)],[0, np.sin(ang), np.cos(ang)]])
        Ry = np.array([[np.cos(ang), 0, np.sin(ang)],[0,1,0],[-np.sin(ang), 0, np.cos(ang)]])
        # Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0,0,1]])
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0, 0, 1]])

        
        Rot_array = Rx
        # Rot_array = np.dot(Rot_array, Ry)
        # Rot_array = np.dot(Rot_array, Rz)

        latent_array[:,p] = np.dot(latent_array[:,p], Rot_array)
        diff = pf.get_periodic_difference(float(p), max_pos/4, max_pos)
        latent_array[2,p] += 0.5*(np.exp(-1 * ((diff**2)/(2*(200**2)))))
        # diff = pf.get_periodic_difference(float(p), 3*max_pos/4, max_pos)
        # latent_array[2,p] += -(1+np.exp(-1 * ((diff**2)/(2*(200**2)))))# - (1+np.exp(-1 * ((diff**2)/(2*(400**2)))))
        # latent_array[:,p] = (1 + 0.5*(np.sin(2*np.pi*(p/max_pos)))) * np.dot(latent_array[:,p], Rx)
        # latent_array[:,p] = (1+np.exp(-1 * (((p-max_pos/2)%max_pos)**2)/(2*(200**2)))) * np.dot(latent_array[:,p], Rx)

        # latent_array[:,p] = np.dot(latent_array[:,p], Rx)
        
        
    # xx = np.linspace(0, 1500)
    # plt.figure(1)
    # ax = plt.gca()
    # ax.plot(xx, 1+np.exp(-1 * ((xx-max_pos/2)**2)/(2*(200**2))))

    fig = plt.figure().add_subplot(projection='3d')
    ax = plt.gca()
    ax.plot(latent_array[0], latent_array[1], latent_array[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # # return


def example_generated_data_pca_and_position_prediction():
    ########## PARAMETERS ###########
    
    #General
    max_pos = pparam.MAX_POS
    trial_bins = 30
       
    #Data preprocessing
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    position_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    
    
    #Velocity generation
    session_list_to_train_velocity = np.arange(3)
    # session_list_to_train_velocity = [3,4,5,6]
    eliminate_v_zeros_to_train_velocity = False
    
    ### Position generation
    dt = 0.01 # time interval, in seconds
    num_neurons = 50 # Number of neurons
    num_trials = 50
    
    ### Firing rate generation
    simulated_type = 'gaussian'
    ## Probability weights by position for the firing rate generation
    ## (1) Homogenous
    firing_rate_position_probs = np.ones(max_pos)/max_pos 
    
    # ## (2) Reward bias
    # firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    # firing_rate_position_probs[:150] = firing_rate_position_probs[-150:] = 2.
    # firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## (3) Delete sections
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[250:1250] = 0
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## Probability weights by position for the firing rate standard deviation
    ## (1) Homogeneous
    firing_rate_std_probs = np.ones(max_pos)/max_pos 
    
    
    # ## (2) Small only
    # firing_rate_std_probs = np.ones(max_pos)
    # firing_rate_std_probs[250:] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)

    # ## (3) Small or big
    # firing_rate_std_probs = np.ones(4*max_pos)
    # firing_rate_std_probs[250:(4*max_pos - 250)] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)
    
    # firing_rate_kwargs={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}

    ## Latent parameters
    simulated_type = 'latent'
    
    error_std = .1
    
    # error_std = np.array([0.05]*max_pos)
    # error_std[500:1000] = 3.
    latent_type = 'deformed circle 2'
    latent_type = 'deformed circle sigmoid 1'
    firing_rate_kwargs={"error_std":error_std, 'latent_type':latent_type}


    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    ## Plotting parameters
    fig_num = 1
    fs = 15
    pca_plot_bin_size = 30 #Only used for plotting pca
    
    ## Store velocities from real data
    compute_and_store_velocity_data(trial_bins, session_list_to_train_velocity, time_bin_size, position_bin_size, gaussian_size, 
                                    data_used, running, eliminate_v_zeros=eliminate_v_zeros_to_train_velocity)
    
    ### Create simulated firing rates
    firing_rate_funs_by_neuron = create_simulated_firing_rates(num_neurons, simulated_type, **firing_rate_kwargs)

    # ## Generate a session worth of data, position and spikes
    generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity, plot=True)


    fig_num += 1
    ## Pre-process generated data
    data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
    data_dict = preprocess_generated_data(data_dict, time_bin_size, position_bin_size, gaussian_size, eliminate_v_zeros)
    pca_input_data = data_dict['amplitudes_binned_normalized']
    position = data_dict['distance']


    

    
    ## Perform PCA
    pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
    position, pca, _ = pf.warping(position, pca, 150, max_pos=pparam.MAX_POS, 
                                  warp_sampling_type = 'interpolation', warp_based_on = 'time', return_flattened=True)
    

    
    
    pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                            predictor_name=predictor_name, predictor_default=None, return_error=error_type)
    
    
    
    
    ########### PLOTS ###########
    ## Plot PCA
    angle = 75
    angle_azim = -90
    rows = 2 #One row per session
    cols = 1 #raw data and trial averaged
    fig, axs = plt.subplots(rows, cols, subplot_kw={"projection": "3d"}, figsize=(9,7)) 
    
    #Plot all trials
    ax = axs.ravel()[0]
    pf.plot_pca_with_position(pca, position, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, scatter=True, cbar=False, cbar_label='Position (mm)',
                                alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                ms = 10, lw=3)

    #Plot trial average

    
    ax = axs.ravel()[1]
    position_unique, pca_average, pca_std = pf.compute_average_data_by_position(pca, position, position_bin_size=pca_plot_bin_size, max_pos=max_pos)
    pf.plot_pca_with_position(pca_average, position_unique, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, cbar=False, cbar_label='Position (mm)',
                                alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                scatter=False, ms = 250, lw=6)
    

    ### PLOT POSITION PREDICTION ###
    markersize=50
    fig_num += 1
    fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
    ax = plt.gca()
    
    timesteps = np.arange(len(position))

    for plot_idx, p in enumerate([position, pos_pred]):
        label = pparam.PREDICTION_LABELS[plot_idx]
        color = pparam.PREDICTION_COLORS[label]
        ax.scatter(timesteps, p, s=markersize, color=color, label=label)
        break
        
    # ax.scatter(timesteps, pos_pred_shuffle, s=markersize, color='red', label='shuffle')

    ax.set_xlim([0,2000])
        
    ax.set_xlabel('Timestep', fontsize=fs+4)
    # ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.set_ylabel('Position (mm)', fontsize=fs+4)
    ax.set_title('Generated session, SSE=%.1f (cm)'%(error), 
                 pad=20, fontsize=fs+4)
    ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
    fig.tight_layout()
    
def quantify_generated_data_cca_single_pair():
    
    ########## PARAMETERS ###########
    
    #General
    max_pos = pparam.MAX_POS
    trial_bins = 30
       
    #Data preprocessing
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    position_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    num_components = 'all'
    
    #Velocity generation
    session_list_to_train_velocity = np.arange(3)
    session_list_to_train_velocity = [0,1,2]
    running_to_train_velocity = False #If True, only "running" bins are used in generating data
    eliminate_v_zeros_to_train_velocity = False
    
    ### Position generation
    dt = 0.01 # time interval, in seconds
    num_neurons = 50 # Number of neurons
    num_trials = 50 # Number of trials per generated session
    
    ### Firing rate generation


    ## Probability weights by position for the firing rate generation
    ## (1) Homogenous
    firing_rate_position_probs = np.ones(max_pos)/max_pos 
    
    ## (2) Reward bias
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[:150] = firing_rate_position_probs[-150:] = 2.
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## (3) Delete sections
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[250:1250] = 0
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## Probability weights by position for the firing rate standard deviation
    ## (1) Homogeneous
    firing_rate_std_probs = np.ones(max_pos)/max_pos 
    
    
    # ## (2) Small only
    # firing_rate_std_probs = np.ones(max_pos)
    # firing_rate_std_probs[250:] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)

    # ## (3) Small or big
    # firing_rate_std_probs = np.ones(4*max_pos)
    # firing_rate_std_probs[250:(4*max_pos - 250)] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)
    
    simulated_type1 = 'gaussian'
    firing_rate_kwargs1 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}
    
    simulated_type2 = 'gaussian'
    firing_rate_kwargs2 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}

    ## Latent parameters    
    error_std1 = 4.
    error_std2 = 4.

    # error_std = np.array([0.05]*max_pos)
    # error_std[500:1000] = 3.
    
    simulated_type1 = 'latent'
    latent_type1 = 'deformed circle 1'
    latent_type1 = 'deformed circle sigmoid 1'

    firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
    
    simulated_type2 = 'latent'
    latent_type2 = 'deformed circle 1'
    # latent_type2 = 'deformed circle double'
    latent_type2 = 'deformed circle sigmoid 2'

    firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
    
    
    
    simulated_type_list = [simulated_type1, simulated_type2]    
    firing_rate_kwargs_list = [firing_rate_kwargs1, firing_rate_kwargs2]


    ## Session generation
    number_of_generated_session_pairs = 2
    number_of_sessions_to_align = len(simulated_type_list)
    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    ## CCA parameters ##
    CCA_dim = 12
    return_warped_data = False
    return_trimmed_data = False
    sessions_to_align = 'all'
    cca_shuffle = False
    warping_bins = 150
    warp_based_on = 'position'
    skip_alignment = False
    
    ## Plotting parameters
    fig_num = 1
    fs = 15
    pca_plot_bin_size = 30 #Only used for plotting pca
       
    ## Store velocities from real data
    compute_and_store_velocity_data(trial_bins, session_list_to_train_velocity, time_bin_size, position_bin_size, gaussian_size, 
                                    data_used, running_to_train_velocity, eliminate_v_zeros=eliminate_v_zeros_to_train_velocity)
    
    

    
    #Generate multiple pairs of sessions
    self_error_list = []
    unaligned_error_list = []
    aligned_error_list = []
    
    self_error_array_total = np.zeros((number_of_sessions_to_align,0))
    unaligned_error_array_total = np.zeros((number_of_sessions_to_align,number_of_sessions_to_align,0))
    aligned_error_array_total = np.zeros((number_of_sessions_to_align,number_of_sessions_to_align,0))
    
    for npair in range(number_of_generated_session_pairs):
    
        
    
        #Generate pair of sessions
        position_list = []
        pca_list = []
        variance_explained_list = []
        
        for i in range(number_of_sessions_to_align):
    
            ### Create simulated firing rates
            firing_rate_kwargs = firing_rate_kwargs_list[i]
            simulated_type = simulated_type_list[i]
            firing_rate_funs_by_neuron = create_simulated_firing_rates(num_neurons, simulated_type, **firing_rate_kwargs)


            # ## Generate a session worth of data, position and spikes
            generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity, plot=False)
            
            
            ## Pre-process generated data
            data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
            data_dict = preprocess_generated_data(data_dict, time_bin_size, position_bin_size, gaussian_size, eliminate_v_zeros)
            pca_input_data = data_dict['amplitudes_binned_normalized']
            position = data_dict['distance']
        
        
            
        
            
            ## Perform PCA
            pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
            ## Perform PCA
            pca = decomposition.PCA(n_components=num_neurons)
            pca.fit(pca_input_data.T)
            
            #PCA over time
            pca_data = pf.project_spikes_PCA(pca_input_data, pca_instance = pca, num_components = num_components)
            variance_explained = pca.explained_variance_ratio_
                
            position_list.append(position)
            pca_list.append(pca_data)
            variance_explained_list.append(variance_explained)
            

    
        
        #Align the pair of sessions
    
        M = number_of_sessions_to_align
        
        #Set PCA dimension
        pca_list = mCCA_funs.set_dimension_of_pca_list(pca_list, CCA_dim, variance_explained_list)
        
        #Perform mCCA
        pos_list_aligned, pca_dict_aligned, mCCA = mCCA_funs.perform_warped_mCCA(position_list, pca_list, max_pos, warping_bins, warp_based_on, 
                                                                                 return_warped_data, return_trimmed_data, cca_shuffle, skip_alignment)
    
        #Normalize PCA after alignment changes
        pca_dict_aligned = mCCA_funs.normalize_pca_dict_aligned(pca_dict_aligned, mCCA)
        
        #Find space with best alignment
        best_space = mCCA_funs.return_best_mCCA_space(pos_list_aligned, pca_dict_aligned, max_pos=1500, verbose=False)
        pca_list_aligned = pca_dict_aligned[best_space]
                    
        pca_list = [pca_dict_aligned[m][m] for m in range(M)]
        unaligned_error_array, aligned_error_array = mCCA_funs.get_cross_prediction_errors(pos_list_aligned, pca_list, pos_list_aligned, pca_dict_aligned, 
                                                                                           max_pos, cv_folds, error_type, predictor_name)
        
        print(unaligned_error_array)
        print(aligned_error_array)
        # self_error_array_total += unaligned_error_array.diagonal()
        # unaligned_error_array_total += unaligned_error_array
        # aligned_error_array_total += aligned_error_array
        
        self_error_array_total = np.hstack((self_error_array_total, unaligned_error_array.diagonal().reshape((2,1))))
        unaligned_error_array_total = np.dstack((unaligned_error_array_total, unaligned_error_array))
        aligned_error_array_total = np.dstack((aligned_error_array_total, aligned_error_array))
        
        # self_error = np.average(unaligned_error_array.diagonal())
        # unaligned_error = np.average(unaligned_error_array[[0,1],[1,0]])
        # aligned_error = np.average(aligned_error_array[[0,1],[1,0]])
        
        # self_error_list.append(self_error)
        # unaligned_error_list.append(unaligned_error)
        # aligned_error_list.append(aligned_error)
        
        # self_error_list.append(unaligned_error_array.diagonal()[0])
        # unaligned_error_list.append(unaligned_error_array[0,1])
        # aligned_error_list.append(aligned_error_array[0,1])
        
        # self_error_list.append(unaligned_error_array.diagonal()[1])
        # unaligned_error_list.append(unaligned_error_array[1,0])
        # aligned_error_list.append(aligned_error_array[1,0])
        
        # self_error_list.append(unaligned_error_array.diagonal()[0])
        # unaligned_error_list.append(unaligned_error_array[0,1])
        # aligned_error_list.append(aligned_error_array[0,1])



        self_error_list.extend(unaligned_error_array.diagonal())
        unaligned_error_list.extend(unaligned_error_array[[0,1],[1,0]])
        aligned_error_list.extend(aligned_error_array[[0,1],[1,0]])
        

        
        #Plot one example
        if npair == 0:
            angle = 75
            angle_azim = -90
            rows = 1 
            cols = 2 #Unaligned and aligned
            fig, axs = plt.subplots(rows, cols, subplot_kw={"projection": "3d"}, figsize=(9,7)); fig_num += 1
                    
            
            #Plot unaligned PCAs
            ax = axs.ravel()[0]
            for i in range(2):
                pca = pca_list[i]
                position = pos_list_aligned[i]

                position_unique, pca_average, pca_std = pf.compute_average_data_by_position(pca, position, position_bin_size=pca_plot_bin_size, max_pos=max_pos)
                pf.plot_pca_with_position(pca_average, position_unique, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, cbar=False, cbar_label='Position (mm)',
                                            alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                            scatter=False, ms = 250, lw=6)
            #Plot aligned PCAs
            ax = axs.ravel()[1]
            for i in range(2):
                pca = pca_list_aligned[i]
                position = pos_list_aligned[i]

                position_unique, pca_average, pca_std = pf.compute_average_data_by_position(pca, position, position_bin_size=pca_plot_bin_size, max_pos=max_pos)
                pf.plot_pca_with_position(pca_average, position_unique, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, cbar=False, cbar_label='Position (mm)',
                                            alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                            scatter=False, ms = 250, lw=6)
                
                
                
            ### PLOT POSITION PREDICTION ###
            markersize=50
            fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
            ax = plt.gca()
            
            position = pos_list_aligned[0]
            position_pred = pos_list_aligned[1]

            for plot_idx, p in enumerate([position, position_pred]):
                label = pparam.PREDICTION_LABELS[plot_idx]
                color = pparam.PREDICTION_COLORS[label]
                timesteps = np.arange(len(p))
                ax.scatter(timesteps, p, s=markersize, color=color, label=label)
                
            # ax.scatter(timesteps, pos_pred_shuffle, s=markersize, color='red', label='shuffle')

            ax.set_xlim([0,2000])
                
            ax.set_xlabel('Timestep', fontsize=fs+4)
            # ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
            ax.tick_params(axis='x', labelsize=fs)
            ax.tick_params(axis='y', labelsize=fs)
            ax.spines[['right', 'top']].set_visible(False)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
            ax.set_ylabel('Position (mm)', fontsize=fs+4)
            # ax.set_title('Generated session, SSE=%.1f (cm)'%(error), 
            #              pad=20, fontsize=fs+4)
            ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
            fig.tight_layout()
            
            
            
    
    
    cca_results_dict = {}
    cca_results_dict[pparam.CCA_LABELS[0]] = self_error_array_total
    cca_results_dict[pparam.CCA_LABELS[1]] = unaligned_error_array_total
    cca_results_dict[pparam.CCA_LABELS[2]] = aligned_error_array_total
    
    if number_of_sessions_to_align == 2:
        #Plot for 1 aligning to 2, and 2 aligning to 1
        for i in range(2):
                    
            ref_idx = i
            target_idx = (i+1)%2
    
            fig = plt.figure(num=fig_num, figsize=(4,4)); fig_num += 1
            ax  = plt.gca()
            all_bars_width = .8
            num_of_bars = 1
            barwidth = all_bars_width/(num_of_bars)
            cca_labels = pparam.CCA_LABELS[:3]
            xpos_list = np.arange(len(cca_labels))
            
            
        
            for label_idx, label in enumerate(cca_labels):
                xpos = xpos_list[label_idx]
                error_array = cca_results_dict[label]
                if label == 'Self':
                    errors = error_array[target_idx]
                else:
                    errors = error_array[ref_idx, target_idx]
                avg = np.average(errors)
                # err = scipy.stats.sem(errors) ## Change also height in significance!
                err = np.std(errors)/np.sqrt(len(errors))
                
                ax.bar(xpos, avg, width=barwidth, alpha=0.7, edgecolor=None, color=pparam.CCA_COLORS[label_idx])
                ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CCA_COLORS[label_idx], alpha=0.6)
                
                xx_scatter = [xpos]*len(errors)
                ax.scatter(xx_scatter, errors, s=30, color = pparam.CCA_COLORS[label_idx])
                
            
            
                
            xlabels = cca_labels
            ax.set_xticks(xpos_list, xlabels, fontsize=fs)
            ax.set_ylabel('Error (cm)', fontsize=fs)
            ax.tick_params(axis='y', which='major', labelsize=fs)
            
            ax.spines[['right', 'top']].set_visible(False)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.xaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)
                ax.yaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)    
                
            fig.tight_layout()
    else:
        ref_idx = i
        target_idx = (i+1)%2

        fig = plt.figure(num=fig_num, figsize=(4,4)); fig_num += 1
        ax  = plt.gca()
        all_bars_width = .8
        num_of_bars = 1
        barwidth = all_bars_width/(num_of_bars)
        cca_labels = pparam.CCA_LABELS[:3]
        xpos_list = np.arange(len(cca_labels))
        
        
    
        for label_idx, label in enumerate(cca_labels):
            xpos = xpos_list[label_idx]
            error_array = cca_results_dict[label]
            if label == 'Self':
                errors = error_array
            else:
                errors = error_array[ref_idx, target_idx]
            avg = np.average(errors)
            # err = scipy.stats.sem(errors) ## Change also height in significance!
            err = np.std(errors)/np.sqrt(len(errors))
            
            ax.bar(xpos, avg, width=barwidth, alpha=0.7, edgecolor=None, color=pparam.CCA_COLORS[label_idx])
            ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CCA_COLORS[label_idx], alpha=0.6)
            
            xx_scatter = [xpos]*len(errors)
            ax.scatter(xx_scatter, errors, s=30, color = pparam.CCA_COLORS[label_idx])
            
        
        
            
        xlabels = cca_labels
        ax.set_xticks(xpos_list, xlabels, fontsize=fs)
        ax.set_ylabel('Error (cm)', fontsize=fs)
        ax.tick_params(axis='y', which='major', labelsize=fs)
        
        ax.spines[['right', 'top']].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            ax.xaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)
            ax.yaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)    
            
        fig.tight_layout()
            

def simulated_mCCA_quantification():
    ########## PARAMETERS ###########
    
    #General
    max_pos = pparam.MAX_POS
    trial_bins = 30
       
    #Data preprocessing
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    position_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    num_components = 'all'
    
    #Velocity generation
    session_list_to_train_velocity = np.arange(3)
    session_list_to_train_velocity = [0,1,2]
    running_to_train_velocity = False #If True, only "running" bins are used in generating data
    eliminate_v_zeros_to_train_velocity = False
    
    ### Position generation
    dt = 0.01 # time interval, in seconds
    num_neurons = 50 # Number of neurons
    num_trials = 50 # Number of trials per generated session
    
    ### Firing rate generation


    ## Probability weights by position for the firing rate generation
    ## (1) Homogenous
    firing_rate_position_probs = np.ones(max_pos)/max_pos 
    
    ## (2) Reward bias
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[:150] = firing_rate_position_probs[-150:] = 2.
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## (3) Delete sections
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[250:1250] = 0
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## Probability weights by position for the firing rate standard deviation
    ## (1) Homogeneous
    firing_rate_std_probs = np.ones(max_pos)/max_pos 
    
    
    # ## (2) Small only
    # firing_rate_std_probs = np.ones(max_pos)
    # firing_rate_std_probs[250:] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)

    # ## (3) Small or big
    # firing_rate_std_probs = np.ones(4*max_pos)
    # firing_rate_std_probs[250:(4*max_pos - 250)] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)
    
    simulated_type1 = 'gaussian'
    firing_rate_kwargs1 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}
    
    simulated_type2 = 'gaussian'
    firing_rate_kwargs2 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}

    ## Latent parameters    
    error_std1 = .01
    error_std2 = .01

    # error_std = np.array([0.05]*max_pos)
    # error_std[500:1000] = 3.
    
    simulated_type1 = 'latent'
    # latent_type1 = 'deformed circle 3'
    latent_type1 = 'deformed circle sigmoid 1'

    firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
    
    simulated_type2 = 'latent'
    # latent_type2 = 'deformed circle 2'
    # latent_type2 = 'deformed circle twist 2'
    latent_type2 = 'deformed circle sigmoid 2'

    firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
    
    
    



    ## Session 
    mice_num = 10
    B_session_num = 3
    T_session_num = 4
    P_session_num = 2
    session_num = B_session_num + T_session_num + P_session_num
    
    simulated_type_list = [simulated_type1] * B_session_num + [simulated_type2]*T_session_num + [simulated_type1]*P_session_num
    firing_rate_kwargs_list = [firing_rate_kwargs1] * B_session_num + [firing_rate_kwargs2]*T_session_num + [firing_rate_kwargs1]*P_session_num
    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    ## CCA parameters ##
    CCA_dim = 12
    return_warped_data = True
    return_trimmed_data = False
    sessions_to_align = 'all'
    cca_shuffle = False
    warping_bins = 150
    warp_based_on = 'position'
    skip_alignment = False
    
    ## TCA params ##
    TCA_method = "ncp_hals" #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
    TCA_factors = 'max' #int, or 'max' to get the maximum possible (determined by CCA)
    TCA_replicates = 10
    TCA_convergence_attempts = 10 #Number of times TCA can fail before giving up
    TCA_on_LDA_repetitions =20
    
    ## LDA params ##
    LDA_components = 1
    LDA_imbalance_prop = .51
    LDA_imbalance_repetitions = 10
    LDA_trial_shuffles = 0
    LDA_session_shuffles = 0
    session_comparisons = 'BT' #'airpuff', 'BT', 'TP', 'BP'
    
    ## Plotting parameters
    fig_num = 1
    fs = 15
    pca_plot_bin_size = 50 #Only used for plotting pca
    
    ## Store velocities from real data
    compute_and_store_velocity_data(trial_bins, session_list_to_train_velocity, time_bin_size, position_bin_size, gaussian_size, 
                                    data_used, running_to_train_velocity, eliminate_v_zeros=eliminate_v_zeros_to_train_velocity)
    
    

    
    #Generate multiple pairs of sessions    
    unaligned_error_array_total = np.zeros((session_num, session_num, 0))
    aligned_error_array_total = np.zeros((session_num, session_num, 0))
    
    
        
    for mnum in range(mice_num):
        #Generate list of sessions
        position_list = []
        pca_list = []
        variance_explained_list = []
        
        for i in range(session_num):
    
            ### Create simulated firing rates
            firing_rate_kwargs = firing_rate_kwargs_list[i]
            simulated_type = simulated_type_list[i]
            firing_rate_funs_by_neuron = create_simulated_firing_rates(num_neurons, simulated_type, **firing_rate_kwargs)
    
    
            # ## Generate a session worth of data, position and spikes
            generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity, plot=False)
            
            
            ## Pre-process generated data
            data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
            data_dict = preprocess_generated_data(data_dict, time_bin_size, position_bin_size, gaussian_size, eliminate_v_zeros)
            pca_input_data = data_dict['amplitudes_binned_normalized']
            position = data_dict['distance']
        
        
            
        
            
            ## Perform PCA
            pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
            ## Perform PCA
            pca = decomposition.PCA(n_components=num_neurons)
            pca.fit(pca_input_data.T)
            
            #PCA over time
            pca_data = pf.project_spikes_PCA(pca_input_data, pca_instance = pca, num_components = num_components)
            variance_explained = pca.explained_variance_ratio_
                
            position_list.append(position)
            pca_list.append(pca_data)
            variance_explained_list.append(variance_explained)
            
    
    
        
        #Align the list of sessions
    
        M = session_num
        
        #Set PCA dimension
        pca_list = mCCA_funs.set_dimension_of_pca_list(pca_list, CCA_dim, variance_explained_list)
        
        #Perform mCCA
        pos_list_aligned, pca_dict_aligned, mCCA = mCCA_funs.perform_warped_mCCA(position_list, pca_list, max_pos, warping_bins, warp_based_on, 
                                                                                 return_warped_data, return_trimmed_data, cca_shuffle, skip_alignment)
    
        #Normalize PCA after alignment changes
        pca_dict_aligned = mCCA_funs.normalize_pca_dict_aligned(pca_dict_aligned, mCCA)
        
        #Find space with best alignment
        best_space = mCCA_funs.return_best_mCCA_space(pos_list_aligned, pca_dict_aligned, max_pos=1500, verbose=False)
        pca_list_aligned = pca_dict_aligned[best_space]
                    
        pca_list_unaligned = [pca_dict_aligned[m][m] for m in range(M)]
        unaligned_error_array, aligned_error_array = mCCA_funs.get_cross_prediction_errors(pos_list_aligned, pca_list_unaligned, pos_list_aligned, pca_dict_aligned, 
                                                                                           max_pos, cv_folds, error_type, predictor_name)
        

        
        
        unaligned_error_array_total = np.dstack((unaligned_error_array_total, unaligned_error_array))
        aligned_error_array_total = np.dstack((aligned_error_array_total, aligned_error_array))
            
        
        if mnum == 0:
            ## Plot unaligned vs aligned sessions
            
            fig_CCAcomp, axs_CCAcomp = plt.subplots(nrows=1, ncols = 2, squeeze=False, figsize=(5,3), num=fig_num, subplot_kw={'projection':'3d'})
            # fig_CCAcomp, axs_CCAcomp = plt.subplots(nrows=2, ncols = 2, squeeze=False, figsize=subfig_b_size, num=fig_num)
        
            fig_num += 1
            
            subplot_counter = 0
            for alignment_idx in range(2): #0 is unaligned, 1 is aligned
                
                ax = axs_CCAcomp.ravel()[alignment_idx]
                if alignment_idx == 0:
                    pca_list = pca_list_unaligned
                elif alignment_idx == 1:
                    pca_list = pca_list_aligned
                    
                pos_list = pos_list_aligned
                num_sessions = len(pca_list)
                for sidx in range(num_sessions):
                    pca = pca_list[sidx]
                    pos = pos_list[sidx]
                    pos_bins, pca_avg, _ = pf.compute_average_data_by_position(pca, pos, position_bin_size=pca_plot_bin_size)
                    pf.plot_pca_with_position(pca_avg, pos_bins, ax=ax, scatter=False, cbar=None, 
                                              angle = None, angle_azim = None)
                subplot_counter += 1    
                        
            fig_CCAcomp.tight_layout()
            
            
            fig_CCAmap, axs_CCAmap = plt.subplots(1, 2, num=fig_num, figsize=(6,4)); fig_num += 1
    
    ## Plot colormap of unaligned vs aligned
    unaligned_error_array_avg = np.average(unaligned_error_array_total, axis=2)        
    aligned_error_array_avg = np.average(aligned_error_array_total, axis=2)        

    plot_data_list = [unaligned_error_array_avg, aligned_error_array_avg]
    min_error = np.min([np.min(error_array) for error_array in plot_data_list])
    max_error = np.max([np.max(error_array) for error_array in plot_data_list])
    title_list = ['Unaligned', 'Aligned']
    session_list = np.arange(num_sessions)
    fs = 15
    for alignment_idx in range(2):
        ax = axs_CCAmap.ravel()[alignment_idx]
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

        
    axs_CCAmap[0].set_ylabel('Trained on', fontsize=fs+4)
    axs_CCAmap[1].set_xlabel('Predicted on', fontsize=fs+4)
    
    
    fig_CCAmap.subplots_adjust(right=0.90)
    cbar_ax = fig_CCAmap.add_axes([1.05, 0.17, 0.035, 0.7])    
    norm = mpl.colors.Normalize(vmin=0, vmax=max_error)    
    cbar = fig_CCAmap.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=pparam.ERROR_CMAP), cax=cbar_ax, orientation='vertical', fraction=0.01)
    cbar.ax.set_ylabel('Prediction error (cm)', fontsize=fs+7, rotation=270, labelpad=25)    
    cbar.ax.tick_params(axis='both', which='major', labelsize=fs+5)    
    
    fig_CCAmap.subplots_adjust(wspace=0, hspace=0)
    fig_CCAmap.tight_layout()
    
    
    
    ## Barplot of alignment errors
    
    fig = plt.figure(num=fig_num, figsize=(4,4)); fig_num += 1
    ax  = plt.gca()
    all_bars_width = .8
    num_of_bars = 1
    barwidth = all_bars_width/(num_of_bars)
    cca_labels = pparam.CCA_LABELS[:3]
    xpos_list = np.arange(len(cca_labels))
    

    for label_idx, label in enumerate(cca_labels):
        xpos = xpos_list[label_idx]
        if label == 'Self':
            errors = unaligned_error_array_total.diagonal()
        elif label == 'Unaligned':
            errors = unaligned_error_array_total
            errors = errors[np.where(~np.eye(errors.shape[0], dtype=bool))]
        elif label == 'Aligned':
            errors = aligned_error_array_total
            errors = errors[np.where(~np.eye(errors.shape[0], dtype=bool))]
        errors = errors.ravel()
        avg = np.average(errors)
        # err = scipy.stats.sem(errors) ## Change also height in significance!
        err = np.std(errors)/np.sqrt(len(errors))
        
        ax.bar(xpos, avg, width=barwidth, alpha=0.7, edgecolor=None, color=pparam.CCA_COLORS[label_idx])
        ax.errorbar([xpos], avg, yerr=err, fmt='', markersize=35, markeredgewidth=5, elinewidth = 5, zorder=2, color=pparam.CCA_COLORS[label_idx], alpha=0.6)
        
        # xx_scatter = [xpos]*len(errors)
        # ax.scatter(xx_scatter, errors, s=30, color = pparam.CCA_COLORS[label_idx])
        
    
    
        
    xlabels = cca_labels
    ax.set_xticks(xpos_list, xlabels, fontsize=fs)
    ax.set_ylabel('Error (cm)', fontsize=fs)
    ax.tick_params(axis='y', which='major', labelsize=fs)
    
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        ax.xaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)
        ax.yaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)    
        
    fig.tight_layout()
    
    
    return 
    #Airpuff TCA decoding
    
    # session_list = CCA_analysis_dict[mnum, 'session_list']
    # pos_list = CCA_analysis_dict[mnum, 'pos']
    # pca_list = CCA_analysis_dict[mnum, 'pca']
    # pca_list = CCA_analysis_dict[mnum, 'pca_unaligned']
    session_list = np.arange(session_num)
    pos_list = pos_list_aligned
            
    data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, warping_bins, session_list)
    num_features, num_bins, total_trials = data_by_trial.shape
    num_CCA_dims = num_features
    if TCA_factors == 'max':
        num_TCA_dims = num_CCA_dims
    else:
        num_TCA_dims = int(TCA_factors)
    print('Performing TCA+LDA on M%d // CCA dim: %d, // TCA dim: %d' %(1, num_CCA_dims, num_TCA_dims))       
        
    #Selecting trials to decode
    trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, session_comparisons)
    num_trials_to_keep = len(label_by_trial)
    
    
    label_by_trial_predicted = np.zeros((0, num_trials_to_keep), dtype=int) #1st axis is TCA repetition, 2nd is trial
    
    plotted_factors = False
    while plotted_factors == False:
        # Step 4: TCA
        KTensor = APfuns.perform_TCA(data_by_trial, num_TCA_dims, TCA_replicates, 
                                     TCA_method, TCA_convergence_attempts)
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
        
        ax.set_title('M%d, F1 = %.1f'%(1, f1), fontsize=15)
        fig_name = 'fig3_B_LDA_example_M%d'%1
        plotted_factors = True

    
def simulated_airpuff_LDA_plot():
    pass

def simulated_airpuff_quantification():
    ########## PROCESSING PARAMETERS ###########
    
    #General
    max_pos = pparam.MAX_POS
    trial_bins = 30
       
    #Data preprocessing
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    position_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    num_components = 'all'
    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    ## CCA parameters ##
    CCA_dim = 12
    return_warped_data = True
    return_trimmed_data = False
    sessions_to_align = 'all'
    cca_shuffle = False
    warping_bins = 150
    warp_based_on = 'position'
    skip_alignment = False
    
    # cca_param_dict = {
    #     'CCA_dim':'.9', #11, '.9'
    #     'return_warped_data':True,
    #     'return_trimmed_data':False,
    #     'sessions_to_align':'all',
    #     'shuffle':False,
    #     'skip_alignment':False
    #     }
    
    ap_decoding_param_dict = {
        'exclude_positions':False,
        'pos_to_exclude_from':200,
        'pos_to_exclude_to':1300,
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':'max', #int, or 'max' to get the maximum possible (determined by CCA)
        'TCA_replicates':10,
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':10,
        
        ## LDA params ##
        'LDA_imbalance_prop':.51,
        'LDA_imbalance_repetitions':10,
        'LDA_trial_shuffles':0,
        'LDA_session_shuffles':0,
        'session_comparisons':'BT' #'airpuff', 'BT', 'TP', 'BP'
        }
    
        
    ## Plotting parameters
    fig_num = 1
    fs = 15
    pca_plot_bin_size = 50 #Only used for plotting pca
    
    CCA_random_shifts = 25    #Number of random shifts WARNING: LDA PROB NOT CALCULATED FOR IT!!!
    
    
    
    ########## SIMULATION PARAMETERS ###########
    ## Session generation
    mice_num_per_type = 20
    mouse_types = ['Same', 'Different', 'Alignment shift', 'Poor representation', 'Misaligned']
    # mouse_types = ['Same', 'Different', 'Alignment shift']
    # mouse_types = ['Different']
    # mouse_types = ['Different']

    mouse_type_num = len(mouse_types)
    mice_num = mice_num_per_type * mouse_type_num
    mouse_list = np.arange(mice_num)
    B_session_num = 3
    T_session_num = 3
    P_session_num = 0
    session_num = B_session_num + T_session_num + P_session_num
    session_list = np.arange(session_num)
        
    #Velocity generation
    session_list_to_train_velocity = np.arange(3)
    session_list_to_train_velocity = [0,1,2]
    running_to_train_velocity = False #If True, only "running" bins are used in generating data
    eliminate_v_zeros_to_train_velocity = False
    
    ### Position generation
    dt = 0.01 # time interval, in seconds
    num_neurons = 50 # Number of neurons
    num_trials = 30 # Number of trials per generated session
    
    ### Firing rate generation


    ## Probability weights by position for the firing rate generation
    ## (1) Homogenous
    firing_rate_position_probs = np.ones(max_pos)/max_pos 
    
    ## (2) Reward bias
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[:150] = firing_rate_position_probs[-150:] = 2.
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## (3) Delete sections
    firing_rate_position_probs = np.ones(max_pos) # Probability weights by position for the firing rate generation
    firing_rate_position_probs[250:1250] = 0
    firing_rate_position_probs = firing_rate_position_probs / np.sum(firing_rate_position_probs)
    
    ## Probability weights by position for the firing rate standard deviation
    ## (1) Homogeneous
    firing_rate_std_probs = np.ones(max_pos)/max_pos 
    
    
    # ## (2) Small only
    # firing_rate_std_probs = np.ones(max_pos)
    # firing_rate_std_probs[250:] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)

    # ## (3) Small or big
    # firing_rate_std_probs = np.ones(4*max_pos)
    # firing_rate_std_probs[250:(4*max_pos - 250)] = 0
    # firing_rate_std_probs = firing_rate_std_probs/np.sum(firing_rate_std_probs)
    
    simulated_type1 = 'gaussian'
    firing_rate_kwargs1 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}
    
    simulated_type2 = 'gaussian'
    firing_rate_kwargs2 ={"firing_rate_position_probs":firing_rate_position_probs, "firing_rate_std_probs":firing_rate_std_probs}

    ## Latent parameters    
    error_std1 = .01
    error_std2 = .01

    # error_std = np.array([0.05]*max_pos)
    # error_std[500:1000] = 3.
    
    simulated_type1 = 'latent'
    
    latent_type1 = 'deformed circle 1'
    # latent_type1 = 'deformed circle sigmoid 1'

    firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
    
    simulated_type2 = 'latent'
    
    # latent_type2 = 'deformed circle 2'
    # latent_type2 = 'deformed circle twist 2'
    latent_type2 = 'deformed circle 1'
    # latent_type2 = 'deformed circle sigmoid 2'


    firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
    
    simulated_type_list = [simulated_type1] * B_session_num + [simulated_type2]*T_session_num + [simulated_type1]*P_session_num
    firing_rate_kwargs_list = [firing_rate_kwargs1] * B_session_num + [firing_rate_kwargs2]*T_session_num + [firing_rate_kwargs1]*P_session_num
    
    # firing_

    ########## START SIMULATIONS ###########
    mtype_by_mouse = []
    for mtype in mouse_types:
        mtype_by_mouse.extend([mtype]*mice_num_per_type)
    mtype_by_mouse = np.array(mtype_by_mouse)
    print(mtype_by_mouse)

    ## Store velocities from real data
    compute_and_store_velocity_data(trial_bins, session_list_to_train_velocity, time_bin_size, position_bin_size, gaussian_size, 
                                    data_used, running_to_train_velocity, eliminate_v_zeros=eliminate_v_zeros_to_train_velocity)
    
    

    
    #Generate multiple pairs of sessions    

    
    PCA_analysis_dict = {}
        
    for mnum in range(mice_num):
        
        mtype = mtype_by_mouse[mnum]
        print(mtype)
        if mtype == 'Same':
            error_std1 = .01
            error_std2 = .01
            latent_type1 = 'deformed circle 1'
            firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
            latent_type2 = 'deformed circle 1'
            firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
            firing_rate_kwargs_list = [firing_rate_kwargs1] * B_session_num + [firing_rate_kwargs2]*T_session_num + [firing_rate_kwargs1]*P_session_num

        elif mtype == 'Different':
            error_std1 = .01
            error_std2 = .01
            latent_type1 = 'deformed circle sigmoid 1'
            firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
            latent_type2 = 'deformed circle sigmoid 2'
            firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
            firing_rate_kwargs_list = [firing_rate_kwargs1] * B_session_num + [firing_rate_kwargs2]*T_session_num + [firing_rate_kwargs1]*P_session_num

        elif mtype in ['Misaligned', 'Poor representation', 'Alignment shift']:
            if mtype == 'Poor representation':
                error_std1 = 15
                error_std2 = 15
            else:
                error_std1 = .01
                error_std2 = .01
            latent_type1 = 'deformed circle 1'
            # latent_type1 = 'deformed circle sigmoid 1'
            firing_rate_kwargs1 = {"error_std":error_std1, 'latent_type':latent_type1}
            latent_type2 = 'deformed circle 1'
            # latent_type2 = 'deformed circle sigmoid 2'
            firing_rate_kwargs2 = {"error_std":error_std2, 'latent_type':latent_type2}
            firing_rate_kwargs_list = [firing_rate_kwargs1] * B_session_num + [firing_rate_kwargs2]*T_session_num + [firing_rate_kwargs1]*P_session_num



        #Generate list of sessions
        position_list = []
        pca_list = []
        variance_explained_list = []
        
        for i in range(session_num):
    
            ### Create simulated firing rates
            firing_rate_kwargs = firing_rate_kwargs_list[i]
            simulated_type = simulated_type_list[i]
            firing_rate_funs_by_neuron = create_simulated_firing_rates(num_neurons, simulated_type, **firing_rate_kwargs)
    
    
            # ## Generate a session worth of data, position and spikes
            generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity, plot=False)
            
            
            ## Pre-process generated data
            data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
            data_dict = preprocess_generated_data(data_dict, time_bin_size, position_bin_size, gaussian_size, eliminate_v_zeros)
            pca_input_data = data_dict['amplitudes_binned_normalized']
            position = data_dict['distance']
        
        
            
        
            
            ## Perform PCA
            pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
            ## Perform PCA
            pca = decomposition.PCA(n_components=num_neurons)
            pca.fit(pca_input_data.T)
            
            #PCA over time
            pca_data = pf.project_spikes_PCA(pca_input_data, pca_instance = pca, num_components = num_components)
            variance_explained = pca.explained_variance_ratio_
                
            position_list.append(position)
            pca_list.append(pca_data)
            variance_explained_list.append(variance_explained)
        
        PCA_analysis_dict[mnum, 'position_list'] = position_list
        PCA_analysis_dict[mnum, 'pca_list'] = pca_list
        PCA_analysis_dict[mnum, 'variance_explained_list'] = variance_explained_list
        
        
    ## Align sessions ##
    unaligned_error_array_total = np.zeros((session_num, session_num, 0))
    aligned_error_array_total = np.zeros((session_num, session_num, 0))
    
    CCA_analysis_dict = {}
    
    for mnum in range(mice_num):
        
        mtype = mtype_by_mouse[mnum]
        if mtype == 'Misaligned':
            skip_alignment = True
        elif mtype == 'Alignment shift':
            cca_shuffle = True
        else:
            cca_shuffle = False
            skip_alignment = False
        
        position_list = PCA_analysis_dict[mnum, 'position_list']
        pca_list = PCA_analysis_dict[mnum, 'pca_list']
        variance_explained_list = PCA_analysis_dict[mnum, 'variance_explained_list']
        
        
        #Align the list of sessions
    
        M = session_num
        
        #Set PCA dimension
        pca_list = mCCA_funs.set_dimension_of_pca_list(pca_list, CCA_dim, variance_explained_list)
        
        #Perform mCCA
        pos_list_aligned, pca_dict_aligned, mCCA = mCCA_funs.perform_warped_mCCA(position_list, pca_list, max_pos, warping_bins, warp_based_on, 
                                                                                 return_warped_data, return_trimmed_data, cca_shuffle, skip_alignment)
    
        #Normalize PCA after alignment changes
        pca_dict_aligned = mCCA_funs.normalize_pca_dict_aligned(pca_dict_aligned, mCCA)
        
        #Find space with best alignment
        best_space = mCCA_funs.return_best_mCCA_space(pos_list_aligned, pca_dict_aligned, max_pos=1500, verbose=False)
        pca_list_aligned = pca_dict_aligned[best_space]
                    
        pca_list_unaligned = [pca_dict_aligned[m][m] for m in range(M)]
        unaligned_error_array, aligned_error_array = mCCA_funs.get_cross_prediction_errors(pos_list_aligned, pca_list_unaligned, pos_list_aligned, pca_dict_aligned, 
                                                                                           max_pos, cv_folds, error_type, predictor_name)
        
        print(np.average(unaligned_error_array))
        print(np.average(aligned_error_array))

        CCA_analysis_dict[mnum, 'pos'] = pos_list_aligned
        CCA_analysis_dict[mnum, 'pca'] = pca_list_aligned
        CCA_analysis_dict[mnum, 'pca_unaligned'] = pca_list
        CCA_analysis_dict[mnum, 'pca_dict_aligned'] = pca_dict_aligned
        
        CCA_analysis_dict[mnum, 'best_space'] = best_space
        CCA_analysis_dict[mnum, 'session_list'] = session_list
        CCA_analysis_dict[mnum, 'unaligned_error_array'] = unaligned_error_array
        CCA_analysis_dict[mnum, 'aligned_error_array'] = aligned_error_array  
        CCA_analysis_dict[mnum, 'mCCA_instance'] = mCCA
        
    CCA_analysis_dict['mouse_list'] = mouse_list
    CCA_analysis_dict['num_bins'] = warping_bins
        
            
        
    # # ############# STEP 3: TCA + LDA ############    
    # APdecoding_dict = mf.perform_APdecoding_on_cca_param_dict(CCA_analysis_dict, ap_decoding_param_dict)
    
    
    # session_list = np.arange(session_num)
    TCA_factors = ap_decoding_param_dict['TCA_factors']
    session_comparisons = ap_decoding_param_dict['session_comparisons']
    TCA_replicates = ap_decoding_param_dict['TCA_replicates']
    TCA_method = ap_decoding_param_dict['TCA_method']
    TCA_convergence_attempts = ap_decoding_param_dict['TCA_convergence_attempts']
    TCA_on_LDA_repetitions = ap_decoding_param_dict['TCA_on_LDA_repetitions']
    LDA_imbalance_prop = ap_decoding_param_dict['LDA_imbalance_prop']
    LDA_imbalance_repetitions = ap_decoding_param_dict['LDA_imbalance_repetitions']

    max_tca_on_lda_attempts = 100 #TCA on LDA is repeated until the required number of repetitions is reached, but in case it never converges, this will stop it
    LDA_components = 1

    APdecoding_dict = {}

    for mnum in mouse_list:

        pca_list = CCA_analysis_dict[mnum, 'pca']
        pos_list = CCA_analysis_dict[mnum, 'pos']
                
        data_by_trial, pos_by_trial, snum_by_trial = pf.reshape_pca_list_by_trial(pca_list, pos_list, warping_bins, session_list)
        num_features, num_bins, total_trials = data_by_trial.shape
        num_CCA_dims = num_features
        if TCA_factors == 'max':
            num_TCA_dims = num_CCA_dims
        else:
            num_TCA_dims = int(TCA_factors)
        print('Performing TCA+LDA on M%d // CCA dim: %d, // TCA dim: %d' %(mnum, num_CCA_dims, num_TCA_dims))       
            
        #Selecting trials to decode
        trials_to_keep, label_by_trial = APfuns.get_trials_to_keep_and_labels(snum_by_trial, session_comparisons)
        num_trials_to_keep = len(label_by_trial)
        
        
        label_by_trial_predicted = np.zeros((0, num_trials_to_keep), dtype=int) #1st axis is TCA repetition, 2nd is trial
        
        
        plotted_factors = False
        TCA_on_LDA_counter = 0
        f1_array = np.zeros((0, 2)) #1st axis is TCA repetition, 2nd is class 0 or 1

        while TCA_on_LDA_counter < ap_decoding_param_dict['TCA_on_LDA_repetitions'] and TCA_on_LDA_counter < max_tca_on_lda_attempts:
            # Step 4: TCA
            KTensor = APfuns.perform_TCA(data_by_trial, num_TCA_dims, TCA_replicates, 
                                          TCA_method, TCA_convergence_attempts)
            feature_factors_temp, time_factors_temp, trial_factors_temp = KTensor
            LDA_input = trial_factors_temp[trials_to_keep]           
    
            #LDA on TCA factors
            try:                
                LDA_results_dict = APfuns.perform_LDA(LDA_input, label_by_trial, LDA_components, LDA_imbalance_prop, LDA_imbalance_repetitions)
            except np.linalg.LinAlgError:
                continue
            
            #Update arrays
            f1 = LDA_results_dict['f1']
            # print(f1)
            f1_array = np.vstack((f1_array, f1))
            print(f1)
            TCA_on_LDA_counter += 1
        APdecoding_dict[mnum, 'f1_array'] = f1_array

    print(list(APdecoding_dict.keys()))
    
    #### Process results ####
    f1_avg_by_mouse = np.zeros((len(mouse_list), 2))
    f1_std_by_mouse = np.zeros((len(mouse_list), 2))
    
    for midx, mnum in enumerate(mouse_list):
        f1_array = APdecoding_dict[mnum, 'f1_array']
        
        #Avg F1
        f1_avg_by_mouse[midx] = np.average(f1_array, axis=0)
        f1_std_by_mouse[midx] = np.std(f1_array, axis=0)
        # f1_std_by_mouse[midx] = np.std(f1_array, axis=0)/np.sqrt(f1_array.shape[0])
    
    #VD vs DD summary plot
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(2*mouse_type_num,5), num=fig_num)
    ax = ax[0,0]; fig_num += 1
    
    xpos_list = np.arange(mouse_type_num)
    shift = 0.1
    ymin = 0.5
    ymax = 0.5
    # mouse_types = [0]
    # mtype_by_mouse = np.array([0] * mice_num)
    for mouse_type_idx, mouse_type_label in enumerate(mouse_types):
        m_idxs = mtype_by_mouse == mouse_type_label
        if np.sum(m_idxs) == 0:
            continue
        for class_idx, class_label in enumerate(range(2)):
            perf_avg = np.average(f1_avg_by_mouse[m_idxs, class_idx])
            perf_std = np.average(f1_std_by_mouse[m_idxs, class_idx])
            ymin = np.minimum(ymin, perf_avg-perf_std)
            ymax = np.maximum(ymax, perf_avg+perf_std)
    
            xx = xpos_list[mouse_type_idx] - shift * (1-2*class_idx)
            label=pparam.AP_DECODING_LABELS[class_idx]
            if mouse_type_idx != 0:
                label=None
            color = pparam.AP_DECODING_COLORS[class_idx]
            ax.errorbar(xx, perf_avg, yerr=perf_std, fmt='_', color=color, alpha=0.9, label=label, 
                              markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
            
            
            
            # if decoding_pair_idx == 0:
            #     print(mouse_type_label, class_label, perf_avg)
                
    # perf_avg = np.average(f1_avg_by_mouse[m_idxs, class_idx])
    # perf_std = np.average(f1_std_by_mouse[m_idxs, class_idx])
    # ymax = np.maximum(ymax, perf_avg)

    # xx = xpos_list[mouse_type_idx] - shift * (1-2*class_idx)
    # label=pparam.AP_DECODING_LABELS[class_idx]
    # if mouse_type_idx == 1:
    #     label=None
    # color = pparam.AP_DECODING_COLORS[class_idx]
    # ax.errorbar(xx, perf_avg, yerr=perf_std, fmt='_', color=color, alpha=0.9, label=label, 
    #                   markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
    
    # if decoding_pair_idx == 0:
    #     print(mouse_type_label, class_label, perf_avg)
                
    
    ax.legend(fontsize=15, frameon=False)
    # ax.set_xlim([-shift-0.1, num_mice-(1-shift-0.1)])
    # yminlim = np.minimum(ax.get_ylim()[0], ymin)
    # ymaxlim = np.maximum(ax.get_ylim()[1], ymax)
    # ax.set_ylim([yminlim, ymaxlim])
        
    ax.legend(fontsize=15, loc = 'upper right', frameon=False)

    xlabels = mouse_types
    ax.set_xticks(xpos_list, xlabels, fontsize=fs)
    decoding_label = 'F1'
    ax.set_ylabel("%s score"%decoding_label, fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)   

    xlims = ax.get_xlim()
    ax.plot(xlims, [0.5, 0.5], '--k', alpha=0.5)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    fig.tight_layout()

def compute_position_prediction_error_on_generated_data():
    ########## PARAMETERS ###########
    
    #General
    max_pos = pparam.MAX_POS
    trial_bins = 30
       
    #Data preprocessing
    time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    position_bin_size = 1  # mm, track is 1500mm, data is in mm
    gaussian_size = 25  # Why not
    data_used = 'amplitudes'
    running = True
    eliminate_v_zeros = True
    
    
    #Velocity generation
    session_list_to_train_velocity = np.arange(3)
    # session_list_to_train_velocity = [3,4,5,6]
    running_to_train_velocity = False #If True, only "running" bins are used in generating data
    eliminate_v_zeros_to_train_velocity = False
    
    ### Position generation
    dt = 0.01 # time interval, in seconds
    num_neurons = 50 # Number of neurons
    num_trials = 50 # Number of trials per generated session
    firing_rate_position_probs = None
    firing_rate_std_probs = None
    
    ## Predictor parameters ##
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    number_of_generated_sessions = 25
    
    ## Plotting parameters
    fig_num = 1
    fs = 15
    pca_plot_bin_size = 30 #Only used for plotting pca
    
    session_types = pparam.SESSION_SNUM_BY_LABEL.keys()
    error_array = np.zeros((len(session_types), number_of_generated_sessions))
    for stype_num, stype in enumerate(session_types):
        session_list_to_train_velocity = pparam.SESSION_SNUM_BY_LABEL[stype]
        print(session_list_to_train_velocity)
    
    
        ## Store velocities from real data
        compute_and_store_velocity_data(trial_bins, session_list_to_train_velocity, time_bin_size, position_bin_size, gaussian_size, 
                                        data_used, running_to_train_velocity, eliminate_v_zeros=eliminate_v_zeros_to_train_velocity)
        
        error_list = []
        for gen_num in range(number_of_generated_sessions):            
            ### Create simulated firing rates
            firing_rate_funs_by_neuron = create_simulated_firing_rates(num_neurons, firing_rate_position_probs, firing_rate_std_probs, max_pos)

            # ## Generate a session worth of data, position and spikes
            generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity, plot=True)
            
            ## Pre-process generated data
            data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
            data_dict = preprocess_generated_data(data_dict, time_bin_size, position_bin_size, gaussian_size, eliminate_v_zeros)
            pca_input_data = data_dict['amplitudes_binned_normalized']
            position = data_dict['distance']
        
        
            
        
            
            ## Perform PCA
            pca = pf.project_spikes_PCA(pca_input_data, num_components = 3)
            position, pca, _ = pf.warping(position, pca, 150, max_pos=pparam.MAX_POS, 
                                          warp_sampling_type = 'interpolation', warp_based_on = 'time', return_flattened=True)
            
        
            
            #Predict position
            pos_pred, error, predictor = pf.predict_position_CV(pca, position, n_splits=cv_folds, shuffle=False, periodic=True, pmin=0, pmax=pparam.MAX_POS,
                                    predictor_name=predictor_name, predictor_default=None, return_error=error_type)
            
            error_list.append(error)
            
        error_array[stype_num] = error_list
        
    print(np.average(error_array, axis=1))
    
    ########### PLOTS ###########
    #Plot result by session type
    fig = plt.figure(fig_num, figsize=(5,3)); fig_num += 1
    ax = plt.gca()
    xpos_list = np.arange(len(session_types))
        
    error_avg = np.average(error_array, axis=1)
    error_std = np.std(error_array, axis=1)

    ax.errorbar(xpos_list, error_avg, yerr=error_std, color='black',
                fmt='_', markersize=20, markeredgewidth=3, elinewidth = 3, zorder=2)
       
    xlabels = session_types
    ax.set_xticks(xpos_list, xlabels, fontsize=fs)
    ax.set_ylabel("SSE (cm)", fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)  
    ax.tick_params(axis='x', labelsize=fs)  

    ymin = 0
    # ymin = np.minimum(ymin, ax.get_ylim()[0])
    ax.set_ylim([ymin, ax.get_ylim()[1]])
   
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.xaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)
    ax.yaxis.set_tick_params(width=pparam.AXIS_WIDTH, length=pparam.TICKS_LENGTH)    
    
    
    
    ## Plot PCA
    fig = plt.figure().add_subplot(projection='3d'); fig_num += 1
    angle = 75
    angle_azim = -90
    rows = 2 #One row per session
    cols = 1 #raw data and trial averaged
    fig, axs = plt.subplots(rows, cols, subplot_kw={"projection": "3d"}, figsize=(9,7)) 
    
    #Plot all trials
    ax = axs.ravel()[0]
    pf.plot_pca_with_position(pca, position, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, scatter=True, cbar=False, cbar_label='Position (mm)',
                                alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                ms = 10, lw=3)

    #Plot trial average

    
    ax = axs.ravel()[1]
    position_unique, pca_average, pca_std = pf.compute_average_data_by_position(pca, position, position_bin_size=pca_plot_bin_size, max_pos=max_pos)
    pf.plot_pca_with_position(pca_average, position_unique, ax=ax, max_pos = pparam.MAX_POS, cmap_name = pparam.PCA_CMAP, fs=15, cbar=False, cbar_label='Position (mm)',
                                alpha=1, angle=angle, angle_azim=angle_azim, axis = 'off', show_axis_labels=False, axis_label=None, 
                                scatter=False, ms = 250, lw=6)
    

    ### PLOT POSITION PREDICTION ###
    markersize=50
    fig_num += 1
    fig = plt.figure(fig_num, figsize=(7,4)); fig_num += 1
    ax = plt.gca()
    
    timesteps = np.arange(len(position))

    for plot_idx, p in enumerate([position, pos_pred]):
        label = pparam.PREDICTION_LABELS[plot_idx]
        color = pparam.PREDICTION_COLORS[label]
        ax.scatter(timesteps, p, s=markersize, color=color, label=label)
        
    # ax.scatter(timesteps, pos_pred_shuffle, s=markersize, color='red', label='shuffle')

    ax.set_xlim([0,2000])
        
    ax.set_xlabel('Timestep', fontsize=fs+4)
    # ax.set_yticks(np.arange(len(neurons_to_plot)), neurons_to_plot)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.set_ylabel('Position (mm)', fontsize=fs+4)
    ax.set_title('Generated session, SSE=%.1f (cm)'%(error), 
                 pad=20, fontsize=fs+4)
    ax.legend(fontsize=fs-4, loc='upper right', frameon=False)
    fig.tight_layout()
    


    
def generate_inhomogeneous_poisson_spikes(rate_list, dt=0.001):
    """
    Generate spike times for a neuron with a time-dependent firihccccbccgfcgtdchr7ycrrccyfjcdgdp9udtng rate using an inhomogeneous Poisson process.
    
    Parameters:
    rate_list (list or array): gives the firing rate at time t (spikes per second) at each time step.
    dt (float): Time step for simulation (seconds).
    
    Returns:
    spike_times (list): List of spike times, given by bin indexes
    """
    spike_times = []
    bin_idx = 0
    total_bins = len(rate_list)
    
    while bin_idx < total_bins:
        rate = rate_list[bin_idx]
        if rate * dt > np.random.rand():
            spike_times.append(bin_idx)
        bin_idx += 1
    
    return spike_times

def generate_inhomogeneous_poisson_spikes_old(rate_func, T, dt=0.001):
    """
    Generate spike times for a neuron with a time-dependent firihccccbccgfcgtdchr7ycrrccyfjcdgdp9udtng rate using an inhomogeneous Poisson process.
    
    Parameters:
    rate_func (function): Function that gives the firing rate at time t (spikes per second).
    T (float): Total duration of the simulation (seconds).
    dt (float): Time step for simulation (seconds).
    
    Returns:
    spike_times (list): List of spike times.
    """
    spike_times = []
    t = 0
    
    while t < T:
        rate = rate_func(t)
        if rate * dt > np.random.rand():
            spike_times.append(t)
        t += dt
    
    return spike_times


    
def rate_func_example(t):
    return 5 + 2 * np.sin(2 * np.pi * t / 5)  



def compute_and_store_velocity_data(trial_bins, session_list = np.arange(9), time_bin_size=1, distance_bin_size=1, gaussian_size=25, 
                                    data_used='amplitudes',running=True, eliminate_v_zeros=True):
    
    ########## PARAMETERS ###########
    max_pos = pparam.MAX_POS
    # trial_bins = 30
       
    mouse_list = np.arange(8)
    
    # time_bin_size = 1  # Number of elements to average over, each dt should be ~65ms
    # distance_bin_size = 1  # mm, track is 1500mm, data is in mm
    # gaussian_size = 25  # Why not
    # data_used = 'amplitudes'
    # running = True
    # eliminate_v_zeros = True
    
    velocity_by_bin_total = np.zeros((trial_bins, 0))

    for mnum in mouse_list:
        for snum in session_list:
    
            #Get data
            data_dict = pf.read_and_preprocess_data(fat_cluster, mnum, snum, gaussian_size, time_bin_size, distance_bin_size, 
                                                    only_running=running, eliminate_v_zeros=eliminate_v_zeros, pos_max=pparam.MAX_POS)
            pca_input_data, position, times = pf.get_data_from_datadict(data_dict, data_used)
    
            #Get velocity
            velocity = pf.compute_velocity(position, pos_max=max_pos)
            pos_by_trial, velocity_by_trial, sampling_warped = pf.warping(position, velocity.reshape((1, len(velocity))), trial_bins, warp_sampling_type='averaging', warp_based_on='position')
            velocity_by_trial = np.squeeze(velocity_by_trial)
            
            #Store velocity
            velocity_by_bin_total = np.hstack((velocity_by_bin_total, velocity_by_trial))
        
    #Save
    sessions_str = ', '.join([str(x) for x in session_list])
    np.save(pparam.OUTPUT_PATH + "velocity_by_bin_total_%s_%d.npy"%(sessions_str, int(trial_bins)), velocity_by_bin_total)

    
    

def create_simulated_firing_rates(num_neurons, simulated_type = 'gaussian', **kwargs):
    '''
        Returns a list of "num_neurons" simulated firing rates by position.
        simulated_type: kind of simulated firing rates
            - gaussian: each neuron's firing rate is taken from a gaussian distribution.
                kwargs: "firing_rate_position_probs" - prob weights for the mean gaussian firing rate
                        "firing_rate_std_probs" - prob weights for the gaussian std of the firing rate
            - latent: each neuron's firing rate is defined by a latent lowD space
    
    '''
    max_pos = pparam.MAX_POS
        
    if simulated_type == 'gaussian':
        
        #Create gaussian firing rate maps
        def create_gaussian_fun(mean=0, std=1, amplitude=1):
            def gaussian(x):
                return amplitude * np.exp(-1 * ((x-mean)**2)/(2*(std**2)))
            
            # def gaussian(x):
            #     constant_lim = 750
            #     if type(x) not in [list, np.ndarray]:
            #         if x <= constant_lim:
            #             return amplitude/3
            #         elif x > constant_lim:
            #             return amplitude * np.exp(-1 * ((x-mean)**2)/(2*(std**2)))
            #     else:
            #         xc = np.copy(x)
            #         xc[xc <= constant_lim] = 1
            #         xc[xc > constant_lim] = amplitude * np.exp(-1 * ((xc[xc > constant_lim]-mean)**2)/(2*(std**2)))
            #         return xc
            return gaussian
        
        firing_rate_funs_by_neuron = []
        
        if "firing_rate_position_probs" not in kwargs or kwargs["firing_rate_position_probs"] is None:
            firing_rate_position_probs = np.ones(max_pos)/max_pos
        else:
            firing_rate_position_probs = kwargs["firing_rate_position_probs"]
            
        if "firing_rate_std_probs" not in kwargs or kwargs["firing_rate_std_probs"] is None:
            firing_rate_std_probs = np.ones(max_pos)/max_pos
        else:
            firing_rate_std_probs = kwargs["firing_rate_std_probs"]

        
        
        for n in range(num_neurons):
            # mean = np.random.uniform(0, max_pos)
            mean = np.random.choice(np.arange(max_pos), size=1, replace=True, p=firing_rate_position_probs)
            # std = np.random.uniform(0, max_pos)
            std = np.random.choice(np.arange(len(firing_rate_std_probs)), size=1, replace=True, p=firing_rate_std_probs)
    
            amplitude = np.random.uniform(0, 10)
            firing_rate_fun = create_gaussian_fun(mean, std, amplitude)
            firing_rate_funs_by_neuron.append(firing_rate_fun)
            
    elif simulated_type == 'latent':
        ''' Generate firing rates from a low dimensional latent structure.
            kwargs must have:
                - latent_type: string specifying the kind of latent
                - error_std: standard deviation of the error. Can be single number of list of length "max_pos"
                
            '''
            
        #Create low d ring
        tt = np.arange(max_pos)
        if kwargs['latent_type'] == 'deformed circle 1':
            x = np.cos(2*np.pi*tt/max_pos)
            y = np.sin(2*np.pi*tt/max_pos)
            # z = np.cos(2*np.pi*tt/max_pos)**2
            z =[0]*len(tt)

            latent_array = np.vstack((x,y,z))

        elif kwargs['latent_type'] == 'deformed circle 2':
            x = np.cos(2*np.pi*((tt/max_pos)))
            y = 4*np.sin(2*np.pi*((tt/max_pos)**3))
            # z = np.cos(2*np.pi*((tt/max_pos)**2))**2
            z =[0]*len(tt)
            latent_array = np.vstack((x,y,z))

        elif kwargs['latent_type'] == 'deformed circle 3':
            x = np.cos(2*np.pi*tt/max_pos) + 0.5*np.cos(2*np.pi*((tt/max_pos)**2))
            y = np.sin(2*np.pi*tt/max_pos) + 0.5*np.sin(2*np.pi*((tt/max_pos)**2))
            z = np.cos(2*np.pi*tt/max_pos)**2
            latent_array = np.vstack((x,y,z))
            
        elif kwargs['latent_type'] == 'deformed circle sigmoid 1':
            sigmoid_center = max_pos/4
            sigmoid_width = max_pos/8
            tt = max_pos * 1/(1 + np.exp(- (tt - sigmoid_center)/sigmoid_width))
            x = np.cos(2*np.pi*tt/max_pos)
            y = np.sin(2*np.pi*tt/max_pos)
            # z = np.cos(2*np.pi*tt/max_pos)**2
            z =[0]*len(tt)
            latent_array = np.vstack((x,y,z))

            
        elif kwargs['latent_type'] == 'deformed circle sigmoid 2':
            sigmoid_center = 3 * max_pos/4
            sigmoid_width = max_pos/8
            tt = max_pos * 1/(1 + np.exp(- (tt - sigmoid_center)/sigmoid_width))
            x = np.cos(2*np.pi*tt/max_pos)
            y = np.sin(2*np.pi*tt/max_pos)
            # z = np.cos(2*np.pi*tt/max_pos)**2
            z =[0]*len(tt)
            latent_array = np.vstack((x,y,z))

        elif kwargs['latent_type'] == 'deformed circle twist':
            x = np.cos(2*np.pi*tt/max_pos)
            y = np.sin(2*np.pi*tt/max_pos)
            # z =[0]*len(pp)
            z = np.cos(2*np.pi*tt/max_pos)

            latent_array = np.vstack((x,y,z))            
                        
            for p in tt:
                ang = 2*np.pi*p/max_pos
                Rx = np.array([[1,0,0],[0, np.cos(ang), -np.sin(ang)],[0, np.sin(ang), np.cos(ang)]])
                # Ry = np.array([[np.cos(ang), 0, np.sin(ang)],[0,1,0],[-np.sin(ang), 0, np.cos(ang)]])
                # Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0, 0, 1]])

                Rot_array = Rx
                # Rot_array = np.dot(Rot_array, Ry)
                # Rot_array = np.dot(Rot_array, Rz)

                latent_array[:,p] = np.dot(latent_array[:,p], Rot_array)
                diff = pf.get_periodic_difference(float(p), max_pos/4, max_pos)
                latent_array[2,p] += 0.5*(np.exp(-1 * ((diff**2)/(2*(200**2)))))
                
        elif kwargs['latent_type'] == 'deformed circle twist 2':
            
            
            x = np.cos(2*np.pi*tt/max_pos)
            y = np.sin(2*np.pi*tt/max_pos)
            # z =[0]*len(pp)
            z = np.cos(2*np.pi*tt/max_pos)

            latent_array = np.vstack((x,y,z))           
                        
            for p in tt:
                ang = 2*np.pi*p/max_pos
                Rx = np.array([[1,0,0],[0, np.cos(ang), -np.sin(ang)],[0, np.sin(ang), np.cos(ang)]])
                Ry = np.array([[np.cos(ang), 0, np.sin(ang)],[0,1,0],[-np.sin(ang), 0, np.cos(ang)]])
                # Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0,0,1]])
                Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0, 0, 1]])

                
                Rot_array = Rx
                # Rot_array = np.dot(Rot_array, Ry)
                # Rot_array = np.dot(Rot_array, Rz)

                latent_array[:,p] = np.dot(latent_array[:,p], Rot_array)
                diff = pf.get_periodic_difference(float(p), max_pos/4, max_pos)
                latent_array[2,p] += 2*(np.exp(-1 * ((diff**2)/(2*(200**2)))))
            
        elif kwargs['latent_type'] == 'deformed circle double':
            x = np.cos(4*np.pi*tt/max_pos)
            y = np.sin(4*np.pi*tt/max_pos)
            z = np.cos(4*np.pi*tt/max_pos)**2
            latent_array = np.vstack((x,y,z))

        #Generate latent weights for each neuron
        latent_dim = latent_array.shape[0]
        neuron_weight_array = np.random.normal(loc=0, scale=1, size=(num_neurons, latent_dim))
        
        firing_rate_funs_by_neuron = []
        error_std = kwargs['error_std']
        if type(error_std) in [int,float]:
            error_std = [error_std]*max_pos
        for n in range(num_neurons):
            def firing_rate_from_latent_space(pos, n=0):
                pos = int(pos%1500)
                return np.dot(latent_array[:,pos], neuron_weight_array[n]) + np.random.normal(loc=0, scale=error_std[pos])
            firing_rate_fun = partial(firing_rate_from_latent_space, n=n)
            firing_rate_funs_by_neuron.append(firing_rate_fun)
            
                
        

    else:
        raise ValueError("WARNING: no accepted simulated_type name was given")
        
    return firing_rate_funs_by_neuron


    
def generate_session_data(dt, num_trials, trial_bins, firing_rate_funs_by_neuron, session_list_to_train_velocity = [0,1,2], plot=True):
    ''' Generate one session of position and spike amplitude readings
        dt: time step
        N: number of neurons
        num_trials: number of trials that the simulation will go through
        trial_bins: number of bins in each trial (should be the same as used during generation)
        firing_rate_funs_by_neuron: list of functions over position with each simulated firing rate
        session_list_to_train_velocity: sessions used to train the velocity generator
        firing_rate_position_probs: probability weights for each position
    '''
    max_pos = pparam.MAX_POS
    num_neurons = len(firing_rate_funs_by_neuron)
    
    #Load velocity data
    sessions_str = ', '.join([str(x) for x in session_list_to_train_velocity])

    velocity_by_trial = np.load(pparam.OUTPUT_PATH + "velocity_by_bin_total_%s_%d.npy"%(sessions_str, int(trial_bins)), allow_pickle=True)[()]
    

    #Step 1: Generate positions
    velocity_avg_by_bin = np.average(velocity_by_trial, axis=1)
    velocity_std_by_bin = np.std(velocity_by_trial, axis=1)
    velocity_generator = create_velocity_generator(velocity_avg_by_bin, velocity_std_by_bin, max_pos=max_pos)
    position_gen, times_gen = simulate_position_trials(velocity_generator, dt=dt, max_trials=num_trials, max_pos=max_pos)
    simulation_duration = len(position_gen)
        
    #Step 2: generate simulated amplitudes
    gaussian_window = 10
    amplitude_array = np.zeros((num_neurons, simulation_duration))
    spikes_array = np.zeros((num_neurons, simulation_duration))

    for n in range(num_neurons):
        firing_rate_fun = firing_rate_funs_by_neuron[n]
        rate_by_time = [firing_rate_fun(p) for p in position_gen]
        spike_times = generate_inhomogeneous_poisson_spikes(rate_by_time, dt=dt)
        spike_onehot = np.zeros(simulation_duration)
        spike_onehot[spike_times] = 1
        
        gx = np.arange(-3*gaussian_window, 3*gaussian_window, 1)
        gaussian = np.exp(-(gx/gaussian_window)**2/2)
        amplitude_simulated = convolve(spike_onehot, gaussian, mode="same")
        amplitude_array[n,:] = amplitude_simulated
        spikes_array[n, :] = spike_onehot
    
    data_dict = {'position':position_gen,
                 'amplitudes':amplitude_array,
                 'times':times_gen,
                 'spikes':spikes_array
                 }    
    np.save(pparam.OUTPUT_PATH + "generated_session.npy", data_dict)
    
    #Plot
    if plot == True:
        fig_num = 1
        fig, axs = plt.subplots(5,5, figsize=(8,8)); fig_num += 1
        fs = 12
        for n in range(25):
            ax = axs.ravel()[n]
            xx = np.linspace(0, max_pos, 100)
            firing_rate_fun = firing_rate_funs_by_neuron[n]
            firing_rate_list = [firing_rate_fun(p) for p in xx]
            ax.plot(xx, firing_rate_list, lw=2, color='black')
            # ax.set_ylabel('Var. explained (ratio)', color=color1, fontsize=fs)
            ax.tick_params(axis='x', labelsize=fs)                
            ax.tick_params(axis='y', labelsize=fs)
            ax.set_xlim([0,1500])
            
            if n == 0:
                ax.set_ylabel('Firing rate (Hz)', fontsize=fs)
            if n == 20:
                ax.set_xlabel('Position (mm)', fontsize=fs)
        fig.tight_layout()

    return data_dict



def preprocess_generated_data(data_dict, time_bin_size=1, position_bin_size=1, gaussian_size=25, eliminate_v_zeros=True):
    ## Parameters ##
    max_pos = pparam.MAX_POS
    
    data_dict = np.load(OUTPUT_PATH +"generated_session.npy", allow_pickle=True)[()]
    
    position = np.array(data_dict['position'])
    amplitudes = data_dict['amplitudes']
    times = np.array(data_dict['times'])
    

    
    #Spike pre-processing (smoothing, binning)
    data = pf.smoothing(amplitudes.astype(float), gaussian_size, axis=1)
    data = pf.sum_array_by_chunks(data, time_bin_size, 1)/time_bin_size # num neurons X num time bins

    #Times
    times = np.array(times)
    dt = np.average(times[1:] - times[:-1]) #Time interval in miliseconds
    
    #Position
    position = position % 1500 #Eliminates negative distances, which I've seen at least in animal 5 session 2 for some reason
    position = pf.sum_array_by_chunks(position, time_bin_size, 0)/time_bin_size
    position = (position // position_bin_size) * position_bin_size
                
    if eliminate_v_zeros == True:
        position_original = np.copy(position)
        position, data_cut, velocity = pf.compute_velocity_and_eliminate_zeros(position_original, data, pos_max=max_pos)
        data = data_cut
            
        times = np.arange(len(position)) * dt


    
    data_name = 'amplitudes'
    data_normalized, data_mean, data_std = pf.normalize_data(data, axis=1)
    data_dict[data_name + '_binned_normalized'] = data_normalized*1
    data_dict[data_name + '_mean'] = data_mean
    data_dict[data_name + '_std'] = data_std
        
    #Position-related values    
    pos_diff = np.diff(position)
    overround = [0] + list(np.where(pos_diff < -1000)[0]+1)
    num_trials = len(overround)-1
    position_by_trial = [position[overround[i] : overround[i+1]] for i in range(num_trials)]           

    
    data_dict['dt'] = dt
    data_dict['distance'] = position    
    data_dict['overround'] = overround
    data_dict['num_trials'] = num_trials
    data_dict['distance_by_trial'] = position_by_trial
        
    return data_dict


    

    
    
def create_velocity_generator(velocity_avg_by_bin, velocity_std_by_bin, max_pos=1500):
    num_bins = velocity_avg_by_bin.size
    bin_list = np.arange(0, max_pos, int(max_pos/num_bins))
    def velocity_generator(pos):
        ''' Indexing works assuming that bin_list starts at 0 '''
        bin_idx = bisect.bisect_right(bin_list, pos)-1
        return np.random.normal(loc=velocity_avg_by_bin[bin_idx], scale=velocity_std_by_bin[bin_idx])
    return velocity_generator

def simulate_position_trials(velocity_generator, dt= 0.01, max_trials=1, max_pos=1500):
    pos = 0
    trial_number = 0
    time = 0
    
    pos_list = [0]
    time_list = [0]
    
    # num_random_stops = np.random.choice([0,1,2,3])
    # vel_multiplier = [1]*max_pos
    # for stop in range(num_random_stops):
    #     xx = np.arange(max_pos)
    #     stop_pos = np.random.uniform(0,1)*max_pos
    #     stop_std = np.random.uniform(0,1)*max_pos*0.75
    #     vel_multiplier_current = (1/num_random_stops) * np.exp(-1 * ((xx-stop_pos)**2)/(2*(stop_std**2)))
    #     vel_multiplier += vel_multiplier_current
    # vel_multiplier = np.array(vel_multiplier)
    # # vel_multiplier = 1/vel_multiplier
    
        
    
    while trial_number < max_trials:
        
        num_random_stops = np.random.choice([0,1,2,3])
        vel_multiplier = [1]*max_pos
        for stop in range(num_random_stops):
            xx = np.arange(max_pos)
            stop_pos = np.random.uniform(0,1)*max_pos
            stop_std = np.random.uniform(0,1)*max_pos*0.25
            vel_multiplier_current = (5/num_random_stops) * np.exp(-10 * ((xx-stop_pos)**2)/(2*(stop_std**2)))
            vel_multiplier += vel_multiplier_current
        vel_multiplier = 1/np.array(vel_multiplier)
        
        while pos < max_pos:
            v = velocity_generator(pos) * vel_multiplier[int(pos)]
            pos += v
            time += dt
            
            pos_list.append(pos%max_pos)
            time_list.append(time)
            
        pos = pos%max_pos
        trial_number += 1
            
    return pos_list, time_list
        
        


with h5py.File(FAT_CLUSTER_PATH, 'r') as fat_cluster:

    if __name__ == '__main__':
        tt = time.time()
        main()
        # np.random.seed(9)
        print('Time Ellapsed: %.1f' % (time.time() - tt))
        plt.show()

