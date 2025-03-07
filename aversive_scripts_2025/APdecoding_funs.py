# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:05:58 2024

@author: Albert

Functions to perform TCA, LDA, and one after the other
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import tensortools

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import project_parameters as pparam
import processing_functions as pf

def main():
    test_class_inequality()
    # test_LDA_weight_function()
    # test_TCA_in_false_data()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Analysis functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
def perform_TCA(data_by_trial, TCA_factors, TCA_replicates, TCA_method, TCA_convergence_attempts, TCA_replicate_selection=0,
                verbose=False, return_ensemble=False):
    ''' Performs TCA on a data matrix 
        data_by_trial: array of size "num features" X "num time bins" X "num trials"
        TCA_factors: number of TCA dimensions for each factor
        TCA_replicates: number of ensembles (recommended high, otherwise it won't converge well)
        TCA_method: "cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        TCA_convergence_attempts: how many times to try TCA. Will raise an error if it hasn't converged after enough attempts
        TCA_replicate_selection: index of the replicate to get (defaults to 0)
    '''
    TCA_attempts_counter = 0

    while TCA_attempts_counter < TCA_convergence_attempts:
        try:
            TCA_ensemble = tensortools.Ensemble(fit_method=TCA_method)
            TCA_ensemble.fit(data_by_trial, ranks=range(1, TCA_factors+1), replicates=TCA_replicates, verbose=verbose)
        except IndexError:
            TCA_attempts_counter += 1  
        except np.linalg.LinAlgError:
            TCA_attempts_counter += 1  
        except Exception as error:
            print('Unexpected error', error)
            raise error 
        else:
            break
        
    if TCA_attempts_counter == TCA_convergence_attempts:
        print('WARNING: TCA did not converge!')

    # num_components = TCA_factors
    KTensor = TCA_ensemble.factors(TCA_factors)[TCA_replicate_selection]
    
    if not return_ensemble:
        return KTensor
    else:
        return KTensor, TCA_ensemble
    
    
    
def get_LDA_axis(LDA_obj, num_features):
    #Get transformation matrix
    ID = np.eye(num_features)
    LDA_ax = np.zeros(num_features)
    for dim_idx, euler_vector in enumerate(ID):
        euler_vector = euler_vector.reshape(1, num_features)
        euler_proj = LDA_obj.transform(euler_vector).squeeze()
        LDA_ax[dim_idx] = euler_proj
    LDA_ax = LDA_ax/np.linalg.norm(LDA_ax)
    assert np.allclose(2*euler_proj, LDA_obj.transform(2*euler_vector).squeeze())
    return LDA_ax

def perform_LDA(LDA_input, labels, LDA_components = 1, imbalance_prop = 0.6, imbalance_repetitions = 25):
    ''' LDA_input: size "num samples" X "num features"
        labels: size "num samples". ASSUMES THEY ARE "INT" THAT START AT 0 AND INCREASE 1 BY 1, NO NEGATIVES OR DECIMALS
        LDA_components: int
        imbalance_prop: float between 0 and 1 (if the proportion of a class is larger than this, deal with data imbalance during training)
    '''
    
    num_features = LDA_input.shape[1]
    num_samples = len(labels)
    unique_labels = np.unique(labels)
    
    samples_by_label = [np.sum(labels==label) for label in unique_labels]
    prop_by_label = [samples_by_label[lidx]/num_samples for lidx in range(len(unique_labels))]
    majority_class_idx, majority_prop = np.argmax(prop_by_label), np.max(prop_by_label)

    LDA_results_dict = {}
    LDA_projection = np.zeros((num_samples, LDA_components))
    label_predicted = np.zeros(num_samples, dtype=int)
    LDA_prob_correct = np.zeros(num_samples)
    
    
    if not majority_prop > imbalance_prop:
        selected_trials_list = [np.ones(num_samples, dtype=bool)]
        imbalance_repetitions = 1
        
    else:
        #Randomly select trials from the most common class. To ensure all trials appear at least once, after every
        #randomization we create a new one that includes all unused trials (plus old ones until the required number of samples is reached)
        selected_trials_list = []
        #How many subsets of the larger class of size equal to the smaller class, are needed to sample all larger class trials at least once?
        min_selections_needed_to_include_all_samples = int(np.ceil(np.max(samples_by_label)/np.min(samples_by_label))) 
        #Round up to the nearest number multiple of "min_selections_needed_to_include_all_samples"
        imbalance_repetitions = int(min_selections_needed_to_include_all_samples * (np.ceil(imbalance_repetitions/min_selections_needed_to_include_all_samples))) 

        for rep in range(int(imbalance_repetitions/min_selections_needed_to_include_all_samples)):
            majority_class = unique_labels[majority_class_idx]
            majority_bool = labels == majority_class
            minority_bool = labels != majority_class
            minority_samples = np.min(samples_by_label)
            majority_idxs = np.where(majority_bool)[0]            
            majority_idxs_shuffled = np.random.choice(majority_idxs, len(majority_idxs), replace=False)
            for subset in range(min_selections_needed_to_include_all_samples):
                majority_idxs_selected = majority_idxs_shuffled[subset * minority_samples : (subset+1)*minority_samples]
                #The last iteration will have the remaining unselected trials, which might be less than the minority samples
                #Last iteration will have remaining unselected trials. If they are less than minority, will with previously used trials (randomly chosen)
                if len(majority_idxs_selected) != minority_samples:
                    already_used_idxs = majority_idxs_shuffled[:subset * minority_samples]
                    filler_idxs = np.random.choice(already_used_idxs, minority_samples - len(majority_idxs_selected), replace=False)
                    majority_idxs_selected = np.hstack((majority_idxs_selected, filler_idxs))
                selected_trials = np.copy(minority_bool)
                selected_trials[majority_idxs_selected] = True
                selected_trials_list.append(selected_trials)
        
        
    #Note: not all samples will be used on every repetition, we later restrict which part of the matrices we use
    label_by_rep = -np.ones((imbalance_repetitions, num_samples), dtype=int) #Default label is -1
    proj_by_rep = np.zeros((imbalance_repetitions, num_samples, LDA_components))
    prob_by_rep = np.zeros((imbalance_repetitions, num_samples))
    f1_by_rep = np.zeros((imbalance_repetitions, len(unique_labels))) #one per class
    acc_by_rep = np.zeros((imbalance_repetitions, len(unique_labels))) #one per class
    
    weights_by_rep = np.zeros((imbalance_repetitions, num_features))-1 #One per feature and sample
    
    
    for rep in range(imbalance_repetitions):
        selected_trials_bool = selected_trials_list[rep]   
        num_samples_by_rep = np.sum(selected_trials_bool)
        
        weights_by_fold = np.zeros((num_samples_by_rep, num_features)) #Each training we get weights for all features
        for sample_idx in range(num_samples_by_rep):
            selected_trials_bool_current = np.copy(selected_trials_bool)
            test_idx = np.where(selected_trials_bool)[0][sample_idx]
            selected_trials_bool_current[test_idx] = False

            
            LDA_train_input = LDA_input[selected_trials_bool_current]
            LDA_train_label = labels[selected_trials_bool_current]
            
            LDA_test_input = LDA_input[[test_idx]]
            LDA_test_label = labels[test_idx]
            
            LDA = LinearDiscriminantAnalysis(solver="eigen", #svd, lsqr, eigen
                                              shrinkage=0,  #None, "auto", float 0-1
                                              n_components=LDA_components, #Dimensionality reduction
                                              store_covariance=False #Only useful for svd, which doesn't automatically calculate it
                                              )
            LDA.fit(LDA_train_input, LDA_train_label)
            
        
            proj_by_rep[rep, test_idx] = np.squeeze(LDA.transform(LDA_test_input))
            label_by_rep[rep, test_idx] = (LDA.predict(LDA_test_input))
            prob_by_rep[rep, test_idx] = (LDA.predict_proba(LDA_test_input)[0,LDA_test_label]) #WARNING: assumes label corresponds to label index
            weights_by_fold[sample_idx, :] = np.abs(LDA.coef_)
            # weights_by_fold[sample_idx, :] = LDA.coef_



        selected_labels = labels[selected_trials_bool]
        pred_labels = label_by_rep[rep, selected_trials_bool]
        
        #F1
        f1 = pf.multiclass_f1(selected_labels, pred_labels)
        f1_by_rep[rep] = f1
        
        #Accuracy
        acc_list = []
        for label in unique_labels:
            label_bool = selected_labels == label
            acc = np.sum(pred_labels[label_bool] == selected_labels[label_bool])/np.sum(label_bool)
            acc_list.append(acc)
            
            
        # acc = np.sum(pred_labels == labels[selected_trials_bool])/num_samples_by_rep
        acc_by_rep[rep] = acc_list
        weights_by_rep[rep] = np.average(weights_by_rep[rep], axis=0)
        weights_by_rep[rep] = np.average(weights_by_fold, axis=0)




    #Average repetitions
    f1_avg = np.average(f1_by_rep, axis=0)
    acc_avg = np.average(acc_by_rep, axis=0)
    weights = np.average(weights_by_rep, axis=0)
    for sample_idx in range(num_samples):
        sample_labels = label_by_rep[:, sample_idx]
        prediction_label_count = [np.sum(sample_labels == label) for label in unique_labels]
        most_predicted_label = unique_labels[np.argmax(prediction_label_count)]
    
        rep_idxs_of_most_predicted_label = sample_labels == most_predicted_label
        LDA_projection[sample_idx,:] = np.average(proj_by_rep[rep_idxs_of_most_predicted_label])
        LDA_prob_correct[sample_idx] = np.average(prob_by_rep[rep_idxs_of_most_predicted_label])
        label_predicted[sample_idx] = most_predicted_label

    LDA_results_dict['f1'] = f1_avg
    LDA_results_dict['accuracy'] = acc_avg
    LDA_results_dict['label_predicted'] = label_predicted
    LDA_results_dict['LDA_prob'] = LDA_prob_correct
    LDA_results_dict['weights'] = weights
    LDA_results_dict['LDA_projection'] = LDA_projection
    
    return LDA_results_dict
    
    
def get_trials_to_keep_and_labels(snum_by_trial, session_comparisons):
    ''' Given a list of session numbers, select and label them depending on "session_comparisons"
        snum_by_trial: list of session numbers
        session_comparison: string, can be
            - airpuff: takes all sessions, splits airpuff from non-airpuff (B and P vs T)
            - BT: compares B vs T sessions
            - BP: compares B vs P sessions
            - TP: compares T vs P sessions
    '''
    stype_by_trial = np.array([pparam.SESSION_TYPE_LABELS.index(pparam.SESSION_LABEL_BY_SNUM[snum]) for snum in snum_by_trial])
    if session_comparisons == 'airpuff':
        #AP vs no AP
        trials_to_keep = np.ones(len(stype_by_trial), dtype=bool)
        label_by_trial = pparam.get_AP_labels_from_snum_by_trial(snum_by_trial)
        
    elif session_comparisons == 'BT':
        #B vs T sessions
        trials_to_keep = np.array([stype in [0,1] for stype in stype_by_trial])
        label_by_trial = stype_by_trial[trials_to_keep]


    elif session_comparisons == 'BP':
        #B vs T sessions           
        trials_to_keep = np.array([stype in [0,2] for stype in stype_by_trial])
        label_by_trial = stype_by_trial[trials_to_keep]
        label_by_trial[label_by_trial == 2] = 1 #Convert 0/2 to 0/1 labels

    elif session_comparisons == 'TP':
        #B vs T sessions            
        trials_to_keep = np.array([stype in [1,2] for stype in stype_by_trial])
        label_by_trial = stype_by_trial[trials_to_keep]
        label_by_trial[label_by_trial == 1] = 0
        label_by_trial[label_by_trial == 2] = 1 #Convert 1/2 to 0/1 labels
        
    else:
        print('WARNING: WRONG "session_comparisons" label')
        
    return trials_to_keep, label_by_trial
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plotting functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
def plot_LDA_projection(trial_factors_LDA_projection, label_by_trial, snum_by_trial=None, plot_legend=True, ax=None):
    ''' Plot LDA projection. Assumes it's either 1d or 2d
        LDA_projection is the output of scipy's LDA.transform, has shape "num_trials" X "LDA dimension"
        label_by_trial has shape "num_trials"
        snum_by_trial has shape "num_trials", indicates the session number. Used to mark session separation (optional)
    '''
    
    # ### HACKISH WAY OF REJECTING OUTLIERS
    # # print(trial_factors_LDA_projection.shape)
    # m = 7.
    # data = trial_factors_LDA_projection.ravel()
    # d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d/mdev if mdev else np.zeros(len(d))
    # # print(np.sum(np.invert(s<m)))    
    # trial_factors_LDA_projection = data[s<m]
    # trial_factors_LDA_projection = trial_factors_LDA_projection[:, np.newaxis]
    # label_by_trial = label_by_trial[s<m]
    # snum_by_trial = snum_by_trial[s<m]
    # ### HACKISH WAY OF REJECTING OUTLIERS

    fs = 15
    
    
    if ax is None:
        fig_num = plt.gcf().number + 1
        plt.figure(fig_num, figsize=(4,3))
        ax = plt.gca()
    
    if trial_factors_LDA_projection.ndim == 1:
        LDA_components = 1
    else:
        LDA_components = trial_factors_LDA_projection.shape[1]
    unique_labels = np.unique(label_by_trial)
    

    label_colors = ['royalblue', 'indianred', 'forestgreen']
    if LDA_components == 1:
        
        label_names = ['Label 1', 'Label 2']
        
        if snum_by_trial is not None: #If Airpuff sessions are included, assume the label tracks "AP" vs "no AP"
            stype_by_trial = [pparam.SESSION_LABEL_BY_SNUM[snum] for snum in snum_by_trial]
            if pparam.SESSION_TYPE_LABELS[1] in stype_by_trial:
                label_names = ['No AP', 'AP']

            

        ldat = np.arange(len(label_by_trial))
        for label_idx, label in enumerate(unique_labels):
            label_idxs = np.where(label_by_trial == label)[0]
            t = ldat[label_idxs]                
            lda1 = trial_factors_LDA_projection[label_idxs]
            ax.scatter(t, lda1, s=100, color=label_colors[label_idx], label=label_names[label])
            ax.set_xlabel('Trial number' , fontsize=fs)
            ax.set_ylabel('LDA projection (AU)', fontsize=fs)
            if ax.get_ylim()[1] > 15:
                ax.set_ylim([ax.get_ylim()[0], 7.5])
            
    elif LDA_components == 2:
        label_names = ['B', 'T', 'P']

        ldax, lday = trial_factors_LDA_projection[:,0], trial_factors_LDA_projection[:, 1]
        for label_idx, label in enumerate(unique_labels):
            label_idxs = np.where(label_by_trial == label)[0]
            x = ldax[label_idxs]; y = lday[label_idxs]

            ax.scatter(x, y, s=100, color=label_colors[label_idx], label=label_names[label])
            ax.set_xlabel('LDA 1' , fontsize=fs)
            ax.set_ylabel('LDA 2', fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)  
    ax.tick_params(axis='x', labelsize=fs)

    if plot_legend:
        # ax.legend(fontsize=15, ncol=2, loc='lower center') 
        ax.legend(fontsize=fs, loc='lower right', frameon=False) 

    if snum_by_trial is not None: #Plot session limitators
        add_session_delimiters_to_trial_plot(ax, snum_by_trial)
        
        
    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    fig = plt.gcf()
    fig.tight_layout()
        
    return fig, ax

def add_session_delimiters_to_trial_plot(ax, snum_by_trial, fs=15, add_stype_labels = True):
    ''' For a plot whose x axis is the trial number, add session delimiters.
        snum_by_trial: array of size "num trials", each element is the session number
        assumes sessions 3,4,5,6 have an airpuff, and are marked differently
    '''
    ymin, ymax_original = ax.get_ylim()
    ymax = ymax_original*1.2
    
    snum_unique = np.sort(np.unique(snum_by_trial))        
        
    stype_start_trial = -1
    stype_end_trial = 0
    for snum_idx, snum in enumerate(snum_unique):
        stype_idx = pparam.SESSION_TYPE_LABELS.index(pparam.SESSION_LABEL_BY_SNUM[snum])
        
        session_trial_idxs = (snum_by_trial == snum)
        session_first_trial_idx = np.argmax(session_trial_idxs)
        session_last_trial_idx = len(session_trial_idxs) - np.argmax(session_trial_idxs[::-1]) - 1
        
        if snum_idx == 0:
            stype_end_trial = session_first_trial_idx
        
        #What is the next type?
        if snum_idx != len(snum_unique)-1:
            snum_next = snum_unique[snum_idx+1]
            stype_idx_next = pparam.SESSION_TYPE_LABELS.index(pparam.SESSION_LABEL_BY_SNUM[snum_next])
        else:
            stype_idx_next = -1
        if stype_idx != stype_idx_next:    
            stype_start_trial = stype_end_trial
            stype_end_trial = session_last_trial_idx
            
            if add_stype_labels == True:
                mid_trial = stype_start_trial + (stype_end_trial - stype_start_trial)/2
                ax.text(mid_trial, ymax_original, "%s"%pparam.SESSION_TYPE_LABELS_SHORT[stype_idx], fontsize=fs, weight='bold')
            
            if stype_idx == 1:
                ax.fill_between([stype_start_trial, stype_end_trial], ymin, ymax, color='gray', alpha=0.2)

        ax.plot([session_last_trial_idx] * 100, np.linspace(ymin, ymax, num=100), '--', color='gray', alpha=0.5)
        

    ax.set_ylim([ymin, ymax])
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
def test_TCA_in_false_data():
    fig_num = 1
    num_trials = 10
    num_times = 20
    num_neurons = 6
    num_factors = 2
    std = 0.1
    
    #Generate trial factors
    trial_factors = np.ones((num_factors, num_trials))
    trial_factors[0,:] = np.linspace(0, 1, num_trials) #One increases with trials
    trial_factors[1,:] = np.cos(2 * np.pi * np.arange(num_trials)/num_trials) #Another goes down then up

    plt.figure(fig_num); fig_num += 1
    plt.plot(trial_factors[0,:])
    plt.figure(fig_num); fig_num += 1
    plt.plot(trial_factors[1,:])
    
    #Generate time factors
    time_factors = np.ones((num_factors, num_times))
    time_factors[0,:] = np.sin(np.pi * np.arange(num_times)/(num_times)); time_factors[0,int(num_times/2):] = 1
    time_factors[1,:] = np.cos(np.pi * np.arange(num_times)/(num_times)); time_factors[1,int(num_times/2):] = 0

    plt.figure(fig_num); fig_num += 1
    plt.plot(time_factors[0,:])
    plt.figure(fig_num); fig_num += 1
    plt.plot(time_factors[1,:])
    
    #Generate neuron factors
    neuron_factors = np.ones((num_factors, num_neurons))
    neuron_factors[0, :] = [0.6, 0.9, 0.7, 0.1, 0.2, 0.1]
    neuron_factors[1, :] = [0.05, 0.4, 0.2, 0.7, 0.85, 0.8]

    
    plt.figure(fig_num); fig_num += 1
    plt.plot(neuron_factors[0,:], 'o')
    plt.figure(fig_num); fig_num += 1
    plt.plot(neuron_factors[1,:], 'o')
    
    #Generate recordings
    data = np.zeros((num_neurons, num_times, num_trials))
    
    for n in range(num_neurons):
        for t in range(num_times):
            for k in range(num_trials):
                data[n,t,k] = np.sum(neuron_factors[:, n] * time_factors[:, t] * trial_factors[:, k] + np.random.normal(loc=0, scale=std, size=2))
    
    
    # TCA_ensemble = tensortools.Ensemble(fit_method=TCA_method)
    # TCA_ensemble.fit(data_by_trial, ranks=range(1, TCA_factors+1), replicates=TCA_replicates, verbose=verbose)
    KTensor = perform_TCA(data, num_factors, 10, 'ncp_hals', 10, TCA_replicate_selection=0, verbose=False)
    
    trial_factors_tca = KTensor[2].T
    print(trial_factors.shape)
    
    plt.figure(fig_num); fig_num += 1
    plt.plot(trial_factors_tca[0,:])
    plt.figure(fig_num); fig_num += 1
    plt.plot(trial_factors_tca[1,:])
    
    
    
    
def test_LDA_weight_function():
    ''' How to extract the importance of each underlying dimension for a classification problem? 
        We put all important information in one dimension
    
    '''
    fig_num = 1
    colors = ['royalblue', 'indianred', 'forestgreen']
    
    num_classes = 2
    num_features = 3
    num_samples = 250
    std = 1.5
    
    #Test 1: putting all information in one feature
    centers = np.zeros((num_classes, num_features))
    centers[0,0] = 1
    centers[1,0] = 1
    centers[0,1] = 1
    centers[1,1] = -1
    labels = np.around(np.linspace(0,num_classes-1,num_samples), decimals=0).astype(int)
    
    print(labels)
    
    data = np.zeros((num_samples, num_features))
    for sample in range(num_samples):
        label = labels[sample]
        data[sample] = centers[label] + np.random.normal(loc=0.0, scale=std, size=num_features)
        
        
        

    LDA = LinearDiscriminantAnalysis(solver="eigen", #svd, lsqr, eigen
                                      shrinkage=0,  #None, "auto", float 0-1
                                      n_components=num_classes-1, #Dimensionality reduction
                                      store_covariance=False #Only useful for svd, which doesn't automatically calculate it
                                      )
    LDA.fit(data, labels)
    
    # proj_list_rep[rep,:] = (np.squeeze(LDA.transform(LDA_test_input)))
    label_predicted = (LDA.predict(data))
    accuracy = np.sum(labels==label_predicted)/num_samples
    weights = LDA.coef_[0]
    print(label_predicted)
    
    LDA_ax = get_LDA_axis(LDA, num_features)
    print(accuracy)
    print(LDA_ax)
    print(LDA.coef_)
    print(LDA.coef_.shape)
    # prob_list_rep[rep] = (LDA.predict_proba(LDA_test_input)[0,LDA_test_label]) #WARNING: assumes label corresponds to label index
    
    # print(LDA_test_input.shape, LDA.predict_proba(LDA_test_input).shape)
    # raise TypeError
    
    # LDA_ax_rep[rep,:] = LDA_ax
                 
        
        
        
        
        
        
    plt.figure(fig_num, figsize=(4,4)); fig_num += 1
    for class_label in range(num_classes):
        class_idxs = labels == class_label
        color = colors[class_label]
        plt.plot(data[class_idxs,0], data[class_idxs,1], 'o', color=color, label=class_label)
    # plt.legend(fontsize=15)
    ax = plt.gca()
    ax.axis('off')
    ax.set_title('Accuracy: %.2f'%accuracy, fontsize=20)
    
    rotations = [0, 90]
    for axis in range(2):
        plt.figure(fig_num, figsize=(6,2)); fig_num += 1
        for class_label in range(num_classes):
            class_idxs = labels == class_label
            color = colors[class_label]
            plt.hist(data[class_idxs,axis], color=color, label=class_label, alpha=0.5)
        # plt.legend(fontsize=15)
        ax = plt.gca()
        ax.invert_xaxis()
        if axis == 0:
            ax.set_xlabel('LDA weight: %.2f'%weights[axis], fontsize=14)
            ax.yaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(bottom=False, labelbottom=False)
        elif axis == 1:
            ax.set_ylabel('LDA weight: %.2f'%weights[axis], fontsize=14)
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
    
    
    
    

def test_class_inequality():
    ''' Test for the expected accuracy and f1 scores when classes are imbalanced 
        We compare the original imbalanced labels with completely random labels    
    '''
    
    s1 = 3; s2 = 10; N = 1000
    ref = np.array([0]*s1 + [1]*s2)
    ref_shuffle = np.random.choice(ref, size=len(ref), replace=False)
    f1_avg = 0
    p = np.array([0.,0.])
    for i in range(N):
        # ran = np.random.randint(0,2,size=s1+s2)
        ran = np.random.choice(ref, size=len(ref), replace=False)
        f1_avg += pf.multiclass_f1(ref, ran)
        p[0] += np.sum(ran[:s1] == 0)/s1
        p[1] += np.sum(ran[s1:s2] == 1)/s2
    f1_avg/=N
    p/=N
    print(s1/(s1+s2), s2/(s1+s2))
    print(f1_avg)
    print(p)







    
    
if __name__ == '__main__':
    tt = time.time()
    main()
    # np.random.seed(9)
    print('Time Ellapsed: %.1f' % (time.time() - tt))
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    