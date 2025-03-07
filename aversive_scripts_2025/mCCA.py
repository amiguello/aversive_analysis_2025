# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:05:41 2023

@author: Albert

Functions for the multiset CCA

"""
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh

import project_parameters as pparam
import processing_functions as pf



def main():
    pass



class MultisetCCA(object):
    
    def __init__(self):
        pass
        
    def fit(self, dataset_list, cca_dim=None):
        ''' dataset_list: list of datasets, each assumed to be of size (N features) X (N samples)
            feature number must be the same across datasets, but samples can be different
            If cca_dim is None, use all. Otherwise must be int.
        '''
        self.M = len(dataset_list)        
        
        features_list = [dataset_list[m].shape[0] for m in range(self.M)]
        if len(set(features_list)) > 1:
            print(features_list)
            raise ValueError('Not all datasets have the same number of features')
            
        samples_list = [dataset_list[m].shape[1] for m in range(self.M)]
        if len(set(samples_list)) > 1:
            raise ValueError('Not all datasets have the same number of samples')


        self.features = dataset_list[0].shape[0]
        
        if cca_dim == None:
            cca_dim = self.features
        self.cca_dim = cca_dim


        ##### SET UP CCA GEV PROBLEM #####        

        RR = np.zeros((self.M * self.features,)*2)
        DD = np.zeros((self.M * self.features,)*2)
        for m1 in range(self.M):
            for m2 in range(self.M):
                Rjk = np.matmul(dataset_list[m1], dataset_list[m2].T)
                RR[m1 * self.features: (m1+1) * self.features, m2 * self.features : (m2+1) * self.features] = Rjk
    
                if m1 == m2:
                    DD[m1 * self.features: (m1+1) * self.features, m1 * self.features: (m1+1) * self.features] = Rjk
    
        A = (1/(self.M-1)) * (RR - DD)
        B = DD
    
        eigvals, eigvecs = eigh(A, b=B)
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, axis=1)
        
        self.eigvals = eigvals[:self.features]
        self.inv_transf_list = [eigvecs[m*self.features : (m+1)*self.features, :self.features] for m in range(self.M)]
        self.transf_list = [np.linalg.inv(self.inv_transf_list[m]) for m in range(self.M)]
        
        return self.transf_list, self.inv_transf_list
        
    def to_canonical(self, data, m):
        ''' Transform from the specified dataset's space to canonical.
        Data must be (N features) X (N samples) '''
        return self.transf_list[m] @ data
    
    def from_canonical(self, canonical_data, m):
        ''' Transform from canonical space to the specified dataset's space 
            Canonical data must be of size (cca_dim) X (N samples)'''
        return self.inv_transf_list[m] @ canonical_data
    
    def align(self, data, m1, m2):
        ''' Transform data from dataset m1's space to dataset m2's space
            data must be (N_features) X (N_samples) '''
        return self.from_canonical(self.to_canonical(data, m1), m2)


def perform_warped_mCCA(pos_list, data_list, max_pos=1500, warping_bins = 150, warp_based_on='position', return_warped_data=True, return_trimmed_data = True, shuffle=False):
    ''' 
        Perform multiset CCA on "raw" M datasets. They are allowed to be of different sizes, so trimming, warping, etc. must be performed.
        Because of this, data warping (making each trial the same size, with each trial defined as passing the max_pos mark) is performed as well
        
        INPUT:
        
        pos_list: list of M position arrays, of size samples1, samples2, etc.
        data_list: list of M data arrays, each of size (i features) X (j samples)
            Note that neither features nor samples are assumed to be equal across datasets.
            They will be trimmed to account for that
        max_pos : maximum value that can be taken by position, we assume periodicity
        warping_bins : each trial will be split into this amount of bins
        warp_based_on : 'position' or 'time', type of warping (see warping() for details)
        return_warped_data : if True, returns the data warped to have the same number of bins per trial.
        return_trimmed_data : if True, returns the data trimmed so each session has the same number of trials. 
        shuffle: randomly shifts each session independently
        
        OUTPUT:
        pos_list_aligned : the position values of the aligned data
        data_dict_aligned : {dataset index : [list of datasets aligned to it]}
        mCCA: multiset CCa object (includes transformation matrices as attributes)
            
    '''
    
    M = len(pos_list)
    
    
    # ############## [Optional] RANDOMLY SHIFTING DATA AS CONTROL ############
    if shuffle:
        for m in range(M):
            pos = pos_list[m]; data = np.copy(data_list[m])
            min_shift = int(0.25*pparam.MAX_POS)
            max_shift = int(0.75*pparam.MAX_POS)
            random_shift = np.random.randint(min_shift, max_shift, size=1, dtype=int)[0]
            idxs = list(range(data.shape[1]))
            idxs = idxs[random_shift:] + idxs[:random_shift]
            pos_list[m] = pos[idxs]
            # data_list[m] = data[:, idxs]
    # # ############## [Optional] RANDOMLY SHIFTING DATA AS CONTROL ############

            
    
    
    #Step 1: warp every dataset so they have the same number of samples per round
    pos_list_warped = []
    data_list_warped = []
    for m in range(M):
        pos = pos_list[m]; data = np.copy(data_list[m])
        pos_warped, data_warped, _ = pf.warping(pos, data, warping_bins, max_pos=max_pos, 
                                            warp_sampling_type = 'averaging', warp_based_on = warp_based_on, return_flattened = True)
        
    
        # ############# STEP 2: mCCA ############
        
        
        
        pos_list_warped.append(pos_warped); data_list_warped.append(data_warped)
        
    #Step 2: trim dataset
    pos_list_trimmed, data_list_trimmed = trim_dataset_list(pos_list_warped, data_list_warped, max_pos=max_pos)
    cca_dim = data_list_trimmed[0].shape[0]
    
    #Step 3: mCCA!
    mCCA = MultisetCCA()
    transf_list, inv_transf_list = mCCA.fit(data_list_trimmed)

    if return_warped_data == True and return_trimmed_data == False:
        data_dict_aligned = {m1 : [mCCA.align(data_list_warped[m2][:cca_dim], m2, m1) for m2 in range(M)] for m1 in range(M)}
        data_dict_aligned['canonical'] = [mCCA.to_canonical(data_list_warped[m][:cca_dim], m) for m in range(M)]
        pos_list_aligned = pos_list_warped
        
    elif return_warped_data == True and return_trimmed_data == True:
        data_dict_aligned = {m1 : [mCCA.align(data_list_trimmed[m2], m2, m1) for m2 in range(M)] for m1 in range(M)}
        data_dict_aligned['canonical'] = [mCCA.to_canonical(data_list_trimmed[m], m) for m in range(M)]

        pos_list_aligned = pos_list_trimmed
        
    elif return_warped_data == False and return_trimmed_data == False:
        # print(data_list[0].shape, data_list_trimmed[0].shape)
        data_dict_aligned = {m1 : [mCCA.align(data_list[m2][:cca_dim], m2, m1) for m2 in range(M)] for m1 in range(M)}
        data_dict_aligned['canonical'] = [mCCA.to_canonical(data_list[m][:cca_dim], m) for m in range(M)]
        pos_list_aligned = pos_list
        
    elif return_warped_data == False and return_trimmed_data == True:
        pos_list_trimmed, data_list_trimmed = trim_dataset_list(pos_list, data_list, max_pos=max_pos) 
        data_dict_aligned = {m1 : [mCCA.align(data_list_trimmed[m2][:cca_dim], m2, m1) for m2 in range(M)] for m1 in range(M)}
        data_dict_aligned['canonical'] = [mCCA.to_canonical(data_list_trimmed[m][:cca_dim], m) for m in range(M)]        
        pos_list_aligned = pos_list_trimmed        
        

    return pos_list_aligned, data_dict_aligned, mCCA

def set_dimension_of_pca_list(pca_list, pca_dim, variance_explained_list=None):
    ''' Given a list of PCA arrays, reduce all to the SAME specified pca_dim.
            e.g. if specified dimension if 10, and one pca has only 8 components, then ALL will be reduced to 8
            
        pca_list: list of PCA arrays, each of dimension "num features" X "num timepoints"
        pca_dim: specifies dimensions to keep
            -if None, take all
            -if int, take that many
            -if 'X%', where "X" is a scalar, take dimensions that explain at least X% of the variance for each PCA (needs variance explained list)
            -if 'X', where "X" is a float between 0 and 1, interpret it as taking X*100% of the variance (needs variance explained list)
        variance_explained_list: list of variance explained (output from scikit learn's PCA)
    
    '''


    num_components_to_set = np.inf
    for pca_idx, pca in enumerate(pca_list):
        
        current_num_components = pca.shape[0]
        
        if pca_dim is None:
            num_components = pca_dim
            
        elif type(pca_dim) in [int, float, np.int32]:
            num_components = int(pca_dim)
            
        elif type(pca_dim) in [str]:
            if variance_explained_list is None:
                raise TypeError('Please specify variance explained if "pca_dim" is set to a proportion')
            variance_explained = variance_explained_list[pca_idx]
            if '%' in pca_dim: #Dimensions to explain 'X%' of the variance
                variance_to_explain = float(pca_dim[:pca_dim.find('%')])
                num_components = pf.dimensions_to_explain_variance(variance_explained, variance_to_explain/100)
            else: #num_components is also variance to explain, but as a float between 0 and 1
                variance_to_explain = float(pca_dim)        
                num_components = pf.dimensions_to_explain_variance(variance_explained, variance_to_explain)
        num_components_to_set = np.min([num_components_to_set, num_components, current_num_components])
        
    num_components_to_set = int(num_components_to_set)
    
    # #### DELETE THIS TOTAL HACK ####
    
    # num_components_to_set = np.minimum(14, num_components_to_set)
    # #### DELETE THIS TOTAL HACK ####

    
    
    pca_list_reduced = []        
    for pca_idx, pca in enumerate(pca_list):
        pca_list_reduced.append(pca[:num_components_to_set])
        
    return pca_list_reduced            
            
def trim_dataset_list(pos_list, data_list, max_pos=1500):
    ''' Trim datasets to have same number of rounds and dimensions '''
    M = len(data_list)
    
    get_endtimes_from_list = lambda pos_list : [pf.get_round_endtimes_pairs(pos, max_pos=max_pos, full_rounds_only=True, pos_thresh=25) for pos in pos_list]
    round_endtimes_list = get_endtimes_from_list(pos_list)
    min_rounds = np.min([len(round_endtimes_list[i]) for i in range(M)])        
    
    pos_list_trimmed = []
    data_list_trimmed = []
    
    for m in range(M):
        startend_tuples = round_endtimes_list[m]
        start_first_round = startend_tuples[0][0]
        end_min_round = startend_tuples[min_rounds-1][1]
        
        pos_trimmed = pos_list[m][start_first_round:end_min_round]
        pos_list_trimmed.append(pos_trimmed)
        
        data_trimmed = data_list[m][:, start_first_round:end_min_round]
        data_list_trimmed.append(data_trimmed)
    
    return pos_list_trimmed, data_list_trimmed


def normalize_pca_dict_aligned(pca_dict_aligned, mCCA):
    ''' Used on the output of perform_warped_mCCA. Adds the canonical space.
        Normalizes all PCAs individually
    '''
        
    pca_dict_aligned_normalized = {k:[pf.normalize_data(pca, axis=1)[0] for pca in pca_list] for k,pca_list in pca_dict_aligned.items()}
        
    
    return pca_dict_aligned_normalized


def get_cross_prediction_errors(pos_list, data_list, pos_list_aligned, data_dict_aligned, max_pos=1500,
                                n_splits = 5, error_type='sse', predictor_name='Wiener'):
    ''' Given M sets of data, return the prediction errors when predicting on unaligned vs aligned data
    
        INPUTS:
        
        pos_list: list of M position arrays, of size samples1, samples2, etc.
        data_list: list of M data arrays, each of size (N features) X (j samples). Used as the reference for the self-prediction, may be different than the aligned data with itself
        pos_list_aligned list of M position arrays, of size samples1, samples2, etc. for the aligned datasets (can be the same as pos_list)
        data_dict_aligned: element i is a list of the datasets from data_list aligned with dataset i.
                            Each dataset must have N features, but can have different samples from the unaligned data
        For the other parameters, see predict_distance_CV
        
        OUTPUTS:
        unaligned_error_array: element (ij) is the error of predicting position in dataset j using predictor from i
    '''
    M = len(pos_list)
    # data_list = [data_dict_aligned[m][m] for m in range(M)]; pos_list = pos_list_aligned #Uncomment to have the save 
    
    
    #A: self prediction
    # pca_list_reduced = [pca[:cca_dim] for pca in pca_list]
    self_error_list = []
    self_predictor_list = []

    for m in range(M):
        data, position = data_list[m], pos_list[m]
        #CV error
        # pos_pred, error_dict, _ = pf.predict_distance_CV(data, position, n_splits=n_splits, dmax=max_pos, predictor_name=predictor_name)
        # self_error_list.append(error_dict[error_type])
        
        pos_pred, error, _ = pf.predict_position_CV(data, position, n_splits=n_splits, pmax=max_pos, predictor_name=predictor_name)
        self_error_list.append(error)

        
        #Get predictor using the full dataset
        _, _, predictor = pf.predict_position_CV(data, position, n_splits=0, pmax=max_pos, predictor_name=predictor_name)
        self_predictor_list.append(predictor)
        
    unaligned_error_array = np.zeros((M,M))
    #B: unaligned cross-prediction
    for m1 in range(M):
        predictor = self_predictor_list[m1]
        for m2 in range(M):
            if m2 == m1:
                error = self_error_list[m1]
            else:
                data, position = data_list[m2], pos_list[m2]
                pos_pred, error = pf.predict_position_from_predictor_object(data, position, predictor, periodic=True, pmax=max_pos)

            unaligned_error_array[m1, m2] = error
        
        
    aligned_error_array = np.zeros((M,M))
    #C: aligned cross-prediction
    for m1 in range(M):
        predictor = self_predictor_list[m1]
        for m2 in range(M):
            if m2 == m1:
                error = self_error_list[m1]
            else:
                data, position = data_dict_aligned[m1][m2], pos_list_aligned[m2]
                pos_pred, error = pf.predict_position_from_predictor_object(data, position, predictor, periodic=True, pmax=max_pos)

            aligned_error_array[m1, m2] = error
            
    return unaligned_error_array, aligned_error_array

def return_best_mCCA_space(pos_list, pca_dict_aligned, max_pos=1500, return_error_list = False, verbose=True):
    ''' 
        Use the prediction error to find the best data alignment, and normalize each session separately if indicated
        pos_list: list of position arrays for each space
        pca_dict_aligned_with_canonical: output from "normalize_pca_dict_aligned" function. Assumes it has the "canonical" space as well
        set_align_to: if not None, all sessions will be aligned to this one (rather than choosing the one with most SI)
        return_aligned_dict: if True, the dictionary with all the aligned PCAs (after normalization) and the chosen space is returned
    '''
    
    cv_folds = 5
    predictor_name = 'Wiener'
    error_type = 'sse'
    
    M = len(pos_list)
    
    ################## THIS CHECKS THE CANONICAL SPACE AS WELL #######################
    # There are snum+1 ways of aligning the data (in canonical space + to the spaces of each session)
    # We choose the one that minimizes prediction error
    
    # #Create lists and normalize if indicated
    sessions_to_test_alignment = range(M+1)
        
    #Get cross prediction errors
    pca_list = [pca_dict_aligned[m][m] for m in range(M)]
    unaligned_error_array, aligned_error_array = get_cross_prediction_errors(pos_list, pca_list, pos_list, pca_dict_aligned, max_pos, cv_folds, error_type, predictor_name)
    error_list = list(np.mean(aligned_error_array, axis=1))
    
    #Add the canonical space
    canonical_ref = 0
    pca_to_test = pca_dict_aligned['canonical'][canonical_ref]; pos = pos_list[canonical_ref]
    _, _, predictor = pf.predict_position_CV(pca_to_test, pos, n_splits=0, pmax=max_pos, predictor_name=predictor_name)

    error_list_to_avg = []
    for m, pca_to_test in enumerate(pca_dict_aligned['canonical']):
        if m == canonical_ref:
            continue
        pos = pos_list[m]
        position_pred, error = pf.predict_position_from_predictor_object(pca_to_test, pos, predictor, periodic=True, pmax=max_pos)

        
        error_list_to_avg.append(error)
    error_list.append(np.mean(error_list_to_avg))        

    #Select best space to align
    # error_list[1] = -1
    best_alignment = np.argmin(error_list)
    if best_alignment == M:
        best_alignment = 'canonical'
        
    if verbose==True:
        print('Optimized alignment, error: %.2f' %np.around(error_list[best_alignment], decimals=2), np.around(error_list, decimals=2), best_alignment)    

    # ################## THIS CHECKS THE CANONICAL SPACE AS WELL #######################  

            
    if return_error_list == False:
        return best_alignment
    else:
        return best_alignment, error_list
    
if __name__ == '__main__':
    TT = time.time()
    np.random.seed(1)
    main()
    print('Time Ellapsed: %.1f' % (time.time() - TT))
    plt.show()













