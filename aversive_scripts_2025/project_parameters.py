# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 03:21:38 2023

@author: Albert

Script that handles global variables
    Includes hardcoded paths, useful to share code across computers.

HOW DATA PATHS WORK:
    There is a general project folder which includes both the data and the paths to store it.    
    recognized_project_paths - must include the path the folder where the project will be stored
    FAT_CLUSTER_PATH - must be the path to the data, stored in "BigFatCluster.mat"
    PLACE_CELL_PATH_DOMBECK - must be path to the dombeck place cell classification matrix, "place_cell_bool_dombeck.mat"
    OUTPUT_PATH - folder where data will be stored. Will be created if it doesn't exist
    RESULTS_PATH - folder where figure panels will be stored. Will be created if it doesn't exist
    
HOW TO SET PROJECT FOLDER IN A NEW COMPUTER:
    (1) Create a folder for the project and add it to "recognized_project_paths" (it's made as a list to allow it to work across multiple computers at once)
    (2) Copy the main data (BigFatCluster.mat) and make sure "FAT_CLUSTER_PATH" has the correct path to it
    (3) Copy the secondary data ("place_cell_bool_dombeck.mat") and make sure "PLACE_CELL_PATH_DOMBECK" has the correct path to it (losonczy isn't used in the final figures)
                                                                                                                                    




"""

import numpy as np

import os.path
import matplotlib.pyplot as plt

##### Paths #####
recognized_project_paths = [
    "D:\Albert\MPI_Brain\Bonn Project",
    "D:\MPI_Brain\Data\Bonn Project",
    "C:\\Users\Albert\MyFiles\MPI_Brain\Data\Bonn Project"
    ]

for path in recognized_project_paths:
    path_exists = os.path.isdir(path)
    if path_exists:
        project_path = path
        break
else:
    raise FileNotFoundError('WARNING: No recognized project data path found in this computer')
    
#Main data path
FAT_CLUSTER_PATH = project_path + "\WithAmpltd\BigFatCluster.mat"

#Complementary data
PLACE_CELL_PATH_DOMBECK = project_path + "\place_cell_bool_dombeck.mat"        
PLACE_CELL_PATH_LOSONCZY = project_path + "\place_cell_bool_losonczy.mat"

#Storage paths
OUTPUT_PATH = project_path + "\Output\\"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
FIGURES_PATH = project_path + "\Figures\\"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


##### FIXED EXPERIMENT SETUP VARIABLES #####

#Experiment variables
MAX_POS = 1500

#Animals
MOUSE_TYPE_LABELS = ["ID", "DD"]
MOUSE_TYPE_INDEXES = {MOUSE_TYPE_LABELS[0]:[0,1,2,3], MOUSE_TYPE_LABELS[1]:[4,5,6,7]}
MOUSE_TYPE_LABEL_BY_MOUSE = {mnum : MOUSE_TYPE_LABELS[mnum > 3] for mnum in range(8)}

#Sessions
# SESSION_NAMES = ['B1', 'B2', 'B3', 'T1', 'T2', 'Tn-1', 'Tn', 'P1', 'P2', 'Ex1','Ex2','B1','B2','T1','T2','P1','P2']
SESSION_NAMES = ['B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'T4', 'P1', 'P2', 'Ex1','Ex2','B1','B2','T1','T2','P1','P2']
SESSION_TIMES = [0, 24, 25, 48, 49, 72, 73, 96, 97] #Hours since B1
SESSION_TYPE_LABELS = ['Baseline', 'Training', 'Probe']
SESSION_TYPE_LABELS_SHORT = ['B', 'T', 'P']
SESSION_SNUM_BY_LABEL = {SESSION_TYPE_LABELS[0]:[0,1,2],
                         SESSION_TYPE_LABELS[1]:[3,4,5,6],
                         SESSION_TYPE_LABELS[2]:[7,8],
                         }
SESSION_LABEL_BY_SNUM = {} #[snum : stype]
for snum in range(9):
    for stype in SESSION_TYPE_LABELS:
        if snum in SESSION_SNUM_BY_LABEL[stype]:        
            SESSION_LABEL_BY_SNUM[snum] = stype
            
            
#Mouse num : list of tuples with repeatead session pairs
SESSION_REPEATS = {4: [(3,5), (4,6)],
                   5: [(3,5), (4,6)],
                   6: [(1,2)]}


##### PLOTTING VARIABLES #####

#Plotting colors
PCA_CMAP = 'twilight'

SHUFFLE_DEFAULT_COLOR = 'gray'

PREDICTION_LABELS = ['Real', 'Predicted']
PREDICTION_COLORS = {PREDICTION_LABELS[0]:'black', PREDICTION_LABELS[1]:'gray'}

CCA_LABELS = ['Self', 'Unaligned', 'Aligned', 'Aligned (shift)']
CCA_COLORS = ['steelblue', 'deepskyblue', 'teal', 'darkslateblue']
ERROR_CMAP = 'viridis'

SESSION_TYPE_COLORS = ['royalblue', 'indianred', 'forestgreen']
SESSION_TYPE_COLORS_DICT= {SESSION_TYPE_LABELS[i]:SESSION_TYPE_COLORS[i] for i in range(len(SESSION_TYPE_COLORS))}

MOUSE_TYPE_COLORS = ['darkgreen', 'mediumseagreen']
MOUSE_TYPE_COLORS_BY_MTYPE = {MOUSE_TYPE_LABELS[0]:MOUSE_TYPE_COLORS[0], MOUSE_TYPE_LABELS[1]:MOUSE_TYPE_COLORS[1]}
MOUSE_TYPE_COLORS_BY_MNUM = {mnum:MOUSE_TYPE_COLORS[MOUSE_TYPE_LABELS.index(MOUSE_TYPE_LABEL_BY_MOUSE[mnum])] for mnum in range(8)}


AP_DECODING_LABELS = ['No AP', 'AP', 'Trial Shuffle', 'Session Shuffle', 'Alignment shift']
AP_DECODING_COLORS = ['slateblue', 'darkviolet', 'gray', 'darkkhaki', 'darksalmon']

DISTANCE_LABELS = ['Within', 'Between']
DISTANCE_CMAP = 'Purples'

CELL_TYPE_COLORS = ['mediumslateblue', 'orchid']
CELL_TYPE_LABELS = ['Non-place cell', 'Place cell']

WEIGHT_TYPES_COLORS = ['lightseagreen', 'goldenrod']
WEIGHT_TYPES_LABELS = ['position', 'air puff']


##### DEFAULT PARAMETERS #####
##### Used as a reference for functions that use dictionaries as input #####

ANALYSIS_NAME_LIST = ['preprocessing', 'alignment', 'APdecoding']

#Preprocessing parameters + dimensionality reduction
preprocessing_dict_name = "PCA_analysis_dict"

preprocessing_param_dict = {
    #session params
    'mouse_list':range(8),
    'session_list':range(9),
    
    #Preprocessing parameters
    'time_bin_size':1,
    'distance_bin_size':1,
    'gaussian_size':25,
    'data_used':'amplitudes',
    'running':True,
    'eliminate_v_zeros':False,
    'num_components':'all',
    }

#CCA parameters
cca_dict_name = "CCA_analysis_dict"

cca_param_dict = {
    'CCA_dim':6,
    'return_warped_data':True,
    'return_trimmed_data':False,
    'sessions_to_align':'all'
    }

#AP decoding parameter dictionary
APdecoding_dict_name = "APdecoding_analysis_dict"
APdecoding_param_dict = {
        'limit_position':False, #If True, the regions of the belt where airpuff will be decoded are restructed
        'min_pos_to_filter':1000, #lower limit (mm, MAX_POS considered the periodic maximum)
        'max_pos_to_filter':1500, #higher limit (mm, MAX_POS considered the periodic maximum)
        
        ## TCA params ##
        'TCA_method': "ncp_hals", #"cp_als", "mcp_als", "ncp_bcd", "ncp_hals"
        'TCA_factors':12, #Dimensionality of each TCA factor
        'TCA_replicates':10, #Number of times TCA is repeated
        'TCA_convergence_attempts':10, #Number of times TCA can fail before giving up
        'TCA_on_LDA_repetitions':2, #Number of times the whole TCA-LDA analysis is repeated, as results may vary
        
        ## LDA params ##
        'LDA_imbalance_prop':.6, #If one class appears in proportion higher than this threshold, perform imbalance repetitions
        'LDA_imbalance_repetitions':5, #Number of times LDA decoding is repeated using a subset of each class with equal number of samples per class
        'LDA_trial_shuffles':0, #Number of trial shuffles
        'LDA_session_shuffles':0, #Number of sessions shuffles
        # 'sessions_to_decode':[0,1,2,3,4,5,6,7,8], #Old version, take out?
        'session_comparisons':'BT' #'airpuff', 'BT', 'BP', 'PT', indicates which sessions to compare against each other
        }

default_param_dicts_names = {ANALYSIS_NAME_LIST[0]:preprocessing_dict_name,
                       ANALYSIS_NAME_LIST[1]:cca_dict_name,
                       ANALYSIS_NAME_LIST[2]:APdecoding_dict_name}


default_param_dicts = {ANALYSIS_NAME_LIST[0]:preprocessing_param_dict,
                       ANALYSIS_NAME_LIST[1]:cca_param_dict,
                       ANALYSIS_NAME_LIST[2]:APdecoding_param_dict}



    







def get_session_list_from_mouse(mnum):
    ''' Given a mouse number, get session numbers and labels avoiding repetition '''
    
    slist = list(range(9))
    if mnum in SESSION_REPEATS:
        for repeated_pair in SESSION_REPEATS[mnum]:
            if repeated_pair[0] in slist and repeated_pair[1] in slist:
                slist.remove(repeated_pair[1])      
                
    snames = [SESSION_NAMES[snum] for snum in slist]
    return slist, snames

def get_AP_labels_from_snum_by_trial(snum_by_trial):
    #Define label
    no_ap_trials = SESSION_SNUM_BY_LABEL[SESSION_TYPE_LABELS[0]] + SESSION_SNUM_BY_LABEL[SESSION_TYPE_LABELS[2]]
    ap_trials = SESSION_SNUM_BY_LABEL[SESSION_TYPE_LABELS[1]]
    
    # no_ap_trials = [0,3,5]
    # ap_trials = [1,2,4,6,7,8]
    
    label_by_trial = []
    for snum in snum_by_trial:
        if snum in no_ap_trials:
            label_by_trial.append(0)
        elif snum in ap_trials:
            label_by_trial.append(1)
    label_by_trial = np.array(label_by_trial)
    return label_by_trial
 














