
"""
Created on Thu Sep 30 19:37:30 2021

@author: Albert

Various useful functions for processing Negar's data.
- General "util" functions
- Pre-processing
- PCA analysis
- Position prediction
- Plotting

"""

#Global project variables
import project_parameters as pparam
SESSION_NAMES = pparam.SESSION_NAMES

from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2_score_fun, f1_score
from sklearn import decomposition

from scipy import interpolate
from scipy.ndimage import gaussian_filter

from Neural_Decoding.decoders import WienerFilterRegression, WienerCascadeRegression, KalmanFilterRegression, SVRegression, NaiveBayesRegression, XGBoostRegression
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#Structure Index
import matplotlib


###############################################################################
########################## GENERAL FUNCTIONS ##################################
###############################################################################

def smoothing(data, std, axis=0):
    '''
    Applies a gaussian filter of given size and std along the specified axis
    '''
    
    sigma = np.zeros(data.ndim)
    sigma[axis] = std
    return gaussian_filter(data, sigma=sigma)


def sum_array_by_chunks(array, bin_size, axis=-1):
    ''' Sums every X elements of array along specified axis, with X being chunk_size
    
        Parameters:
        ----------
        array: numpy array, any dimensions
        chunk_size: number of elements from array that will be summed
        axis: axis of array along which the sum will be performed
        pro
        Returns:
        ------------
        array_out: array of dimensions same as input array, except for "axis" which will be its original size divided by chunk_size
        
        eg: 
            array = [[0 0 1 0 0 1],
                     [1 1 0 1 1 1]]
            axis = -1 (or, equivalently, 1)
            chunk_size = 2
            
            array_out = [[0, 1, 1],
                         [2, 1, 2]]
        
    '''
    
    if 0 > axis:
        axis = axis + array.ndim
    
    shape = array.shape
    
    # Pad the axis "axis" with zeros to make it even with the chunk size
    axis_residual = shape[axis] % bin_size
    if axis_residual > 0:
        pad_width = np.zeros((array.ndim, 2))
        pad_width[axis][1] = bin_size - axis_residual
        array = np.pad(array, pad_width.astype(int), mode='constant',constant_values = 0)

    array_reshaped = array.reshape(shape[:axis] + (-1, bin_size) + shape[axis+1:])
    array_out = array_reshaped.sum(axis=axis+1)
    return array_out

def get_periodic_difference(prediction, objective, period, get_sign = False):
    ''' Computes the difference between a prediction and its objective but taking periodicity into account
        Assumes they are all 1D array of same size
        If "get_sign" is True, the sign of the difference is kept.
            Example: for period=1500, 20-50 = -30; 1250-50 = -300; 50-20 = 30, 50-1250 = 300
            Explanation: the shortest path from 50 to 1250 is to move 300 units left. The shortest path from 1250 to 50 is 300 units to the right.
    '''
    
    if type(prediction) in [int, float, np.float64]:
        prediction = [prediction]
    if type(objective) in [int, float, np.float64]:
        objective = [objective]

    prediction = np.array(prediction); objective = np.array(objective)
    
    assert prediction.ndim==1 and objective.ndim==1
    extended_objective = np.stack((objective-period, objective, objective+period))
    raw_diff = extended_objective - prediction
    extended_diff_arg= np.argmin(np.abs(raw_diff), axis=0)
    extended_diff = (raw_diff)[extended_diff_arg, range(len(prediction))]
    if get_sign == False:
        extended_diff = np.abs(extended_diff)
    return extended_diff


def dimensions_to_explain_variance(variance_explained, variance_to_explain):
    ''' 
        Returns number of dimensions needed to explain a given pca variance.
    
        variance_explained should be the output of pca_instance.explained_variance_ratio_ (from sklearn)
        variance_to_explain must be a float between 0 and 1, representing % of variance to explain
    '''
    variance_explained_cum = np.cumsum(variance_explained)
    dimensions_for_x = np.argmax(variance_explained_cum > variance_to_explain) + 1
    return dimensions_for_x



def get_round_endtimes(position, diff_thresh = 1000):
    ''' Get round endtimes. Assumes position goes from its maximum value to 0 after a round end.
        Accounts for reverse movement 
            e.g. 1495, 1499, 3, 1497, 6, 10... counts as 1 single round change
        position is 1-D array
        diff_thresh is how much the position value must change in a single step to consider it a round end.
            Recommend to take 1/2 or 2/3 of maximum position
    '''
    pos_diff = np.diff(position)
    overround = list(np.where(pos_diff < -diff_thresh)[0]+1)
    any_round_back_jumps = np.any(pos_diff > diff_thresh) #E.g. 0, 3, 5, 1448, 1449...

    if any_round_back_jumps:
        overround_back = list(np.where(pos_diff > diff_thresh)[0]+1)
        num_timepoints = len(position)
        #Round counter jumps +1 when going from 1500 to 0, and jumps -1 when going 0 to 1500
        round_counter = np.zeros(num_timepoints, dtype = int)
        for round_end in overround:
            round_counter[round_end:] += 1
        for jump_back in overround_back:
            round_counter[jump_back:] -= 1
        round_counter = list(round_counter)
        num_rounds = np.max(round_counter)
        overround = [round_counter.index(round_idx) for round_idx in range(1,num_rounds+1)]

    return overround


def get_round_endtimes_pairs(position, max_pos = 1500, diff_thresh = None, full_rounds_only = False, pos_thresh = 25):
    ''' Returns a list of pairs with the start and end points of each trial.
        Convenience function to avoid problems with whether or not to include the zero
        pos_thresh: if full_rounds_only is chosen, any round that starts or ends at a 
        position further away from the end by more than "pos_thresh" will be taken as incomplete
        '''
    if diff_thresh is None:
        diff_thresh = 0.66*max_pos
    round_ends = get_round_endtimes(position, diff_thresh= diff_thresh)
    last_time = len(position)
    round_ends_full = [0] + round_ends + [last_time]
    pairs = [(round_ends_full[i], round_ends_full[i+1]) for i in range(len(round_ends_full)-1)]
    if full_rounds_only == True:
        if max_pos < pos_thresh:
            print('WARNING: pos_thresh is larger than round size, every round will be taken as incomplete')
        if position[0] > pos_thresh: #First round starts after position 0, is incomplete
            del pairs[0]
        if position[-1] < (max_pos - pos_thresh): #Last round finishes before max position, is incomplete
            del pairs[-1]
    return pairs
    




def warping(position, data, trial_bin_num, max_pos=1500, warp_sampling_type = 'interpolation', warp_based_on = 'time', return_flattened = False):
    ''' Function that splits the data into trials, and warps each trial into an equal amount of bins
        position: array of shape "timesteps"
        data: array of shape "num features" X "timesteps"
        trial_bin_num: number of bins the trial is split on
        max_pos: maximum value of the "position" vector
        warp_sampling_type: how is the sampling performed
            interpolation - data is linearly interpolated, each bin sampled at a specific location
            averaging - data is averaged for all points that fall within each bin
        warp_based_on: which variable is used to determine start and end of bins. Must be 'time' or 'position'.
        return_flattened: if True, data is outputed with the last axis being "num bins * num trials"
        
        
        RETURNS:
        position_warped: shape "num bins" X "num trials"
        data_warped: shape "num features" X "num bins" X "num trials"
        sampling_warped: 
    '''
    tt = np.arange(len(position))    
    endtime_pairs = get_round_endtimes_pairs(position, max_pos = max_pos, full_rounds_only = True)
    # print(endtime_pairs)
    num_trials = len(endtime_pairs)            
    num_data_dim = data.shape[0]
    position_warped = np.zeros(trial_bin_num * num_trials)
    data_warped = np.zeros((num_data_dim, len(position_warped)))
    sampling_warped = []

    if warp_sampling_type == 'interpolation':
        spline_order = 1
        interpolator = partial(interpolate.InterpolatedUnivariateSpline, k=spline_order)
        # interpolator = partial(interpolate.interp1d, kind='linear')        
        position_interp = interpolator(tt, position)
        data_interp_list = [interpolator(tt, data[dim,:]) for dim in range(num_data_dim)]
    
    for round_idx, round_pair in enumerate(endtime_pairs):
    
        start_idx, end_idx = round_pair
        start_warped_idx, end_warped_idx = trial_bin_num * np.array([round_idx, round_idx+1])
        position_round = position[start_idx:end_idx]
        data_round = data[:, start_idx:end_idx]
        tt_round = np.arange(len(position_round))
        
        if warp_based_on == 'time':
            #Choose the times by splitting the trial times in equal parts
            twarp_round = np.arange(trial_bin_num) * (end_idx-start_idx)/trial_bin_num
    
        elif warp_based_on == 'position':
            #Choose the times by splitting the position in equal parts. Select first time a position is reached
            poswarp = np.arange(trial_bin_num) * max_pos / trial_bin_num
            # print(list(next(x[0] for x in enumerate(position_round) if x[1] >= pos) for pos in poswarp))
            try:
                twarp_round = [next(t_visit for (t_visit, pos_visit) in enumerate(position_round) if pos_visit >= pos_bin) for pos_bin in poswarp] #Select first time each position is reached in that round
            except StopIteration: #This happens if no position about a given bin is visited this round
                last_visited_bin = next(bin_idx for (bin_idx, bin_pos) in enumerate(poswarp) if bin_pos >= position_round[-1])
                twarp_round = [next(x[0] for x in enumerate(position_round) if x[1] >= pos) for pos in poswarp[:last_visited_bin]]
                twarp_round = twarp_round + [twarp_round[-1]] * (len(poswarp) - last_visited_bin) #Select last possible time for all non-reached bins
                # raise(StopIteration)
            else:
                pass
        if warp_sampling_type == 'interpolation':
            #Get the warped data by interpolating the trial and sampling
            twarp = start_idx + np.array(twarp_round)
            position_warped[start_warped_idx:end_warped_idx] = position_interp(twarp)
            for dim in range(num_data_dim):
                data_warped[dim, start_warped_idx:end_warped_idx] = data_interp_list[dim](twarp)
                
                
        elif warp_sampling_type == 'averaging':                     
            #Get the warped data by averaging
            if warp_based_on == 'time':
                sampling_bins = list(twarp_round) + [end_idx]
                reference_for_bin = tt_round                      
            elif warp_based_on == 'position':
                sampling_bins = list(poswarp) + [max_pos]
                reference_for_bin = position_round
                
            prev_idxs_in_bin = [0]
            for bin_start_idx in range(len(sampling_bins)-1):
                bin_start = sampling_bins[bin_start_idx]
                bin_end = sampling_bins[bin_start_idx+1]
                bin_warped_idx = int(start_warped_idx + bin_start_idx)
                idxs_in_bin = np.bitwise_and(bin_start <= reference_for_bin, reference_for_bin < bin_end)
                if not np.any(idxs_in_bin): #If bin is empty, take previous one (if no previous one exists, take 0)
                    idxs_in_bin = np.copy(prev_idxs_in_bin)
                twarp_round[bin_start_idx] = np.average(tt_round[idxs_in_bin])
                
                # position_warped[bin_warped_idx] = np.average(position_round[idxs_in_bin]) ### Uncomment this to average the position within the bin
                position_warped[bin_warped_idx] = sampling_bins[bin_start_idx] ### Uncomment this to simply have the bin start as the position for the bin
                
                data_warped[:, bin_warped_idx] = np.average(data_round[:, idxs_in_bin], axis=1)
                prev_idxs_in_bin = idxs_in_bin                      
    
                
            twarp = start_idx + np.array(twarp_round)
        sampling_warped.extend(twarp)

                                               
    if return_flattened == False:
        position_warped = unflatten_warped_data(position_warped, num_trials)    
        data_warped = unflatten_warped_data(data_warped, num_trials)
        sampling_warped = unflatten_warped_data(np.array(sampling_warped), num_trials)

    
    
    return position_warped, data_warped, sampling_warped

def unflatten_warped_data(array, num_trials):
    ''' "array" must be an array with the last axis having size "num_trial_bins" * "num trials"
        This function unflattens this last axis into "num_trial_bins" X "num trials"
        array can be of any dimension, only the last axis is assumed to be the warped time one
    '''
    if array.shape[-1] % num_trials == 0:
        trial_bin_num = int(array.shape[-1] / num_trials)
    else:
        print('Incorrect num trials, array is not a multple of num_trials')
        raise ValueError('Incorrect num trials, array is not a multple of num_trials')
    new_shape = tuple([dim for dim in array.shape[:-1]] + [trial_bin_num, num_trials])

    return array.reshape(new_shape, order='F')
    
def flatten_warped_data(array):
    ''' Flattens the last two dimensions of an array of the form ... X "num_trial_bins" X "num_trials"
        array can be of any dimensions (larger than 1), only the last two axis are assumed to be number of bins and trials
    '''
    new_shape = tuple([dim for dim in array.shape[:-2]] + [array.shape[-2] * array.shape[-1]])
    return array.reshape((new_shape), order='F')

def get_idxs_in_periodic_interval(values, left_value, right_value, period):
    ''' values: array with a single dimension of size S
        left_value: start of interval (included)
        right_value: end of interval (excluded included)
        period: max value of "values" before they go to zero
    '''
    values = np.copy(values)
    if left_value < right_value:
        idxs = np.bitwise_and(values >= left_value , values < right_value)
    else:
        values[values < left_value] += period
        idxs = np.bitwise_and(values >= left_value , values < (period+right_value))
    return idxs


def get_pval_greater_or_lesser(sample1, sample2):
    
        tstat, pvalg = scipy.stats.ttest_ind(sample1, sample2, equal_var=True, permutations=None, alternative='greater')

        tstat, pvall = scipy.stats.ttest_ind(sample1, sample2, equal_var=True, permutations=None, alternative='less')
        
        if pvalg < pvall:
            pval = pvalg
        else:
            pval = -pvall      
            
        return pval  




###############################################################################
########################## PRE-PROCESSING FUNCTIONS ###########################
###############################################################################



def read_CAIM(data_cluster, mouse_num, session_num, trim_data_selection=None):
    '''
    Returns relevant variables from the data cluster
    mouse_num: between 0 and 7, 0-3 are V-D and 4-7 are D-D
    session_num: between 0 and 17
        - 0-2 are baseline trials (B1-B3)
        - 3-6 are airpuff trials at 50cm (T1, T2, Tn-1, Tn)
        - 7-8 are extinction trials (P1-P2)
        - 9-11 are 2nd round of baseline trials (B1-B2)
        - 12-13 are airpuff trials at 100cm (T1, T2)
        - 14 is extinction trial (P1)
        
    trim_data_selection: if not None, it should be a two-element tuple with the start and end of the timepoints to select. Must be floats between 0 and 1.
                        Example: for 5000 timepoints, (0.1,0.2) would select the data from timepint 500 to 1000        
    '''
    
    spikes_ref = data_cluster['CAIM']['S'][mouse_num, session_num] #CAIM for data, S for spikes, first axis is rat number (0-7), second is session (0-17)
    spikes = np.transpose(data_cluster[spikes_ref])
        
    amplitudes = data_cluster[data_cluster['CAIM']['SRaw'][mouse_num, session_num]]
    amplitudes = np.transpose(amplitudes)

    times = data_cluster[data_cluster['CAIM']['behave'][mouse_num,session_num]]['tsscn'][0]
    
    distance = data_cluster[data_cluster['CAIM']['behave'][mouse_num, session_num]]['distance'][0]

    airpuff_ref = data_cluster['CAIM']['AP'][mouse_num, session_num]
    airpuff_bool = data_cluster[airpuff_ref][0]
    
    running_bool = data_cluster[data_cluster['CAIM']['behave'][mouse_num, session_num]]['running'][0]
    
    skaggs_info = data_cluster[data_cluster['CAIM']['cclust'][mouse_num, session_num]][15,:]
    # print(skaggs_info.shape)
    # print(skaggs_info)
    # raise TypeError
    
    num_neurons, num_timepoints = spikes.shape
    

    ######## Mouse 2, session B1 (indexes 1,1) is messed up after t=10030, we deal with it separately
    if (int(mouse_num), int(session_num)) == (1,1):
        thr = 10030
        spikes = spikes[:,:thr]
        
        amplitudes = amplitudes[:, :thr]
        amplitudes = amplitudes[:, :thr]
        times = times[:thr]        
        distance = distance[:thr]
        airpuff_bool = airpuff_bool[:thr]
        running_bool = running_bool[:thr]
        num_neurons, num_timepoints = spikes.shape
        
    #Trim data in a specified way, if specified
    if trim_data_selection is not None:
        start_prop, end_prop = trim_data_selection
        start = int(start_prop * num_timepoints)
        end = int(end_prop * num_timepoints)
        spikes = spikes[:, start:end]
        amplitudes = amplitudes[:, start:end]
        times = times[start:end]
        distance = distance[start:end]
        airpuff_bool = airpuff_bool[start:end]
        running_bool = running_bool[start:end]
        num_timepoints = len(times)
        

    
    
    data_dict = {'spikes':spikes, 'amplitudes': amplitudes, 'times':times, 'distance':distance, 'AP':airpuff_bool, 
                 'running':running_bool, 'num_neurons':num_neurons, 'num_timepoints':num_timepoints,
                 'mouse_num':mouse_num, 'session_num':session_num, 'session_name':SESSION_NAMES[session_num],
                 'skaggs':skaggs_info}
    
    # data_dict['running'] = running_bool
        
    return data_dict



 
def read_and_preprocess_data(data_cluster, mouse_num, session_num, gaussian_size, time_bin_size, distance_bin_size, trim_data_selection=None, 
                             only_running=True, eliminate_v_zeros=False, distance_limits=None, pos_max=1500, **kwargs):
    ''' Apply read_CAIM function and a bunch of preprocessing, convenience function
        data_cluster: the CAIM .mat file, opened with h5py
        mouse_num: int, should be 0-7
        session_num: int, should be 0-17
        gaussian_size: size (in bin elements) of the gaussian filter
        time_bin_size: size (in bin elements) that are averaged together for analysis
        distance_bin_size: size (* in mm *) of the distance bin. 
            e.g. if "10", all elements 5-15 will become 10, 15-25 to 20, etc.
        trim_data_selection: if not None, it should be a two-element tuple with the start and end of the timepoints to select
        **kwargs are the keyword arguments for the "read_CAIM" function
        
        
        
        '''

    data_dict = read_CAIM(data_cluster, mouse_num, session_num, trim_data_selection=trim_data_selection, **kwargs)
    
    #Spikes
    spikes = data_dict['spikes']
    output_data = [['spikes', spikes]]
    
    if 'amplitudes' in data_dict:
        amplitudes = data_dict['amplitudes']
        output_data.append(['amplitudes', amplitudes])
        
    #Spike pre-processing (smoothing, binning)
    for idx, [data_name, data] in enumerate(output_data):
        data = smoothing(data.astype(float), gaussian_size, axis=1)
        data = sum_array_by_chunks(data, time_bin_size, 1)/time_bin_size # num neurons X num time bins
        # print('Avg spiking rates')
        # print(np.average(data, axis=1))
        # print("")
        # data, data_mean, data_std = normalize_data(data, axis=1); data = data*1 ##!"·"!·## Uncomment for the "old" normalizing behavior ##!"·"!·##
        output_data[idx][1] = data
    
    #Times
    times = data_dict['times']
    dt = np.average(times[1:] - times[:-1]) #Time interval in miliseconds
    
    #Position
    distance = data_dict['distance']
    distance = distance % 1500 #Eliminates negative distances, which I've seen at least in animal 5 session 2 for some reason
    distance = sum_array_by_chunks(distance, time_bin_size, 0)/time_bin_size
    distance = (distance // distance_bin_size) * distance_bin_size
        
    if 'running' in data_dict:
        running_bool = sum_array_by_chunks(data_dict['running'].astype(float), time_bin_size, 0)/time_bin_size
        running_bool = np.around(running_bool, decimals=0).astype(bool)
        data_dict['running'] = running_bool
        
    if only_running == True:
        for idx, [_, data] in enumerate(output_data):
            data_cut = data[:, running_bool]
            output_data[idx][1] = data_cut
        distance = distance[running_bool]
        times = times[running_bool]
        
    if eliminate_v_zeros == True:
        distance_original = np.copy(distance)
        for idx, [_, data] in enumerate(output_data):
            distance, data_cut, _ = compute_velocity_and_eliminate_zeros(distance_original, data, pos_max=pos_max)
            output_data[idx][1] = data_cut
            
        times = np.arange(len(distance)) * dt

    #After all other pre-processing is done, normalize
    for data_name, data in output_data:
        data_normalized, data_mean, data_std = normalize_data(data, axis=1)
        data_dict[data_name + '_binned_normalized'] = data_normalized*1
        data_dict[data_name + '_mean'] = data_mean
        data_dict[data_name + '_std'] = data_std
        # data_dict[data_name + '_binned_normalized'] = data ##!"·"!·## Uncomment for the "old" normalizing behavior ##!"·"!·##
        
    #Position-related values    
    dist_diff = np.diff(distance)
    overround = [0] + list(np.where(dist_diff < -1000)[0]+1)
    num_trials = len(overround)-1
    distance_by_trial = [distance[overround[i] : overround[i+1]] for i in range(num_trials)]           
        
    data_dict['dt'] = dt
    data_dict['distance'] = distance    
    data_dict['overround'] = overround
    data_dict['num_trials'] = num_trials
    data_dict['distance_by_trial'] = distance_by_trial
    
    return data_dict

def compute_velocity_and_eliminate_zeros(position, data, pos_max = 1500):
    ''' Given a 1D array "position" of size "timepoints", return the velocity at each point. "position" is assumed periodic in the range [0, pos_max]
        Additionally eliminate all the points where it is zero, by collapsing the position elements (e.g. 1,3,5,5,5,6 becomes 1,3,5,5,6)
        "Data" is a related 2D matrix of size "features X timepoints", the collapsed points are averaged accordingly so no information is ignored
    '''
    
    v = get_periodic_difference(position[2:], position[:-2], pos_max)/2 #Centered difference
    v0 = get_periodic_difference([position[1]], [position[0]], pos_max) #Forward difference
    vend = get_periodic_difference([position[-1]], [position[-2]], pos_max) #Backward difference
    v = list(v0) + list(v) + list(vend)
    # v = compute_velocity(position, pos_max=pos_max)
    zero_v_bool = np.array(v) < 1e-10
    
    if np.any(zero_v_bool): #for ever-increasing velocity this can only happen when consecutive positions remain the same (3 minimum at the center, 2 min at the tails of the array)              
    
        diff_for_zeros = np.diff((zero_v_bool==0).astype(int)) #Will be 1 before 0, -1 at the point where a 0 stops
        zero_vel_interval_starts = np.where(diff_for_zeros==-1)[0]
        zero_vel_interval_ends = np.where(diff_for_zeros==1)[0]+1
        
        starts_num = len(zero_vel_interval_starts); ends_num = len(zero_vel_interval_ends)
        #Is the first interval at the start? Calculate the mean neuronal activity of this interval, assign it to the last element of the interval (which has v != 0)
        if ends_num > 0 and ((starts_num == 0 and ends_num == 1) or zero_vel_interval_starts[0] > zero_vel_interval_ends[0]):
            end = zero_vel_interval_ends[0]
            data[:, end] = np.mean(data[:, :end+1], axis=1)
            zero_vel_interval_ends = zero_vel_interval_ends[1:]
            ends_num = ends_num-1
        
        
        
        #Is the last interval at the end? Calculate the mean neuronal activity of this interval, assign it to the last element of the interval (which has v != 0)
        if starts_num > 0 and ((starts_num == 1 and ends_num == 0) or zero_vel_interval_starts[-1] > zero_vel_interval_ends[-1]):
            start = zero_vel_interval_starts[-1]
            data[:, start] = np.mean(data[:, start:], axis=1)
            zero_vel_interval_starts = zero_vel_interval_starts[:-1]
            starts_num = starts_num-1
    
        #Are there any zero vel intervals in the middle? Average neuronal activities within the zero vel intervals, substitute value at its edges            
        if starts_num > 0:
            for start,end in zip(zero_vel_interval_starts, zero_vel_interval_ends):
                mid = int(np.ceil(start + (end-start)/2))
                data[:, start] = np.mean(data[:, start:mid], axis=1)
                data[:, end] = np.mean(data[:, mid:end+1], axis=1)
            
        non_zero_v_bool = np.invert(zero_v_bool)
        position = position[non_zero_v_bool]
        data = data[:, non_zero_v_bool]
        v = list(np.array(v)[non_zero_v_bool])
        
    return position, data, v

def normalize_data(data, axis=1):
    ''' Elements along given axis have their average subtracted and are divided by their standard deviation '''
    data_mean = np.mean(data, axis=axis, keepdims=True)
    data_std = np.std(data, axis=axis, keepdims=True)
    data_std[data_std==0] = 1 #For cases where there's no variation in the data to avoid dividing by zero
    return (data-data_mean)/data_std, data_mean, data_std

def get_data_from_datadict(data_dict, data_used, distance_limits=None, pos_max=1500):
    ''' Convenience function that returns the relevant data from the data_dict
        data_dict: output of "read_CAIM" function
        data_used: spikes, amplitudes, or scaled spikes
        running: if True, only running datapoints are used
        eliminate_zero_v: if True, all the points were v=0 (using centered differences) are averaged out, so that distance increases almost everywhere
            NOTE: this is different from "running". "running" is a boolean from the original dataset which sometimes includes points with no detected distance increase, v=0 is applied on top of this
        
        distance_limits: if not None, must be a two element array with the minimum and maximum distance to analyze
        
    
    '''
    
    spikes = data_dict['spikes']; spikes_binned_normalized = data_dict['spikes_binned_normalized']
    amplitudes = data_dict['amplitudes']; amplitudes_binned_normalized = data_dict['amplitudes_binned_normalized']
    distance = data_dict['distance']
    times = data_dict['times']
    num_neurons = spikes.shape[0]        
    
    if data_used == 'spikes':
        output_data = np.copy(spikes_binned_normalized)
    elif data_used == 'amplitudes':
        output_data = np.copy(amplitudes_binned_normalized)
    elif data_used == 'scaled spikes':
        output_data = np.copy(amplitudes_binned_normalized*spikes_binned_normalized)

    if distance_limits is not None:
        dmin, dmax = distance_limits
        dlim_bool = np.bitwise_and(dmin < distance, distance < dmax)
        output_data = output_data[:, dlim_bool]
        distance = distance[dlim_bool]
        times = times[dlim_bool]

    return output_data, distance, times


def load_place_cell_boolean(mnum, snum, criteria='dombeck'):
    from scipy.io import loadmat
    ''' Loads the place cell arrays for a particular animal and session 
        WARNING: the path is hardcoded!
    ''' 
    
    if criteria == 'dombeck':     
        place_cell_path = pparam.PLACE_CELL_PATH_DOMBECK   
    elif criteria == 'losonczy':
        place_cell_path = pparam.PLACE_CELL_PATH_LOSONCZY   

    place_cell_bool_dataset = loadmat(place_cell_path)['place_cell_bool']
    place_cell_bool = np.squeeze(place_cell_bool_dataset[snum, mnum])
    
    return place_cell_bool


def compute_average_data_by_position(data, position, position_bin_size=None, max_pos=1500):
    ''' Given a dataset, calculate its average value for each observed position (using the given bin size)
        Input:
            data: matrix of size "pca dim" X "timepoints"
            position: array of size "timepoints"
            position_bin_size: if None the values from "position" are used to average. If "int", the position values are approximated to the nearest mcm with the bin size
            max_pos: maximum value of the position
            
        Returns:
            position_bins: array of size "num of position bins", contains the position values used to average the data
                e.g.: if bin_size = 20 and the first element is 0, the first data_average element will contain the average of the PCA of all the times the position was between 0 and 20
            data_average: array of size "pca dim" X "num of unique positions", contains the average PCA at each corresponding position bin
        
    '''
    num_dimensions, num_timepoints = data.shape
    position = position % (max_pos+1)
    
    if position_bin_size is not None:
        position = (position // position_bin_size) * position_bin_size
    position_bins = np.unique(position)
    num_bins = len(position_bins)
        
    overround = get_round_endtimes(position, diff_thresh=max_pos * 0.66)
    num_rounds = len(overround)
    
    
    data_average = np.zeros((num_dimensions, num_bins))
    data_std = np.zeros((num_dimensions, num_bins))
    for idx, d in enumerate(position_bins):
        data_filtered = data[:,position==d]
        data_average[:,idx] = np.mean(data_filtered, axis=1)
        data_std[:, idx] = np.std(data_filtered, axis=1)/np.sqrt(num_rounds)
    return position_bins, data_average, data_std

###############################################################################
############################ PCA FUNCTIONS #############################
###############################################################################


def project_spikes_PCA(pca_input_data, pca_instance = None, num_components = 'all', return_pca_instance = False):
    ''' Projects spikes to PCA space.
        pca_input_data: spikes, shape "num features (e.g. neurons)" X "num samples (e.g. timepoints)"
        pca_instance: if None, a pca instance will be created and trained. If not None, it must be a trained instace of sklearn's PCA. 
        num_components: how many pca dimensions to take.
            *if None or 'all', take all
            *if int, take that many
            *if 'X%', where "X" is a scalar, take dimensions that explain X% of the variance
            *if 'X', where "X" is a float between 0 and 1, interpret it as taking X*100% of the variance
        
            
    '''
    num_features = pca_input_data.shape[0]
    if pca_instance is None:
        pca_instance = decomposition.PCA(n_components=num_features)
        pca_instance.fit(pca_input_data.T) 
            
    if num_components is None or num_components == 'all':
        num_components = num_features
        
    elif type(num_components) in [int, float, np.int32]:
        num_components = int(num_components)
        
    elif type(num_components) in [str]:
        variance_explained = pca_instance.explained_variance_ratio_        
        if '%' in num_components: #Dimensions to explain 'X%' of the variance
            variance_to_explain = float(num_components[:num_components.find('%')])
            num_components = dimensions_to_explain_variance(variance_explained, variance_to_explain/100)
        else: #num_components is also variance to explain, but as a float between 0 and 1
            variance_to_explain = float(num_components)        
            num_components = dimensions_to_explain_variance(variance_explained, variance_to_explain)
        # variance_explained_cum = np.cumsum(variance_explained)
        # dimensions_for_x = np.argmax(variance_explained_cum > variance_to_explain) + 1

        
    else:
        raise(TypeError('Unknown data type for "num_components"'))
        
    transform_m = pca_instance.components_[:num_components, :]
    spikes_projected = transform_m @ pca_input_data
    if return_pca_instance == False:
        return spikes_projected
    else:
        return spikes_projected, pca_instance


def reshape_pca_list_by_trial(pca_list, pos_list, num_bins, session_list):
    #Reshape by trial
    pos_aligned_by_trial_list = []
    pca_aligned_by_trial_list = []
    num_trials_list = []
    snum_by_trial_list = []
    for sidx, snum in enumerate(session_list):
        
        pca = pca_list[sidx]
        pos = pos_list[sidx]
        
        ntrials = int(pca.shape[1]/num_bins)
        num_trials_list.append(ntrials)
        pca_aligned_by_trial_list.append(unflatten_warped_data(pca, ntrials))
        pos_aligned_by_trial_list.append(unflatten_warped_data(pos, ntrials))
        
        snum_by_trial_list.extend([snum]*ntrials)
        
    num_trials = sum(num_trials_list)
    pca_by_trial = np.dstack(pca_aligned_by_trial_list)
    pos_by_trial = np.hstack(pos_aligned_by_trial_list)
    snum_by_trial = np.stack(snum_by_trial_list)
    
    return pca_by_trial, pos_by_trial, snum_by_trial






###############################################################################
############################# POSITION PREDICTION FUNCTIONS ##############################
###############################################################################

def from_linear_to_circular_position(position, pmin, pmax):
    
    ''' Takes 1D position data and transforms to circular.
        pmin and pmax are the minimum and maximum position, respectively.
        Returns data in the form "samples" X 2
    '''
    
    angle_d = (position - pmin)/(pmax-pmin) * 2 * np.pi - np.pi
    sin_d = np.sin(angle_d)
    cos_d = np.cos(angle_d)
    angle_pos = np.vstack((sin_d, cos_d)).T
    
    return angle_pos

def initiate_predictor(predictor_name): #Initiates predictor object
    if predictor_name == 'Wiener':              
        predictor = WienerFilterRegression()
        
    elif predictor_name == 'Wiener Cascade':
        predictor = WienerCascadeRegression(degree=4)
        
    elif predictor_name == 'Kalman':
        predictor = KalmanFilterRegression(C=1)
        
    elif predictor_name == 'SVR':
        predictor = SVRegression(C=3, max_iter=-1) #-1 means no iteration limit
        
    elif predictor_name == 'Naive Bayes':
        predictor = NaiveBayesRegression(encoding_model = 'quadratic', res=100)
        
    elif predictor_name == 'XGBoost':
        predictor = XGBoostRegression(max_depth=3, num_round=300, eta=0.3, gpu=-1)
        
        
    else:
        raise NameError('Predictor name for position predictor is not allowed')
    return predictor

def get_prediction(predictor, X, Y): #Get function to handle Kalman filter's different function call
    ''' Convenience function that handles the Kalman filter case differently, X is data set to predict, Y is the actual labels.
        Note that Y is only ever called in the Kalman prediction but only the array's shape is used
        "predictor" must be scikit type trained object (or from "decoders.py")
        X of shape "samples X features"
        Y of shape "samples"
        '''
        
        
    #Predict
    if type(predictor) in [type(KalmanFilterRegression()), type(NaiveBayesRegression())]:
        pred = predictor.predict(X, Y)
    else:
        pred = predictor.predict(X)
    return pred

def get_sse(prediction, objective, period=None):
    ''' Computes the average squared sum of error. prediction and objective are assumed to be 1d and same size '''
    if period is None:
        diff = prediction-objective
        
    else:
        diff = get_periodic_difference(prediction, objective, period)
    
    sse = np.sum((diff)**2)/(prediction.size)
    # sse = np.trapz(np.abs(diff))
    return sse

def get_error_dict(position, position_pred, period=1500):
    sse = get_sse(position_pred, position, period=period)
    sse = np.sqrt(sse)/10 #From mm to cm
    best_delay = 0
    
    if period is not None:
        diff = get_periodic_difference(position_pred, position, period=period)/10 #The 10 is the conversion to cm!
    else:
        diff = (position_pred - position)/10
    diff_std = np.std(diff)
    diff_std = diff_std/2 #Done to match Martin Pofahl's code
    diff_avg = np.mean(np.abs(diff))
    
    r2 = r2_score_fun(position, position_pred)
    
    error_dict = {'sse':sse, 'diff':diff, 'diff_std':diff_std, 'diff_avg':diff_avg, 'r2':r2,
                  'best_delay':best_delay}
    
    return error_dict, position_pred


def predict_position_CV(data, position, n_splits=5, shuffle=False, periodic=True, pmin=0, pmax=1500,
                        predictor_name='Wiener', predictor_default=None, return_error='sse'):
    ''' 
    1D position data is converted to a 2D circle so the quantity to predict is periodic
    
    data: shape "features" X "samples"
    position: 1D array of size "samples" with real values to predict
    n_splits: number of CV folds. If 1 or less no cross-validation is performed (so no test sets)
    shuffle: if True, data is shuffled [DOESN'T WORK AS OF NOW]
    periodic: if True, data is converted                                        
    dmin, dmax: min and max position to do the periodicity
    predictor name: 'Wiener', 'Kalman', 'SVR'
    predictor_default: if None, a type of predictor specified by "predictor_name" is trained on the data. If one is given, that is used instead (we assume it has been trained)
    return_error: must be a error name. 
    '''
    
    
    #Convert to periodic 2D circle
    X = data.T
    if periodic:    
        Y = from_linear_to_circular_position(position, pmin, pmax) # num samples X num features (which are two)
    else:
        Y = np.vstack(position)

  
    #Predict position
    if predictor_default is None:
        #If no previous predictor is given, train it from scratch
        if n_splits > 1:
            kf = KFold(n_splits=n_splits, shuffle=shuffle)
            Y_pred = np.zeros(Y.shape)
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                predictor = initiate_predictor(predictor_name)
                predictor.fit(X_train, Y_train)
                Y_pred[test_index] = get_prediction(predictor, X_test, Y_test)
                
        else: #No validation set
            predictor = initiate_predictor(predictor_name)
            predictor.fit(X,Y)
            Y_pred = get_prediction(predictor, X, Y)

            
    else:
        #If a trained predictor is given, test
        predictor = predictor_default
        Y_pred = get_prediction(predictor, X, Y)


    if periodic:
        angle_d_pred = np.arctan2(Y_pred[:,0], Y_pred[:,1])
        position_pred = ((angle_d_pred+np.pi)/(2*np.pi)) * (pmax-pmin) + pmin
        period = pmax
    else:
        period = None
        position_pred = np.squeeze(Y_pred)
    
    error_dict, position_pred = get_error_dict(position, position_pred, period=period)
    error = error_dict[return_error]
    return position_pred, error, predictor



def predict_position_from_predictor_object(data, position, predictor, periodic=True, pmin=0, pmax=1500, return_error='sse'):
    
    #Convert to periodic 2D circle
    X = data.T
    if periodic:    
        Y = from_linear_to_circular_position(position, pmin, pmax) # num samples X num features (which are two)
    else:
        Y = np.vstack(position)
        
    Y_pred = get_prediction(predictor, X, Y)
        
    if periodic:
        angle_d_pred = np.arctan2(Y_pred[:,0], Y_pred[:,1])
        position_pred = ((angle_d_pred+np.pi)/(2*np.pi)) * (pmax-pmin) + pmin
        period = pmax
    else:
        period = None
        position_pred = np.squeeze(Y_pred)
    error_dict, position_pred = get_error_dict(position, position_pred, period=period)
    error = error_dict[return_error]
    return position_pred, error


def multiclass_f1(true_labels, predicted_labels):
    ''' Returns F1 measures in a multiclass problem. Each class has its own F1.
        Output: array of size "num_unique_labels"
    '''
    
    unique_labels = np.unique(true_labels)
    f1_list = np.zeros(unique_labels.shape)
    
    for idx, label in enumerate(unique_labels):
        true_bool = true_labels == label
        pred_bool = predicted_labels == label
        f1 = f1_score(true_bool, pred_bool)
        f1_list[idx] = f1 

    return f1_list

###############################################################################
############################# PLOTTING FUNCTIONS ##############################
###############################################################################

def plot_colored_line_3d(x, y, z, color_scalar, cmap_name='viridis', fig=1, ax=None, lw=2, cbar=True, 
                         xlim=None, ylim=None, zlim=None, color_norm_limits = None):
    ''' Efficiently plots a 2d colored line using LineCollection
        x: dimension N
        y: dimension N
        z: dimension N
        color_scalar: dimension N, each element is a scalar that defines its color
        cmap: string, indicates matplotlib cmap to use for the colorbar
        figure: which figure. Can be int for its number or a figure instance from matplotlib
        ax: if not None, takes precedence over "figure"
        lw: linewidth, thickness of drawing       
        color_norm: if None, the color will use the min and max of color scalar as limits.
                    If two-element list of the sort [min, max], it will use those instead
        
        Outputs the figure, axis, and colorbar objects
        
    '''
    
    cmap = plt.get_cmap(cmap_name)
    color = cmap(color_scalar)
    # print(color_scalar)
    
    # We are going to create a line for every pair of points, so we reshape our data into a
    # N x line_length x dimensions array. Line length is 2 and dimension 3 (x, y, z).
    #   e.g. element [0,:,:] is [[x0,x1],[y0,y1]]
    points = np.array([x,y,z]).T
    segments = np.stack((points[:-1], points[1:]), axis=-1).transpose(0,2,1)


    if ax is not None: #Axis is given, get figure
        plt.sca(ax)
        fig = plt.gcf()
        
    elif type(fig) is int: #Only fig num is given, create fig and axis
        fig = plt.figure(fig)
        ax = fig.add_subplot(111, projection='3d')
        
    else: #Figure is given, get axis
        # fig = plt.figure(fig)
        ax = fig.gca()
        
    if color_norm_limits is None:
        norm = plt.Normalize(np.min(color_scalar), np.max(color_scalar))
    else:
        norm = plt.Normalize(color_norm_limits[0], color_norm_limits[1])
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_scalar)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)
    
    if cbar:
        cbar = fig.colorbar(line)
        
    if xlim is None:
        ax.set_xlim([np.min(x), np.max(x)])
    else:
        ax.set_xlim(xlim)
        
    if ylim is None:
        ax.set_ylim([np.min(y), np.max(y)])
    else:
        ax.set_ylim(ylim)
        
    if zlim is None:
        ax.set_zlim([np.min(z), np.max(z)])
    else:
        ax.set_zlim(zlim)
    
    return fig, ax, cbar

def add_distance_cbar(fig, cmap, vmin = 0, vmax = 1500, fs=15, cbar_label = 'Position (mm)', cbar_kwargs={'fraction':0.04}):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', **cbar_kwargs)
    cbar.ax.set_title(cbar_label, fontsize=fs)
    cbar.ax.tick_params(axis='both', which='major', labelsize=fs-3)
    return cbar

def plot_pca_with_position(pca, position, ax=None, max_pos = 1500, cmap_name = pparam.PCA_CMAP, fs=15, scatter=True, cbar=False, cbar_label='Position (mm)',
                           alpha=1, angle=50, angle_azim=None, axis = 'off', show_axis_labels=True, axis_label=None, ms = 10, lw=3, line_effect=False):
    '''
    Plots the first three dimensions of a PCA.
    
    Parameters
    ----------
    pca : array of size "num features" X "num samples"
    position : array of size "num samples"
        Indicates the position of the animal per sample.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    max_pos : int, optional
        Maximum value of the position value. The default is 1500.
    cmap_name : string, optional
        Name of the colormap to use. The default is 'hsv'.
    fs : int, optional
        Fontsize. The default is 15.
    scatter : bool, optional
        If True, PCA is plotted using the scatter function. Otherwise lines are drawn. The default is True.
    cbar : bool, optional
        if True, a colorbar is plotted
    alpha : float, optional
        Transparency of the plot. The default is 1.
    angle : int, optional
        Main 3d angle. The default is 50.
    angle_azim : int, optional
        Azimuthal angle. The default is None.
    axis : string, optional
        'on' or 'off'. The default is 'off'.
    show_axis_labels : bool, optional
        Whether or not to label the axis. The default is True.
    axis_label : str, optional
        If None, each axis is labelled as a PCA axis. Otherwise this label is used. The default is None.
    ms : int, optional
        Marker size for the scatter plots. The default is 10.
    lw : int, optional
        Line width for the average PCA plot. The default is 3.

    Returns
    -------
    ax : matplotlib axis object. If None, a new one is created.

    '''
    
    if ax is None:
        ax = plt.subplot(1,1,1,projection = '3d')
    
    fig = plt.gcf()        
    cmap = plt.get_cmap(cmap_name)        
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
    
    x, y, z = pca[:3]
    if scatter==True:
        ax.scatter(x, y, z, color=cmap(position/max_pos), s=ms, marker='o', alpha=alpha)        
    else:
        fig, ax, _ = plot_colored_line_3d(x, y, z, position, cmap_name=cmap_name, ax=ax, lw=lw, cbar=False, 
                                          xlim=xlim, ylim=ylim, zlim=zlim, color_norm_limits = [0,max_pos])
    
    if axis_label is None:
        axis_label = 'PCA'

        
    if show_axis_labels == True:    
        ax.set_xlabel(axis_label + ' D1', fontsize=fs)
        ax.set_ylabel(axis_label + ' D2', fontsize=fs)
        ax.set_zlabel(axis_label + ' D3', fontsize=fs) 
    ax.view_init(elev=angle, azim=angle_azim)
    if cbar == True:
        add_distance_cbar(fig, cmap, vmin = 0, vmax = max_pos, fs=fs, cbar_label=cbar_label)
        
    ax.set_xlim([np.minimum(np.min(x), xlim[0]), np.maximum(np.max(x), xlim[1])])
    ax.set_ylim([np.minimum(np.min(y), ylim[0]), np.maximum(np.max(y), ylim[1])])
    ax.set_zlim([np.minimum(np.min(z), zlim[0]), np.maximum(np.max(z), zlim[1])])
    
    if axis == 'off':
        ax.set_axis_off()
        
    elif axis == 'on':
        ax.grid(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

    return ax

def plot_pca_overlapped(pca_list, pos_list, average_by_session=True, pca_plot_bins=50, ax=None):
    #Plot overlapped average PCAs by session

    if ax is None:
        fig_num = plt.gcf().number+1
        plt.figure(fig_num, figsize=(7,7))
        ax = plt.subplot(projection='3d')

    num_sessions = len(pca_list)
    for sidx in range(num_sessions):
        # cbar = int(m==(M-1))
        cbar = False
        pca = pca_list[sidx]; pos = pos_list[sidx]
        pos_bins, pca_avg, _ = compute_average_data_by_position(pca, pos, position_bin_size=pca_plot_bins)
        
        plot_pca_with_position(pca_avg, pos_bins, ax = ax, scatter=False, cbar=cbar) 
    ax.set_axis_off()
    
    if ax is None:
        fig = plt.gcf()        
        fig.tight_layout() 




def draw_significance(ax, pval, p00, p10, p11, d0, dp, orientation='left', label_padding=0, thresholds = [0.05, 0.005, 0.0005], fs=15):
    ''' 
        ax is the axis plot
        pval is the pvalue to plot
        orientation: left, right, top, bottom
        label_padding: distance from the line center to the text label (in order to be centered, we might want it smaller)
        
        
        For left, the values are as follows:
            
              d0
            |----|
         
            ------ (p00, p11)
            |
            |
            |
            |
            |
            |
            |
            |            
            ------ (p00, p10)
        
        Other orientations simply rotate this. 
        p00 is the coordinate that doesn't change along the line, p10 and p11 are the coordinates indicating start/end
        d0 is the width from that position to the actual line (can be 0)
        dp is the distance from the line to the text
        fs is fontsize
        
        
        
    '''
    
    if orientation in ['left', 'right']:
        
        if orientation == 'left':

            x0 = p00; x1 = p00-d0
            y0 = p10; y1 = p11
            xp = x0 - d0 - dp
            yp = (y1 + y0)/2 + label_padding
        
        if orientation == 'right':

            x0 = p00; x1 = p00 + d0
            y0 = p10; y1 = p11
            xp = x1 + d0
            yp = (y1 + y0)/2 + label_padding
            
        xx = [x0, x1, x1, x0]
        yy = [y0, y0, y1, y1]

    elif orientation in ['top', 'bottom']:
        
        if orientation == 'top':

            x0 = p10; x1 = p11
            y0 = p00; y1 = p00 + d0
            xp = (x1 + x0)/2 + label_padding
            yp = y1 + dp
        
        if orientation == 'bottom':

            x0 = p10; x1 = p11
            y0 = p00; y1 = p00 - d0
            xp = (x1 + x0)/2 + label_padding
            yp = y1 - dp
            
        xx = [x0, x0, x1, x1]
        yy = [y0, y1, y1, y0]         
        
    ax.plot(xx, yy, 'k', lw=2, alpha=0.7)

    pval_label = get_significance_label(pval, thresholds, asterisk=True, ns=True)
    ax.text(xp, yp, pval_label, fontsize = fs, style='italic')


def get_significance_label(pval, thresholds = [0.001, 0.0001], asterisk=False, ns=False):
    thresholds = np.sort(thresholds)[::-1]
    signum = np.sum(pval < np.array(thresholds)) #How many thresholds is pval below?
    
    if signum == 0:
        if ns == True:
            return 'ns'
        else:
            return ''
    else:
        if asterisk == False:
            return "p<" + ("%f" %thresholds[signum-1]).rstrip('0').rstrip('.')
        else:
            return "*" * signum


def add_linear_regression(x, y, ax, color='forestgreen', label=True):
    #Add a linear regression to an axis, adds a label with the r value. Returns rval and the updated axis.
    slope, intercept, rval, pval, stderr = scipy.stats.linregress(x, y)
    linexx = np.linspace(np.min(x), np.max(x))
    lineyy = slope*linexx + intercept
    if label == True:
        label = 'r=%.2f' %rval
    else:
        label = None
    ax.plot(linexx, lineyy, '--', color=color, label=label)
    return rval, pval


# if __name__ == '__main__':
#     import main_functions
#     main_functions.main()
#     tt = time.time()
#     main()
# #    np.random.seed(1)
#     print('Time Ellapsed: %.1f' % (time.time() - tt))
#     plt.show()
