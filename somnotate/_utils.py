#!/usr/bin/env python

"""
TODO
- consistent variable naming for p, min_percentile, max_percentile
"""

import numpy as np

def robust_normalize(arr, p=1., axis=-1, method='standard score'):
    """
    References:
    -----------
    https://en.wikipedia.org/wiki/Normalization_(statistics)
    """

    if method == 'min-max':
        return np.apply_along_axis(_robust_min_max_normalization, axis, arr, p=p)
    elif method == 'standard score':
        return np.apply_along_axis(_robust_standard_score_normalization, axis, arr, p=p)
    else:
        raise ValueError("Method one of: 'min-max', 'standard score'. Current value: {}".format(method))


def _robust_min_max_normalization(vec, p):
    minimum, maximum = np.percentile(vec, [p, 100-p])
    normalized = (vec - minimum) / (maximum - minimum)
    return normalized


def _robust_standard_score_normalization(vec, p):
    truncated = _truncate_signals(vec, p, 100.-p)
    robust_mean = np.mean(truncated)
    robust_std  = np.std(truncated)
    return (vec - robust_mean) / robust_std


def truncate_signals(arr, min_percentile=0.1, max_percentile=99.9, axis=0):
    return np.apply_along_axis(_truncate_signals, axis, arr,
                               min_percentile=min_percentile,
                               max_percentile=max_percentile)


def _truncate_signals(vec, min_percentile, max_percentile):
    min_value = np.percentile(vec, min_percentile)
    max_value = np.percentile(vec, max_percentile)
    vec[vec < min_value] = min_value
    vec[vec > max_value] = max_value
    return vec


def convert_state_vector_to_state_intervals(state_vector, time_resolution=1., mapping=None):
    """
    Convert a state vector, where each entry is assumed to correspond to an interval of
    constant length, into a vector of states and their corresponding contiguous time intervals.

    Arguments:
    ----------
    state_vector -- (total samples, ) ndarray with dtype int
        The state vector.

    time_resolution -- float (default 1.)
       The assumed time duration of each entry in the state vector.

    mapping -- dict int : str or None (default None)
        Optional argument; if provided, entries in the state vector are converted
        to "proper" state labels in the output.

    Returns:
    --------
    states -- list of ints (or str if mapping is not None)
        The state vector.

    intervals -- list of (float start, float stop) tuples
        The contiguous intervals corresponding to each state in the state vector.

    See also:
    ---------
    convert_state_intervals_to_state_vector
    """

    # initialise output
    states = list()
    intervals = list()

    # Loop over the entries of state vector while keeping track of the current state;
    # at each state transition, add the state that just ended and the corrsponding interval to the output.
    current_state  = state_vector[0]
    current_state_start = 0
    for ii, state in enumerate(state_vector):
        if state != current_state:
            # save current state and current interval
            states.append(current_state)
            intervals.append((current_state_start, ii*time_resolution))
            # change current state to new state, start counting new interval
            current_state = state
            current_state_start = ii*time_resolution

    # saving of last state is not triggered by state change!
    states.append(current_state)
    intervals.append((current_state_start, (ii+1)*time_resolution))

    if mapping:
        states = [mapping[state] for state in states]

    return states, intervals


def convert_state_intervals_to_state_vector(states, intervals, mapping,
                                            time_resolution = 1.,
                                            length          = None,
):
    """
    Construct a state vector given a list of states and a corresponding list of intervals.

    Arguments:
    ----------
    states -- list of ints (or str if mapping is not None)
        The state vector.

    intervals -- list of (float start, float stop) tuples
        The contiguous intervals corresponding to each state in the state vector.

    mapping -- dict str : int
        The mapping from states in the state vector to integers.

    time_resolution -- float (default 1.)
       The assumed time duration of each entry in the state vector.

    Returns:
    --------
    state_vector -- (total samples, ) ndarray with dtype int
        The state vector.

    See also:
    ---------
    convert_state_vector_to_state_intervals
    """

    if time_resolution != 1:
        intervals = [(start/time_resolution, stop/time_resolution) for start, stop in intervals]

    if np.any([(isinstance(start, float), isinstance(stop, float)) for start, stop in intervals]):
        import warnings
        warnings.warn("Interval values are converted from floats to integers.")
        # # round up last interval such that the state vector is guaranteed to include the last time point
        # last_start, last_stop = intervals[-1]
        # intervals[-1] = (last_start, np.ceil(last_stop))
        intervals = [(int(np.round(start)), int(np.round(stop))) for start, stop in intervals]

    if not length:
        length = np.max(intervals)

    state_vector = np.zeros((length), dtype=int)
    for state, (start, stop) in zip(states, intervals):
        state_vector[start:stop] = mapping[state]

    return state_vector


def smooth(signals, window_length, window='hanning', axis=-1):
    """
    Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Arguments:
    -----------
    signals: ndarray of floats
        The input signal.

    window_length: Odd integer
        The dimension of the smoothing window.

    window: string
        The type of smoothing window: one of
            - 'flat',
            - 'hanning',
            - 'hamming',
            - 'bartlett',
            - 'blackman'.
        A flat window will produce a moving average smoothing.

    Returns:
    -----------
    output: ndarray of floats
        The smoothed signal.

    Example:
    -----------
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    See also:
    -----------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    Note:
    -----
    Adapted from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    return np.apply_along_axis(_smooth, axis, signals,
                               window_length=window_length, window=window)


def _smooth(x, window_length, window='flat'):

    # window_length = np.max([len(x)/10., 1]).astype(int)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_length:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_length<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is none of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # reflect data for symmetric boundary condition
    s = np.r_[x[window_length-1:0:-1],x,x[-2:-window_length-1:-1]]

    if window == 'flat': # moving average
        w=np.ones(window_length,'d')
    else:
        w=eval('np.'+window+'(window_length)')

    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_length-1:-window_length+1]


def _get_intervals(boolean_mask):

    # base case
    d = np.diff(boolean_mask.astype(np.int8))
    starts = np.where(d == 1)[0] +1
    stops = np.where(d == -1)[0] +1

    # edge cases:

    # all values may be True:
    if np.all(boolean_mask):
        return np.array([[0, len(boolean_mask)]])

    # all values may be False:
    if np.all(~boolean_mask):
        return np.zeros((0, 2))

    # bolean mas may contain False values only at either end
    if len(starts) == 0:
        starts = np.concatenate([[0], starts])
    if len(stops) == 0:
        stops = np.concatenate([stops, [len(boolean_mask)]])

    # boolean mask may begin or end with True values
    if starts[0] > stops[0]:
        starts = np.concatenate([[0], starts])
    if stops[-1] < starts[-1]:
        stops = np.concatenate([stops, [len(boolean_mask)]])

    return np.c_[starts, stops]


def downsample(arr, factor, func=np.nanmean, axis=-1):
    """
    Simple downsampling function.

    TODO:
    - potentially use scipy.signal.decimate instead
    """
    assert isinstance(factor, int), "Downsampling factor needs to be an integer! Currently: {}".format(type(factor))

    # pad array at end if necessary
    if arr.shape[axis] % factor:
        pad_length = factor - arr.shape[axis] % factor
        arr = pad_along_axis(arr, 0, pad_length, axis, 'constant', constant_values=np.nan)

    return np.apply_along_axis(_downsample, axis, arr, func=func, factor=factor)


def _downsample(vec, factor, func=np.nanmean):
    reshaped = np.reshape(vec, (-1, factor))
    return func(reshaped, axis=-1)


def pad_along_axis(arr, before, after, axis, mode, *args, **kwargs):
    pad_with = [(0, 0) for _ in range(arr.ndim)]
    pad_with[axis] = (before, after)
    padded = np.pad(arr, pad_with, mode, *args, **kwargs)
    return padded


def remove_padding_along_axis(arr, before, after, axis):
    indices = range(before, arr.shape[axis]-after)
    trimmed = np.take(arr, indices, axis=axis)
    return trimmed
