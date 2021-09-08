import numpy as np

def calc_zscore(s):
    """
    Function that z-score transforms each value of a 2D array 
    (not along any axis). numba-compatible.

    Parameters
    ----------
    spec : 2D numpy array (numeric)

    Returns
    -------
    spec : 2D numpy array (numeric)
           the z-transformed array

    """
    spec = s.copy()
    mn = np.mean(spec)
    std = np.std(spec)
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            spec[i,j] = (spec[i,j]-mn)/std
    return spec

def pad_spectro(spec,maxlen):
    """
    Function that Pads a spectrogram with shape (X,Y) with 
    zeros, so that the result is in shape (X,maxlen)

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins and Y timeframes
    maxlen: maximal length (integer)

    Returns
    -------
    padded_spec : 2D numpy array (numeric)
                  a zero-padded spectrogram S(X,maxlen) with X frequency bins 
                  and maxlen timeframes

    """
    padding = maxlen - spec.shape[1]
    z = np.zeros((spec.shape[0],padding))
    padded_spec=np.append(spec, z, axis=1)
    return padded_spec
    
    
def pad_transform_spectro(spec,maxlen):
    """
    Function that encodes a 2D spectrogram in a 1D array, so that it 
    can later be restored again.
    Flattens and pads a spectrogram with default value 999
    to a given length. Size of the original spectrogram is encoded
    in the first two cells of the resulting array

    Parameters
    ----------
    spec : 2D numpy array (numeric)
           a spectrogram S(X,Y) with X frequency bins and Y timeframes
    maxlen: Integer 
            n of timeframes to which spec should be padded

    Returns
    -------
    trans_spec : 1D numpy array (numeric)
                 the padded and flattened spectrogram 

    """       
    flat_spec = spec.flatten()
    trans_spec = np.concatenate((np.asarray([spec.shape[0], spec.shape[1]]), flat_spec, np.asarray([999]*(maxlen-flat_spec.shape[0]-2))))
    trans_spec = np.float64(trans_spec)
    
    return trans_spec
