import os
import numpy as np
import soundfile as sf
import io
import librosa



def generate_mel_spectrogram(data, rate, n_mels, window, fft_win , fft_hop, fmax):
    
    """
    Function that generates mel spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    rate: numeric(integer)
          samplerate in Hz
    n_mels: numeric (integer)
            number of mel bands
    window: string
            spectrogram window generation type ('hann'...)
    fft_win: numeric (float)
             window length in s
    fft_hop: numeric (float)
             hop between window start in s 

    Returns
    -------
    result : 2D np.array
             Mel-transformed spectrogram, dB scale

    Example
    -------
    >>> 
    
    """
    spectro = np.nan
    
    try:
        n_fft  = int(fft_win * rate) 
        hop_length = int(fft_hop * rate) 

        s = librosa.feature.melspectrogram(y = data ,
                                           sr = rate, 
                                           n_mels = n_mels , 
                                           fmax = fmax, 
                                           n_fft = n_fft,
                                           hop_length = hop_length, 
                                           window = window, 
                                           win_length = n_fft)

        spectro = librosa.power_to_db(s, ref=np.max)
    except:
        print("Failed to generate spectrogram.")

    return spectro




def read_wavfile(filename, channel=0):    
    """
    Function that reads audio data and sr from audiofile
    If audio is stereo, channel 0 is selected by default.

    Parameters
    ----------
    filename: String
              path to wav file
    
    channel: Integer (0 or 1)
             which channel is selected for stereo files
             default is 0
          
    Returns
    -------
    data : 1D np.array
           Raw audio data (Amplitude)
           
    sr: numeric (Integer)
        Samplerate (in Hz)
    """
    data = np.nan
    sr = np.nan
    
    if os.path.exists(filename):
        try:
            data, sr = sf.read(filename)
            if data.ndim>1:
                data = data[:,channel]
        except:
            print("Couldn't read: ", filename)
    else:
        print("No such file or directory: ", filename)


    return data, sr

