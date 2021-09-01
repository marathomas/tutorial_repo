import os
import numpy as np
import soundfile as sf
import io
import librosa
from scipy.signal import butter, lfilter


def generate_mel_spectrogram(data, rate, n_mels, window, fft_win , fft_hop, fmax, fmin=0):
    
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
                                           fmin = fmin,
                                           n_fft = n_fft,
                                           hop_length = hop_length, 
                                           window = window, 
                                           win_length = n_fft)

        spectro = librosa.power_to_db(s, ref=np.max)
    except:
        print("Failed to generate spectrogram.")

    return spectro


def generate_stretched_mel_spectrogram(data, sr, duration, n_mels, window, fft_win , fft_hop, MAX_DURATION):
    """
    Function that generates stretched mel spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    sr: numeric(integer)
          samplerate in Hz
    duration: numeric (float)
              duration of audio in seconds
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
             stretched, mel-transformed spectrogram, dB scale
    -------
    >>> 
    
    """
    n_fft  = int(fft_win * sr) 
    hop_length = int(fft_hop * sr) 
    stretch_rate = duration/MAX_DURATION
    
    # generate normal spectrogram (NOT mel transformed)
    D = librosa.stft(y=data, 
                     n_fft = n_fft,
                     hop_length = hop_length,
                     window=window,
                     win_length = n_fft
                     )
    
    # Stretch spectrogram using phase vocoder algorithm
    D_stretched = librosa.core.phase_vocoder(D, stretch_rate, hop_length=hop_length) 
    D_stretched = np.abs(D_stretched)**2
    
    # mel transform
    spectro = librosa.feature.melspectrogram(S=D_stretched,  
                                            sr=sr,
                                            n_mels=n_mels,
                                            fmax=4000)
        
    # Convert to db scale
    s = librosa.power_to_db(spectro, ref=np.max)

    return s

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



# Butter bandpass filter implementation:
# from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_bandpass_filter(data, lowcut, highcut, sr, order=5):
    """
    Function that applies a butter bandpass filter on audio data 
    and returns the filtered audio

    Parameters
    ----------
    data: 1D np.array
          audio data (amplitude)
    
    lowcut: Numeric
            lower bound for bandpass filter
            
    highcut: Numeric
             upper bound for bandpass filter
             
    sr: Numeric
        samplerate in Hz
    
    order: Numeric
           order of the filter
    
    Returns
    -------
    filtered_data : 1D np.array
                    filtered audio data 
    """
    
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered_data = lfilter(b, a, data)
    return filtered_data
    
    
