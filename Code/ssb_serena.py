#IMPLEMENTATION OF SSB AMPLITUDE MODULATION IN PYTHON

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
import sounddevice as sd
from utils import *

def load_audio(filename):
    '''
    Load and preprocess the audio file
    '''
    fs, audio = wavfile.read(filename)
    
    # Convert to mono if it's stereo (2D array with shape [samples, 2])
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)  # Average left and right channels
    
    # Normalize
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    # Limit audio length to 10 seconds
    if len(audio) > 10 * fs:  
        audio = audio[:30 * fs]

    return fs, audio


def ssb_modulation_hilbert(x, fc, fs, lower_sideband=False):
    '''
    Implement SSB modulation using the Hilbert transform
    x: signal at transmitter to modulate
    fc: carrier frequency
    fs: signal frequency
    lower_sideband: chooses between USB and LSB

    returns : either lower or upper sideband of the modualated signal
    '''
    analytic_signal = signal.hilbert(x)  
    t = np.arange(len(x)) / fs  # Time vector
    carrier = np.exp(1j * 2 * np.pi * fc * t)  # Carrier signal
    # Modulate the signal and return either upper or lower sideband
    return np.real(analytic_signal * (np.conj(carrier) if lower_sideband else carrier))


def ssb_filter_modulation(x, fc, fs):
    '''
    Implement SSB modulation using a Butterworth bandpass transform
    x: signal at transmitter to modulate
    fc: carrier frequency
    fs: signal frequency
    '''
    t= np.arange(len(x)) / fs
    modulated_signal = x * np.cos(2 *np.pi * fc * t)
    ssb_signal = apply_bandpass_filter(modulated_signal, fc, fc + 500, fs)
    return ssb_signal



def ssb_demodulate_butterworth(r, fc, fs, lower_sideband=False):
    ''''
        This function demodulates the signal at the reciever using the butterworth filter
        r: recieved signal r(t) 
        fc: carrier frequency
        fs: signal frequency
        lower_sideband: chooses between USB and LSB
    '''
    t = np.arange(len(r)) / fs  # Time vector
    carrier = np.exp(-1j * 2 * np.pi * fc * t) 
    if lower_sideband:
        carrier = np.conj(carrier)  # Use conjugate for lower sideband
    demodulated = r * carrier  
    nyq = 0.5 * fs  # Nyquist frequency

    # Low-pass Butterworth filter to extract baseband signal
    b, a = signal.butter(6, 0.1 * nyq / nyq, btype='low')
    return np.real(signal.filtfilt(b, a, demodulated))



def ssb_demodulate_coherent(r, fc, fs):
    ''''
        This function demodulates the signal using a coherent band-pass filter
        r: recieved signal r(t) 
        fc: carrier frequency
        fs: signa frequency
        lower_sideband: chooses between USB and LSB
    '''    
    t = np.arange(len(r))/fs
    carrier = np.cos(2* np.pi * fc * t)
    demodulated = r * carrier
    return apply_lowpass_filter(demodulated, cutoff=4000, fs=fs)

# Apply a low-pass filter to cut off unwanted high-frequency components at the reciever
def apply_lowpass_filter(x, cutoff, fs, order=6):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, x)


def apply_bandpass_filter(r, lowcut, highcut, fs, order=5):
    '''
    This function implements the bandpass filter
    r : recieved signal
    lowcut:
    highcut:
    fs: carrier frequency
    order: 
    '''
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')

    return signal.filtfilt(b, a, r)


# Function to add noise
def add_noise(x, snr_db):
    ''''
    Function to add noise to the signal 
    x: signal
    snr_db: Signal to Noise Ratio in dB (decibels)
    '''
    signal_power= np.mean(np.abs(x)**2)
    noise_power= signal_power / (10**(snr_db/10))
    noise= np.sqrt(noise_power) * np.random.randn(*x.shape)
    return x + noise

