from ssb_serena import *
filename="audio_flute.wav"

def plot_everything (ssb_signal, demodulated_signal, m_d_method, noisy=False):

    # Plot static graphs
    plot_static_graphs(t, [audio, ssb_signal], demodulated_signal, fs)

    # Play and animate the original and SSB modulated signals
    print("Playing original audio and visualizing...")
    sd.play(audio, fs)
    animate_signals_in_subplots(t, [audio, ssb_signal], fs, "Original & SSB Modulated Signals")
    sd.wait()
    '''
    if noisy:
        print("Playing noisy audio and visualizing...")
        sd.play(audio_noisy, fs)
        animate_signals_in_subplots(t, [audio, ssb_signal], fs, "Original & SSB Modulated Signals")
        sd.wait()

    '''
    # Play and animate the demodulated signal
    print("Playing recovered audio...")
    sd.play(demodulated_signal, fs)
    # the sound is very low, we can increase the amplitude
    demodulated_signal = demodulated_signal * 5  # Increase volume by factor of 5
    animate_demodulated_signal(t, demodulated_signal, fs, "Recovered Signal")
    sd.wait()

    # Save the recovered audio
    # normalize first
    demodulated_signal /= np.max(np.abs(demodulated_signal))  # Ensure values are in [-1, 1]
    wavfile.write(f"recovered_audio_{m_d_method}.wav", fs, (demodulated_signal * 32767).astype(np.int16))


# load the audio file
fs, audio= load_audio(filename)
t = np.arange(len(audio)) / fs  # Time vector
fc, N = 5000, 10001  # Carrier frequency and signal length




# Experiment 1
# modulate using hilbert, demodulate using a butterworth filter
print("Experiment 1: \nModulate using hilbert, demodulate using a butterworth filter.")
ssb_signal = ssb_modulation_hilbert(audio, fc, fs, N)
demodulated_signal = ssb_demodulate_butterworth(ssb_signal, fc, fs)
plot_everything(ssb_signal, demodulated_signal, 'm_hilbert_d_butter')

# Experiment 2
# modulate using hilbert, demodulate using a coherent demodulator
print("Experiment 2: \nModulate using hilbert, demodulate using a coherent filter.")
ssb_signal = ssb_modulation_hilbert(audio, fc, fs, N)
demodulated_signal = ssb_demodulate_coherent(ssb_signal, fc, fs)

# Debugging: Check if the signal contains valid values
# normalize audio
audio = audio / np.max(np.abs(audio))
print('checking if audio is normalized  between [-1,1]', audio)
plot_everything(ssb_signal, demodulated_signal ,'m_hilbert_d_coherent')


# Experiment 3
# modulate using a filter, demodulate using a coherent demodulator

print("Experiment 3: \nModulate using a filter, demodulate using a butterworth filter.")
ssb_signal = ssb_filter_modulation(audio, fc, fs)
demodulated_signal = ssb_demodulate_butterworth(ssb_signal, fc, fs)
plot_everything(ssb_signal, demodulated_signal ,'m_filter_d_butter')


# Experiment 4
# modulate using a filter, demodulate using a coherent demodulator
print("Experiment 4: \nModulate using a filter, demodulate using a coherent filter.")
ssb_signal = ssb_filter_modulation(audio, fc, fs)
demodulated_signal = ssb_demodulate_coherent(ssb_signal, fc, fs)
plot_everything(ssb_signal, demodulated_signal ,'m_filter_d_coherent')

'''


# Experiment 5
# Add noise to the signal after modulation to simulate noisy channel
print("Experiment 5: \nAdd noise before modulating using hilbert, demodulate using a butterworth filter.")

for snr_db in [20, 10, 5, 0]:
    print("Signal to noise ratio:", snr_db)
    ssb_signal = ssb_filter_modulation(audio, fc, fs)
    ssb_signal_noisy= add_noise(ssb_signal, snr_db)
    wavfile.write(f"noisy_audio_{snr_db}_snr_db.wav", fs, (ssb_signal_noisy * 32767).astype(np.int16))
    fs, audio_noisy= load_audio(f"noisy_audio_{snr_db}_snr_db.wav")

    demodulated_signal = ssb_demodulate_butterworth(ssb_signal_noisy, fc, fs)
    # the demodulated signal has a lower amplitude, so we need to scale it
    demodulated_signal /= np.max(np.abs(ssb_signal_noisy))  # Normalize
    wavfile.write(f"noisy_audio_recovered_{snr_db}_snr_db.wav", fs, (demodulated_signal * 32767).astype(np.int16))
    plot_everything(ssb_signal_noisy, demodulated_signal, 'm_filter_d_coherent_{snr_db}', noisy=True)


'''