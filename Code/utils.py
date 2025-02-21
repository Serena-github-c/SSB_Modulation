# this module contains the functions to create and animate plots for signals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Animation parameters
chunk_size = 1024  # Size of chunks to process for animation
interval = (chunk_size / 44100) * 1000  # Time interval for animation frames in milliseconds

# Function to update plot for each frame in the animation
def update_plot(frame, t, signals, lines, ax, is_freq=False, ylims=None):
    # Define the data range for this frame
    start, end = frame * chunk_size, (frame + 1) * chunk_size
    if end > len(signals[0]):  # Ensure we don't exceed signal length
        end = len(signals[0])
    
    for i, signal in enumerate(signals):
        if is_freq:  # If plotting in frequency domain
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal[start:end])))  # Compute FFT
            freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))  # Frequency vector
            lines[i].set_data(freq, spectrum)  # Update frequency plot
            ax[i, 1].set_xlim(freq.min(), freq.max())
            ax[i, 1].set_ylim(1e-3, 1.1 * spectrum.max())
        else:  # If plotting in time domain
            lines[i].set_data(t[start:end], signal[start:end])  # Update time plot
            ax[i, 0].set_xlim(t[start], t[end])
            if ylims:  # Set y-limits if provided
                ax[i, 0].set_ylim(ylims)
            else:
                ax[i, 0].set_ylim(-1.1, 1.1)  # Default y-limits
    return lines



# Function to animate original and modulated signals in subplots
def animate_signals_in_subplots(t, signals, fs, window_title):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # Create a 2x2 grid of subplots
    fig.canvas.manager.set_window_title(window_title)  # Set window title
    
    # Set titles for each subplot
    axs[0, 0].set_title("Original Audio - Time Domain")
    axs[0, 1].set_title("Original Audio - Frequency Domain")
    axs[1, 0].set_title("SSB Modulated Signal - Time Domain")
    axs[1, 1].set_title("SSB Modulated Signal - Frequency Domain")

    # Initialize empty lines for animation
    time_lines = [axs[i, 0].plot([], [])[0] for i in range(2)]
    freq_lines = [axs[i, 1].plot([], [])[0] for i in range(2)]

    # Set axis labels
    for ax in axs.flat:
        ax.set_xlabel("Time (s)" if ax in axs[:, 0] else "Frequency (Hz)")
        ax.set_ylabel("Amplitude" if ax in axs[:, 0] else "Magnitude")
        ax.set_yscale('log' if ax in axs[:, 1] else 'linear')
    
    # Initialize function for animation
    def init():
        for line in time_lines + freq_lines:
            line.set_data([], [])
        return time_lines + freq_lines

    # Animation function
    def animate(frame):
        update_plot(frame, t, signals, time_lines, axs)  # Update time domain plots
        update_plot(frame, t, signals, freq_lines, axs, is_freq=True)  # Update frequency domain plots
        if frame * chunk_size >= len(t):  # Stop animation if end is reached
            anim.event_source.stop()
            plt.close(fig)  # Close the figure window when animation stops
        return time_lines + freq_lines

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t) // chunk_size, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()



# Function to animate the demodulated signal in separate plots
def animate_demodulated_signal(t, signal, fs, window_title):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  
    fig.canvas.manager.set_window_title(window_title)  # Set window title

    # Set titles for each subplot
    axs[0].set_title("Demodulated Signal - Time Domain")
    axs[1].set_title("Demodulated Signal - Frequency Domain")

    # Initialize empty lines for animation
    time_line, = axs[0].plot([], [])
    freq_line, = axs[1].plot([], [])
    
    # Set axis labels and limits
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_ylim(-1.1, 1.1)
    
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_yscale('log')

    # Initialize function for animation
    def init():
        time_line.set_data([], [])
        freq_line.set_data([], [])
        return [time_line, freq_line]

    # Animation function
    def animate(frame):
        start, end = frame * chunk_size, (frame + 1) * chunk_size
        if end > len(signal):
            end = len(signal)
        # Update time domain plot
        time_line.set_data(t[start:end], signal[start:end])
        axs[0].set_xlim(t[start], t[end])
        
        # Update frequency domain plot
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal[start:end])))
        freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
        freq_line.set_data(freq, spectrum)
        axs[1].set_xlim(freq.min(), freq.max())
        axs[1].set_ylim(1e-3, 1.1 * spectrum.max())
        
        return [time_line, freq_line]

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t) // chunk_size, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()



# Function to plot static graphs
def plot_static_graphs(t, signals, demodulated_signal, fs):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))  # Create a 3x2 grid of subplots

    # Plot Original Audio - Time Domain
    axs[0, 0].plot(t, signals[0])
    axs[0, 0].set_title("Original Audio - Time Domain")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True)
    axs[0, 0].set_xticklabels([])  # Remove x-axis labels

    # Plot Original Audio - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signals[0])))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[0, 1].plot(freq, spectrum)
    axs[0, 1].set_title("Original Audio - Frequency Domain")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True)  # Add grid lines
    axs[0, 1].set_xticklabels([])  # Remove x-axis labels

    # Plot SSB Modulated Signal - Time Domain
    axs[1, 0].plot(t, signals[1])
    axs[1, 0].set_title("SSB Modulated Signal - Time Domain")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True)  # Add grid lines
    axs[1, 0].set_xticklabels([])  # Remove x-axis labels

    # Plot SSB Modulated Signal - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signals[1])))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[1, 1].plot(freq, spectrum)
    axs[1, 1].set_title("SSB Modulated Signal - Frequency Domain")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True)  # Add grid lines
    axs[1, 1].set_xticklabels([])  # Remove x-axis labels

    # Plot Demodulated Signal - Time Domain
    axs[2, 0].plot(t, demodulated_signal)
    axs[2, 0].set_title("Recovered Signal - Time Domain")
    axs[2, 0].set_xlabel("Time (s)")  # Add x-axis label
    axs[2, 0].set_ylabel("Amplitude")
    axs[2, 0].grid(True)  # Add grid lines

    # Plot Demodulated Signal - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(demodulated_signal)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[2, 1].plot(freq, spectrum)
    axs[2, 1].set_title("Recovered Signal - Frequency Domain")
    axs[2, 1].set_xlabel("Frequency (Hz)")  # Add x-axis label
    axs[2, 1].set_ylabel("Magnitude")
    axs[2, 1].set_yscale('log')
    axs[2, 1].grid(True)  # Add grid lines

    # Adjust layout to make space for x-axis labels
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for x-axis labels
    plt.show()




#NOTE
'''
For the animated signals, since the audio clip is of  10 seconds, the plot is repeating an infinite loop.
The animated plots are shown for a particular snapshot of time between 0 and 10 seconds. So it keeps changing after refreshing the window tab. 
'''