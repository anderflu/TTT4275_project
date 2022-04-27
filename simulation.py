import numpy as np
import math
import matplotlib.pyplot as plt


#Constants
F_s = 10**6
T = 10**(-6)
f_0 = 10**5
omega_0 = 2*np.pi*f_0
phi = np.pi/8
A = 1
N = 513
n_0 = -256
standardDeviation = 0.5


def createSignal(stdDev):
    #Complex gaussian white noise
    w_Re = np.random.normal(0, stdDev, N)                       
    w_Im = np.random.normal(0, stdDev, N)                       
    w = []                                                       
    for n in range(N):
        w.append(w_Re[n] + 1j*w_Im[n])

    #Exponential signal
    s = []
    for n in range(N):
        s.append(A*np.exp(complex(0,1)*(omega_0*n*T+phi)))

    #Total signal
    x = []
    for n in range(N):
        x.append(s[n]+w[n])

    return x, s, w

   
def plot_signals(x, s, w): #Plots the signals S & w and x in two separate plots
    t = np.linspace(start=0, stop=N*T, num=N)
    plt.subplots_adjust(hspace = 0.6)
    plt.subplot(2,1,1)
    plt.title("Signals S and  separately")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.plot(t,s)
    plt.plot(t,w)

    plt.subplot(2,1,2)
    plt.title("Signals S and w combined")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.plot(t,x)

    plt.show()


def peak_freq(x_fft, fft_freqs): #Finds dominant frequency of the total signal
    i = np.argmax(x_fft)
    peak = fft_freqs[i]
    return peak


def fft_x(total_signal): #Find the fft-signal and the frequency axis
    fft_x = np.fft.fft(total_signal)
    fft_freqs = np.fft.fftfreq(n = N, d = T)
    
    return fft_x, fft_freqs


def fft_x_sorted(total_signal): #Sorts fft_freqs chronolgically and lines fft_x up with fft_freqs
    x_fft, fft_freqs = fft_x(total_signal)
    middle_index = np.argmax(fft_freqs) + 1
    first = x_fft[:middle_index]
    second = x_fft[middle_index:]
    x_fft_sorted = []
    for i in range(len(second)):
        x_fft_sorted.append(second[i])
    for i in range(len(first)):
        x_fft_sorted.append(first[i])
    fft_freqs_sorted = fft_freqs.sort() 
    
    return x_fft_sorted, fft_freqs_sorted
    

def plot_spectrum(x_fft, freq): #Plots the power spectrum
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    #plt.xlim(-450000, 450000)
    #plt.ylim(-1, 200)
    plt.plot(freq, 20*np.log10(np.abs(x_fft)))
    plt.show()
    
    
def main():
    x, s, w = createSignal(standardDeviation) 
    x_fft, freq= fft_x(x)
    plot_signals(x,s,w)
    plot_spectrum(x_fft, freq) 
    dominantFreq = peak_freq(x_fft, freq)
    #phase = np.angle(s)
    #dominantPhase = max(phase)
    print("Dominant frequency: ", dominantFreq)
    #print("Dominant phase: ", dominantPhase)
    #phase = np.angle(x[i])
    #print(phase)
    
    
main()