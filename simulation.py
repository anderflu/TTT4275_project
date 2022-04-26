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
N = 10
n_0 = -256
standardDeviation = 0.5





def createSignal(stdDev):                                       #Create the signals
    #Complex gaussian white noise
    w_Re = np.random.normal(0, stdDev, N)                       #List of gaussian noise signal, real values
    w_Im = np.random.normal(0, stdDev, N)                       #List of gaussian noise signal, imaginary values
    w = []                                                      #List of sum of real and imaginary values  
    for n in range(N):
        w.append(w_Re[n] + 1j*w_Im[n])

    #print('W: ', w)
    

    #Exponential signal
    s = []
    for n in range(N):
        s.append(A*math.exp((omega_0*n*T+phi)))

    #Total signal
    x = []
    for n in range(N):
        x.append(s[n]+w[n])

    return x, s, w

def MLE_FFT(sampling_period):
    signal = createSignal(standardDeviation)
    fft_freqs = np.fft.fftfreq(len(signal), sampling_period)
    fft_signal = np.fft.fft(signal)
    phase = np.angle(fft_signal)
    plt.title("Power spectrum of IQ-signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    #plt.xlim(-1500, 1500)
    #plt.ylim(-1, 200)
    plt.plot(fft_signal, 20*np.log10(np.abs(fft_freqs))) # get the power spectrum
    plt.show()
    print("Dominant frequency: ", np.argmax(fft_signal))
    print("Dominant phase: ", phase)

    
def plot_signal(signal):
    plt.plot(signal)
    



def main():
    #MLE_FFT(T)
    x, s, w = createSignal(standardDeviation)
    #print('x: ', x)
    #print('s: ', s)
    print('w: ', w)
    plot_signal(w)
    #plot_signal(x)
    #plot_signal(s)
    plt.show()
main()