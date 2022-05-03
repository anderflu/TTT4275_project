from xml import dom
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as st
from xlwt import Workbook
import scipy

#Constants
F_s = 10**6
T = 10**(-6)
f_0 = 10**5
omega_0 = 2*np.pi*f_0
phi = np.pi/8
A = 1
N = 513
n_0 = -256
iterations = 2
k = [10, 12, 14, 16, 18, 20]
SNRs = [-10, 0, 10, 20, 30, 40, 50, 60]

#CRLB values
P = N*(N-1)/2
Q = N*(N-1)/(2*N-1)/6

#Write to excel file
wb = Workbook()
#sheet_name = input('Write filename for storing data: ')
sheet_name = 'test'
sheet = wb.add_sheet(sheet_name)
path = 'data/' + sheet_name + '.xls'

#Column names
sheet.write(0, 0, 'FFT length')#B1
sheet.write(0, 1, 'SNR[dB]')
sheet.write(0, 2, 'Mean frequency estimate')
sheet.write(0, 3, 'Mean frequency estimate error')
sheet.write(0, 4, 'Mean frequency estimate error variance')
sheet.write(0, 5, 'CRLB frequency variance')
sheet.write(0, 6, 'Mean phase estimate')
sheet.write(0, 7, 'Mean phase estimate error')
sheet.write(0, 8, 'Mean phase estimate error variance')
sheet.write(0, 9, 'CRLB phase variance')

def createSignal(variance):
    #Complex gaussian white noise
    stdDev = np.sqrt(variance)
    w_Re = np.random.normal(0, stdDev, N)                      
    w_Im = np.random.normal(0, stdDev, N)
    w = []                                                       
    for n in range(N):
        w.append(w_Re[n] + 1j*w_Im[n])

    #Exponential signal
    s = []
    for n in range(N):
        s.append(A*np.exp(1j*(omega_0*(n+n_0)*T+phi)))

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

def fft_x(total_signal, fft_size): #Find the fft-signal and the frequency axis
    fft_x = np.fft.fft(total_signal, n = fft_size)
    fft_freqs = np.fft.fftfreq(n = fft_size, d = T)
    
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

def peak_freq(x_fft, fft_freqs): #Finds dominant frequency of the total signal
    i = np.argmax(x_fft)
    peak = fft_freqs[i]
    return peak, i

def calculate_CRLB(var):
    var_omega = (12*var**2)/(A**2*T**2*N*(N**2-1))
    var_phi = (12*var**2)*(n_0**2*N + 2*n_0*P*Q)/(A**2*N**2*(N**2-1))

    return var_omega, var_phi



def main():

    cntr = 0
    
    M = 2**10
    for snr_db in SNRs: #For each SNR-value
            cntr += 1
            snr = 10**(snr_db/10)
            variance = A**2/(2*snr)
            freq_list = [] #Temporarely store frequency values
            phase_list = [] #Temporarely store phase values
            for l in range(iterations): #For each iteration
                x, s, w = createSignal(variance)
                x_fft, freq= fft_x(x, M)

                dominantFreq, n = peak_freq(x_fft, freq)
                m_star = np.argmax(x_fft)
                #omega_hat = 2*np.pi*dominantFreq
                #error_omega = omega_0-omega_hat
                #error_omega_list.append(error_omega)
                omega_hat = m_star/(M*T)
                freq_list.append(dominantFreq)

                phase = np.angle(np.exp(-(1j*omega_hat*n_0*T))*x_fft[n])
                phase_list.append(phase)
                
                #plot_signals(x, s, w)
                #plot_spectrum(x_fft, freq)
            
            mean_freq = st.mean(freq_list)
            mean_freq_error = f_0 - mean_freq
            mean_phase_error_variance = st.variance(freq_list)
            

            mean_phase = st.mean(phase_list)
            mean_phase_error = phi - mean_phase
            mean_phase_error_variance = st.variance(phase_list)

            omega_CRLB, phi_CRLB = calculate_CRLB(variance)

            sheet.write(cntr, 0, '2^'+str(M)) #FFT length
            sheet.write(cntr, 1, snr_db) #SNR[dB]
            sheet.write(cntr, 2, mean_freq) #Mean f estimate
            sheet.write(cntr, 3, mean_freq_error) #Mean f estimate error
            sheet.write(cntr, 4, mean_phase_error_variance) #Mean f estimate error variance
            sheet.write(cntr, 5, omega_CRLB) #CRLB freq
            sheet.write(cntr, 6, mean_phase) #Mean phi esitmate
            sheet.write(cntr, 7, mean_phase_error) #Mean phi estimate error
            sheet.write(cntr, 8, mean_phase_error_variance) #Mean phi estimate error variance
            sheet.write(cntr, 9, phi_CRLB) #CRLB phase
    print('Data successfully written to file')    
    wb.save(path)
                

    
main()


#TO DO:


#Finne dominant fase til signalet
#Hvorfor dele på MT i likning 8


#FFT-size = 2**20 er ikke mulig. Bruk estimat for FFT size 2**10, og tune estimatet med numerical search method
#scipy.optimize.minimize using Nelder-Mead in Python

#Få vekk stygg linje