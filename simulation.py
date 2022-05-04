import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from xlwt import Workbook, XFStyle
import time
import os
from scipy import optimize

start = time.process_time()
 





#Constants
F_s = 10**6
T = 10**(-6)
f_0 = 10**5
omega_0 = 2*np.pi*f_0
phi = np.pi/8
A = 1
N = 513
n_0 = -256
iterations = 10
k = [10, 12, 14, 16, 18, 20]
SNRs = [-10, 0, 10, 20, 30, 40, 50, 60]

#CRLB values
P = N*(N-1)/2
Q = N*(N-1)*(2*N-1)/6

#Write to excel file
wb = Workbook()
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

#Cell formatting
sheet.col(0).width = 2400
sheet.col(1).width = 2000
sheet.col(2).width = 5600
sheet.col(3).width = 5600
sheet.col(4).width = 5600
sheet.col(5).width = 5600
sheet.col(6).width = 5600
sheet.col(7).width = 5600
sheet.col(8).width = 5600
sheet.col(9).width = 5600
'''
dec0 = XFStyle()
dec0.num_format_str = '0.0'
dec3 = XFStyle()
dec3.num_format_str = '0.000000000'
'''

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

    
def plot_spectrum(x_fft, freq): #Plots the power spectrum
    middle_index = np.argmax(freq) + 1
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    #plt.xlim(-450000, 450000)
    #plt.ylim(-1, 200)
    plt.plot(freq[:middle_index], 20*np.log10(np.abs(x_fft[:middle_index])), color = '#1f77b4')
    plt.plot(freq[middle_index:], 20*np.log10(np.abs(x_fft[middle_index:])), color = '#1f77b4')
    plt.show()


def calculate_CRLB(var):
    var_omega = (12*var)/(A**2*T**2*N*(N**2-1))
    var_phi = (12*var)*(n_0**2*N + 2*n_0*P+Q)/(A**2*N**2*(N**2-1))

    return var_omega, var_phi

def functionToBeMinimized(f_variable):
    f_var_sliced=f_variable[0]
    #Creating all nesicarry signal in this function, one can use the create signal function
    s=[] #Signal without noise
    for n in range(N):
        s.append(A*np.exp(1j*(2*np.pi*f_var_sliced*(n+n_0)*T+phi)))
    
    snr_db=60
    snr=10**(snr_db/10)
    variance=A**2/(2*snr)
    stdDev = np.sqrt(variance)
    w_Re = np.random.normal(0, stdDev, N)                      
    w_Im = np.random.normal(0, stdDev, N)
    w= [] #Complex gaussian white noise
    for n in range(N):
        w.append(w_Re[n] + 1j*w_Im[n])
    
    x=[]#Total signal
    for n in range (N):
        x.append(s[n]+w[n])

    fftGuess=np.fft.fft(s,2**10)
    xbFFT=np.fft.fft(x,2**10)

    MSE=np.square(np.subtract(abs(fftGuess),abs(xbFFT))).mean()
    return MSE

def main():

    cntr = 0
    for i in k: #For each FFT-size
        M = 2**i
        for snr_db in SNRs: #For each SNR-value
            cntr += 1
            snr = 10**(snr_db/10)
            variance = A**2/(2*snr)
            freq_list = [] #Temporarely store frequency values
            phase_list = [] #Temporarely store phase values
            for l in range(iterations): #For each iteration
                x, s, w = createSignal(variance)
                x_fft, freq= fft_x(x, M)
                #Two ways of calculating the frequency estimate
                m_star = np.argmax(x_fft)                      #Abs av FFT gir lavere varians
                dominantFreq = freq[m_star]
                f_hat = m_star/(M*T)
                freq_list.append(f_hat)
                
                phase = np.angle(np.exp(-(2j*np.pi*f_hat*n_0*T))*x_fft[m_star])
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

            #Write data to excel
            if (cntr-1)%len(SNRs) ==0:
                sheet.write(cntr, 0, '2^'+str(i)) #FFT length
            sheet.write(cntr, 1, snr_db) #SNR[dB]
            sheet.write(cntr, 2, mean_freq) #Mean f estimate
            sheet.write(cntr, 3, mean_freq_error) #Mean f estimate error
            sheet.write(cntr, 4, mean_phase_error_variance) #Mean f estimate error variance
            sheet.write(cntr, 5, omega_CRLB) #CRLB freq
            sheet.write(cntr, 6, mean_phase) #Mean phi esitmate
            sheet.write(cntr, 7, mean_phase_error) #Mean phi estimate error
            sheet.write(cntr, 8, mean_phase_error_variance) #Mean phi estimate error variance
            sheet.write(cntr, 9, phi_CRLB) #CRLB phase

            '''if (cntr-1)%len(SNRs) ==0:
                sheet.write(cntr, 0, '2^'+str(i)) #FFT length
            sheet.write(cntr, 1, snr_db) #SNR[dB]
            sheet.write(cntr, 2, mean_freq, dec0) #Mean f estimate
            sheet.write(cntr, 3, mean_freq_error, dec0) #Mean f estimate error
            sheet.write(cntr, 4, mean_phase_error_variance, dec3) #Mean f estimate error variance
            sheet.write(cntr, 5, omega_CRLB, dec3) #CRLB freq
            sheet.write(cntr, 6, mean_phase, dec3) #Mean phi esitmate
            sheet.write(cntr, 7, mean_phase_error, dec3) #Mean phi estimate error
            sheet.write(cntr, 8, mean_phase_error_variance, dec3) #Mean phi estimate error variance
            sheet.write(cntr, 9, phi_CRLB, dec3) #CRLB phase'''
    
    wb.save(path)
    print('Data successfully written to '+ sheet_name + '.xls')    
    print('Time spent:', time.process_time() - start, 'seconds')

    print("Excercise 1b):")
    result=optimize.minimize(functionToBeMinimized,100000,method='Nelder-Mead')

    mse = []
    t=[1,2]
    for f in range(60000,140000,100):
        t[0]=f
        mse.append(functionToBeMinimized(t))

    plt.figure(1)
    plt.title("MSE")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mean Square Error")
    plt.plot(np.arange(60000,140000,100),mse)
    plt.show()
    print("The guess with noise and FFT length 2^10 before finetuning: ",)
    print("The guess after finetuning with snr_dB=60: ", result.x[0])

                

    
main()


#TO DO:

#Finne dominant fase til signalet

#Implementere 1b_ny inn i simulation

#FFT-size = 2**20 er ikke mulig. Bruk estimat for FFT size 2**10, og tune estimatet med numerical search method
#scipy.optimize.minimize using Nelder-Mead in Python

#Få vekk stygg linje