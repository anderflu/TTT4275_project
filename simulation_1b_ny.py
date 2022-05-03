from cmath import pi
#from fcntl import F_RDLCK
from xml import dom
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as st
from xlwt import Workbook
from scipy import optimize

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
#Example snr
snr_db=30

#CRLB values
P = N*(N-1)/2
Q = N*(N-1)/(2*N-1)/6

def functionToBeMinimized(f_variable):
    f_var_sliced=f_variable[0]
    #Creating all nesicarry signal in this function, one can use the create signal function
    s=[] #Signal without noise
    for n in range(N):
        s.append(A*np.exp(np.complex(0,1)*((2*np.pi*f_var_sliced)*(n + n_0)*T + phi)))
    
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

print("The guess after finetning: ", result.x[0])