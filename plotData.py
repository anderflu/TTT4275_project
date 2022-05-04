import xlrd
import os
import numpy as np
import matplotlib.pyplot as plt


workbook = xlrd.open_workbook(r"data/test.xls")

sheet1 = workbook.sheet_by_index(0)

row_count = sheet1.nrows
col_count = sheet1.ncols

for cur_row in range(0, row_count):
    for cur_col in range(0, col_count):
        cell = sheet1.cell(cur_row, cur_col)
        #print(cell.value, cell.ctype)

x_axis = [-10,0,10,20,30,40,50,60]
freq_CRLB = []
phase_CRLB = []
for row in range(1,9):
    freq_CRLB_val = sheet1.cell(row,5) 
    phase_CRLB_val = sheet1.cell(row,9)
    freq_CRLB.append(freq_CRLB_val.value/(4*np.pi**2))
    phase_CRLB.append(phase_CRLB_val.value)
    
freq = []
phase = []
for row in range(1, sheet1.nrows):
    freq_val = sheet1.cell(row,4)
    phase_val = sheet1.cell(row,8)
    freq.append(freq_val.value)
    phase.append(phase_val.value)


chunked_freq = []
chunked_phase = []
for i in range(0, len(freq),len(x_axis)):
    chunked_freq.append(freq[i:i+len(x_axis)])
    chunked_phase.append(phase[i:i+len(x_axis)])
    



def plot_variance(x): #Plots the signals S & w and x in two separate plots
    #plt.subplots_adjust(vspace = 0.6)
    plt.subplot(1,2,1)
    plt.title("Variance of frequency")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Variance [Hz^2]")
    plt.yscale('log')
    plt.plot(x,freq_CRLB, color = 'yellow')
    for i in range(6):
        plt.plot(x,chunked_freq[i])
        

    plt.subplot(1,2,2)
    plt.title("Variance of phase")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Variance [rad^2]")
    plt.yscale('log')
    plt.plot(x,phase_CRLB, color = 'yellow')
    for i in range(6):
        plt.plot(x,chunked_phase[i])

    plt.show()


plot_variance(x_axis)
