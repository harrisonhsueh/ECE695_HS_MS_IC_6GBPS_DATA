import numpy as np
import matplotlib.pyplot as plt
import csv

def get_data(filename):
    data=[[],[]]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data[0].append(float(row[0]))
            data[1].append(float(row[1]))
    return np.asarray(data)
#signal is the switching signal
#time is time base
#reference is the other value we plot frequency against later, such as control voltage
def get_freq(signal,reference,time,threshold,interpolate=0):
    rising_edges = []
    VC_rising_edges = []
    frequency = []
    for i in range(1, (len(time))):
        if (signal[i - 1] < threshold and signal[i] > threshold):
            if (interpolate==1):
                proportion_above = (signal[i]-threshold)/(signal[i]-signal[i-1])
                rising_edges.append(time[i]-proportion_above*(time[i]-time[i-1]))
                VC_rising_edges.append(reference[i]-proportion_above*(reference[i]-reference[i-1]))
            else:
                rising_edges.append(time[i])
                VC_rising_edges.append(reference[i])
    rising_edges = np.asarray(rising_edges)
    VC_rising_edges = np.asarray(VC_rising_edges)
    for i in range(1, len(rising_edges)):
        frequency.append(1 / (rising_edges[i] - rising_edges[i - 1]))
    frequency = np.asarray(frequency)
    return rising_edges, VC_rising_edges, frequency
'''
VC = get_data("VC.csv")
VOUT = get_data("VOUT.csv")
VC2 = get_data("VC_541mVparalleltail1uA500ohm90nm.csv")#VC2 = get_data("VC_500mVparalleltail.csv")
VOUT2 = get_data("VOUT_541mVparalleltail1uA500ohm90nm.csv")#VOUT2 = get_data("VOUT_500mVparalleltail.csv")
VC3 = get_data("VC_603mVparalleltail1p52uA2x10kohm180nm.csv")#get_data("VC_520mVparalleltail.csv")
VOUT3 = get_data("VOUT_603mVparalleltail1p52uA2x10kohm180nm.csv")#get_data("VOUT_520mVparalleltail.csv")
VC4 = get_data("VC_729mVbtail1p34uA2mult10kohm90nm.csv")
VOUT4 = get_data("VOUT_729mVbtail1p34uA2mult10kohm90nm.csv")
VC5 = get_data("VC_729mVbtail1p34uA2mult10kohm1000nm.csv")
VOUT5 = get_data("VOUT_729mVbtail1p34uA2mult10kohm1000nm.csv")
VC6 = get_data("VC_729mVbtail1p34uA2mult10kohm1000nm1p1vrefbias.csv")
VOUT6 = get_data("VOUT_729mVbtail1p34uA2mult10kohm1000nm1p1vrefbias.csv")
VC7 = get_data("VC_729mVbtail1p34uA2mult10kohm2000nm1p1vrefbias.csv")
VOUT7 = get_data("VOUT_729mVbtail1p34uA2mult10kohm2000nm1p1vrefbias.csv")
VC8 = get_data("VC_729mVbtail8xdiffpair.csv")
VOUT8 = get_data("VOUT_729mVbtail8xdiffpair.csv")
VC9 = get_data("VC_45nm_length.csv")
VOUT9 = get_data("VOUT_45nm_length.csv")
VC10= get_data("VC_450nm_length.csv")
VOUT10 = get_data("VOUT_450nm_length.csv")
VC11= get_data("VC_450nm_length_5kr_0p468Vbias.csv")
VOUT11 = get_data("VOUT_450nm_length_5kr_0p468Vbias.csv")
VC12= get_data("VC_450nm_length_20kr_0p450Vbias.csv")
VOUT12 = get_data("VOUT_450nm_length_20kr_0p450Vbias.csv")
VC13= get_data("VC_4500nm_length_20kr_0p450Vbias.csv")
VOUT13 = get_data("VOUT_4500nm_length_20kr_0p450Vbias.csv")
VC14= get_data("VC_4500nm_length_20kr_0p450Vbias_0p1noise.csv")
VOUT14 = get_data("VOUT_4500nm_length_20kr_0p450Vbias_0p1noise.csv")
'''
'''
VC15= get_data("VC_1400nm.csv")
VOUT15 = get_data("VOUT_1400nm.csv")
VC16= get_data("VC_1400nm_short_0p01noise.csv")
VOUT16 = get_data("VOUT_1400nm_short_0p01noise.csv")
VC17= get_data("VC_1400nm_short_0p1noise.csv")
VOUT17 = get_data("VOUT_1400nm_short_0p1noise.csv")
VC18= get_data("VC_1400nm_short_1p0noise.csv")
VOUT18 = get_data("VOUT_1400nm_short_1p0noise.csv")
VC19= get_data("VC_1400nm_long_0p1noise.csv")
VOUT19 = get_data("VOUT_1400nm_long_0p1noise.csv")
VC20= get_data("VC_5_stage_first_try.csv")
VOUT20 = get_data("VOUT_5_stage_first_try.csv")
VC21= get_data("VC_redo6GHZ.csv")
VOUT21 = get_data("VOUT_redo6GHZ.csv")
VC22 = get_data("VC_6GHZ_0p1min.csv")
VOUT22 = get_data("VOUT_6GHZ_0p1min.csv")
VC23 = get_data("VC_6GHZ_0p1actualmin.csv")
VOUT23 = get_data("VOUT_6GHZ_0p1actualmin.csv")
VC24 = get_data("VC_STEP_6GHZ.csv")
VOUT24 = get_data("VOUT_STEP_6GHZ.csv")
VC25 = get_data("VC_6GHZ_minOTA.csv")
VOUT25 = get_data("VOUT_6GHZ_minOTA.csv") #weird, broken current sizing
##26 through 28 had miscalculated bias current sizing
VC26 = get_data("VC_6GHZ_2024-04-22.csv") #redocument and recalculated. 0noise, use initial condition, not conservative, 1us
VOUT26 = get_data("VOUT_6GHZ_2024-04-22.csv")
VC27 = get_data("VC_6GHZ_2024-04-22_noOTA.csv") #redocument and recalculated. 0noise, use initial condition, not conservative, 1us
VOUT27 = get_data("VOUT_6GHZ_2024-04-22_noOTA.csv")
VC28 = get_data("VC_6GHZ_2024-04-22_6x125nm.csv") #redocument and recalculated. 0noise, use initial condition, not conservative, 1us
VOUT28 = get_data("VOUT_6GHZ_2024-04-22_6x125nm.csv")
VC29 = get_data("VC_6GHZ_2024-04-22_biasfix.csv") #redocument and recalculated. 0noise, use initial condition, not conservative, 1us
VOUT29 = get_data("VOUT_6GHZ_2024-04-22_biasfix.csv")
'''
VC30 = get_data("VC_6GHZ_2024-04-22_biasfix_conservative.csv") #conservative sim setting
VOUT30 = get_data("VOUT_6GHZ_2024-04-22_biasfix_conservative.csv")
VC31 = get_data("VC_STEP_6GHZ_2024-04-22.csv") #conservative sim setting
VOUT31 = get_data("VOUT_STEP_6GHZ_2024-04-22.csv")

threshold = 0.55
'''
rising_edges, VC_rising_edges, frequency = get_freq(VOUT[1],VC[1],VOUT[0],threshold)
rising_edges2, VC_rising_edges2, frequency2 = get_freq(VOUT2[1],VC2[1],VOUT2[0],threshold)
rising_edges3, VC_rising_edges3, frequency3 = get_freq(VOUT3[1],VC3[1],VOUT3[0],threshold)
rising_edges4, VC_rising_edges4, frequency4 = get_freq(VOUT4[1],VC4[1],VOUT4[0],threshold)
rising_edges5, VC_rising_edges5, frequency5 = get_freq(VOUT5[1],VC5[1],VOUT5[0],threshold)
rising_edges6, VC_rising_edges6, frequency6 = get_freq(VOUT6[1],VC6[1],VOUT6[0],threshold)
rising_edges7, VC_rising_edges7, frequency7 = get_freq(VOUT7[1],VC7[1],VOUT7[0],threshold)
rising_edges8, VC_rising_edges8, frequency8 = get_freq(VOUT8[1],VC8[1],VOUT8[0],threshold)
rising_edges9, VC_rising_edges9, frequency9 = get_freq(VOUT9[1],VC9[1],VOUT9[0],threshold)
rising_edges10, VC_rising_edges10, frequency10 = get_freq(VOUT10[1],VC10[1],VOUT10[0],threshold)
rising_edges11, VC_rising_edges11, frequency11 = get_freq(VOUT11[1],VC11[1],VOUT11[0],threshold)
rising_edges12, VC_rising_edges12, frequency12 = get_freq(VOUT12[1],VC12[1],VOUT12[0],threshold)
rising_edges13, VC_rising_edges13, frequency13 = get_freq(VOUT13[1],VC13[1],VOUT13[0],threshold)
rising_edges14, VC_rising_edges14, frequency14 = get_freq(VOUT14[1],VC14[1],VOUT14[0],threshold)
'''
'''
rising_edges15, VC_rising_edges15, frequency15 = get_freq(VOUT15[1],VC15[1],VOUT15[0],threshold)
rising_edges15b, VC_rising_edges15b, frequency15b = get_freq(VOUT15[1],VC15[1],VOUT15[0],threshold,interpolate=1)
rising_edges16, VC_rising_edges16, frequency16 = get_freq(VOUT16[1],VC16[1],VOUT16[0],threshold)
rising_edges17, VC_rising_edges17, frequency17 = get_freq(VOUT17[1],VC17[1],VOUT17[0],threshold)
rising_edges18, VC_rising_edges18, frequency18 = get_freq(VOUT18[1],VC18[1],VOUT18[0],threshold)
rising_edges19, VC_rising_edges19, frequency19 = get_freq(VOUT19[1],VC19[1],VOUT19[0],threshold)
rising_edges20, VC_rising_edges20, frequency20 = get_freq(VOUT20[1],VC20[1],VOUT20[0],threshold)
rising_edges21, VC_rising_edges21, frequency21 = get_freq(VOUT21[1],VC21[1],VOUT21[0],threshold)
rising_edges22, VC_rising_edges22, frequency22 = get_freq(VOUT22[1],VC22[1],VOUT22[0],threshold)
rising_edges23, VC_rising_edges23, frequency23 = get_freq(VOUT23[1],VC23[1],VOUT23[0],threshold)
rising_edges24, VC_rising_edges24, frequency24 = get_freq(VOUT24[1],VC24[1],VOUT24[0],threshold)
rising_edges25, VC_rising_edges25, frequency25 = get_freq(VOUT25[1],VC25[1],VOUT25[0],threshold)
rising_edges26, VC_rising_edges26, frequency26 = get_freq(VOUT26[1],VC26[1],VOUT26[0],threshold)
rising_edges26b, VC_rising_edges26b, frequency26b = get_freq(VOUT26[1],VC26[1],VOUT26[0],threshold,interpolate=1)
rising_edges27b, VC_rising_edges27b, frequency27b = get_freq(VOUT27[1],VC27[1],VOUT27[0],threshold,interpolate=1)
rising_edges28b, VC_rising_edges28b, frequency28b = get_freq(VOUT28[1],VC28[1],VOUT28[0],threshold,interpolate=1)
rising_edges29b, VC_rising_edges29b, frequency29b = get_freq(VOUT29[1],VC29[1],VOUT29[0],threshold,interpolate=1)
'''
rising_edges30b, VC_rising_edges30b, frequency30b = get_freq(VOUT30[1],VC30[1],VOUT30[0],threshold,interpolate=1)
rising_edges31b, VC_rising_edges31b, frequency31b = get_freq(VOUT31[1],VC31[1],VOUT31[0],threshold,interpolate=1)

# Create two subplots and unpack the output array immediately

f, axs = plt.subplots(3, 1,sharex=True)
axs[0].plot(1e9*VC31[0],VC31[1],label="VCO Control",color="C0")
#ax1.plot(gain_res[0],gain_res[1],':',color="C1")
#ax1.plot(gain_rc[0],gain_rc[1],color="C1")
axs[1].plot(1e9*VOUT31[0],VOUT31[1],label="VCO Output",linewidth=0.5,color="C1")
axs[2].plot(1e9*rising_edges31b[1:],1e-9*frequency31b,label="VCO Output Frequency",color="C2")

#ax2.plot(phase_res[0],phase_res[1],':',color="C1")
#ax2.plot(phase_rc[0],phase_rc[1],color="C1")
#ax1.text(gain_open[0][200],gain_open[1][200],f'({round(gain_open[0][200])}, {round(gain_open[1][200],2)})',horizontalalignment='right')
#ax1.annotate(f'open loop({round(gain_open[0][200])}Hz, {round(gain_open[1][200],2)}dB)',
#            xy=(gain_open[0][200], gain_open[1][200]), xycoords='data',
#            xytext=(-100, -20), textcoords='offset points',
#            arrowprops=dict(arrowstyle="->"))
#ax1.annotate(f'resistor feedback only({round(gain_res[0][200])}Hz, {round(gain_res[1][200],2)}dB)',
#            xy=(gain_res[0][200], gain_res[1][200]), xycoords='data',
#            xytext=(-110, -55), textcoords='offset points',
#            arrowprops=dict(arrowstyle="->"))
#ax1.annotate(f'low pass({round(gain_rc[0][200])}Hz, {round(gain_rc[1][200],2)}dB)',
#            xy=(gain_rc[0][200], gain_rc[1][200]), xycoords='data',
#            xytext=(50, -35), textcoords='offset points',
#            arrowprops=dict(arrowstyle="->"))
#ax1.annotate(f'low pass -3dB({round(gain_rc[0][770]/1000,2)}kHz, {round(gain_rc[1][770],2)}dB)',
#            xy=(gain_rc[0][770], gain_rc[1][770]), xycoords='data',
#            xytext=(20, -50), textcoords='offset points',
#            arrowprops=dict(arrowstyle="->"))

axs[0].set_title('VCO Output, V_control Step')
#axs[0].legend(["Open Loop", "Resistor Feedback Only","Low Pass Filter"])
#ax2.legend(["phase"])
#ax1.set_xscale('log')
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[2].set_xlabel("Time [ns]")
axs[0].set_ylabel("V")
axs[1].set_ylabel("V")
axs[2].set_ylabel("Frequency [GHz]")
axs[0].grid()
axs[1].grid()
axs[2].grid()

plt.show()

'''
plt.plot(VC_rising_edges[1:],frequency,'.',alpha=0.2,markeredgewidth=0)
plt.plot(VC_rising_edges2[1:],frequency2,'.',alpha=0.2,markeredgewidth=0)
plt.plot(VC_rising_edges3[1:],frequency3,'.',alpha=0.2,markeredgewidth=0)
plt.plot(VC_rising_edges4[1:],frequency4,'.',alpha=0.2,markeredgewidth=0)
plt.plot(VC_rising_edges5[1:],frequency5,'.',alpha=0.2,markeredgewidth=0,label="1000nm")
plt.plot(VC_rising_edges6[1:],frequency6,'.',alpha=0.2,markeredgewidth=0,label="1.1V bias")
plt.plot(VC_rising_edges7[1:],frequency7,'.',alpha=0.2,markeredgewidth=0,label="2000nm")
plt.plot(VC_rising_edges8[1:],frequency8,'.',alpha=0.2,markeredgewidth=0,label="8xdiffpair")
plt.plot(VC_rising_edges9[1:],frequency9,'.',alpha=0.2,markeredgewidth=0,label="fixed 45nm")
plt.plot(VC_rising_edges10[1:],frequency10,'.',alpha=0.2,markeredgewidth=0,label="fixed 450nm")
plt.plot(VC_rising_edges11[1:],frequency11,'.',alpha=0.2,markeredgewidth=0,label="fixed 450nm 5krbias")
plt.plot(VC_rising_edges12[1:],frequency12,'.',alpha=0.2,markeredgewidth=0,label="fixed 450nm 20krbias")
plt.plot(VC_rising_edges13[1:],frequency13,'.',alpha=0.2,markeredgewidth=0,label="fixed 4500nm")
plt.plot(VC_rising_edges14[1:],frequency14,'.',alpha=0.2,markeredgewidth=0,label="fixed 4500nm 0.1 noise")
'''
'''
plt.plot(VC_rising_edges15[1:],frequency15,'.',alpha=0.2,markeredgewidth=0,label="5GHz, 10GHz/V design")
plt.plot(VC_rising_edges15b[1:],frequency15b,'.',alpha=0.2,markeredgewidth=0,label="5GHz, 10GHz/V design interpolate")
plt.plot(VC_rising_edges20[1:],frequency20,'.',alpha=0.2,markeredgewidth=0,label="5 stage first try")
#plt.plot(VC_rising_edges21[1:],frequency21,'.',alpha=0.2,markeredgewidth=0,label="redo 6GHz")
plt.plot(VC_rising_edges22[1:],frequency22,'.',alpha=0.2,markeredgewidth=0,label="6GHz 0.1V min")
plt.plot(VC_rising_edges23[1:],frequency23,'.',alpha=0.2,markeredgewidth=0,label="6GHz 0.1V actual min")
plt.plot(VC_rising_edges25[1:],frequency25,'.',alpha=0.2,markeredgewidth=0,label="6GHZ minOTA")
plt.plot(VC_rising_edges26b[1:],frequency26b,'.',alpha=0.2,markeredgewidth=0,label="6GHz 2024-04-22 recalc interpolate")
plt.plot(VC_rising_edges27b[1:],frequency27b,'.',alpha=0.2,markeredgewidth=0,label="6GHz 2024-04-22 no OTA interpolate")
plt.plot(VC_rising_edges28b[1:],frequency28b,'.',alpha=0.2,markeredgewidth=0,label="6x125nm interpolate")
plt.plot(VC_rising_edges29b[1:],frequency29b,'.',alpha=0.2,markeredgewidth=0,label="biasfix")
'''
plt.plot(VC_rising_edges30b[1:],frequency30b,'.',alpha=0.2,markeredgewidth=0,label="6GHz design")

#plt.plot(VC_rising_edges16[1:],frequency16+1e9,'.',alpha=0.2,markeredgewidth=0,label="0p01noise")
#plt.plot(VC_rising_edges17[1:],frequency17+2e9,'.',alpha=0.2,markeredgewidth=0,label="0p1noise")
#plt.plot(VC_rising_edges18[1:],frequency18+3e9,'.',alpha=0.2,markeredgewidth=0,label="1p0noise")
#plt.plot(VC_rising_edges19[1:],frequency19+1e9,'.',alpha=0.2,markeredgewidth=0,label="5GHz, sweep back, offset for plot")



#best fit biasfix conservative sim setting
where_vc_in_range = np.logical_or((VC_rising_edges30b[1:]<0.45),(VC_rising_edges30b[1:]>0.55))
where_vc_in_range_4_6 = np.logical_or((VC_rising_edges30b[1:]<0.55),(VC_rising_edges30b[1:]>0.6))

mvc = np.ma.masked_where(where_vc_in_range, VC_rising_edges30b[1:])
mvc = np.ma.compressed(mvc)
mfreq = np.ma.masked_where(where_vc_in_range, frequency30b)
mfreq = np.ma.compressed(mfreq)
mvc_4_6 = np.ma.masked_where(where_vc_in_range_4_6, VC_rising_edges30b[1:])
mvc_4_6 = np.ma.compressed(mvc_4_6)
mfreq_4_6 = np.ma.masked_where(where_vc_in_range_4_6, frequency30b)
mfreq_4_6 = np.ma.compressed(mfreq_4_6)
#plt.plot(np.unique(mvc), np.poly1d(np.polyfit(mvc, mfreq, 1))(np.unique(mvc)),
#         label="K_VCO="+str(round(np.poly1d(np.polyfit(mvc, mfreq, 1))[1]/1e9,1))+"GHz/V")
plt.plot(np.unique([0.55,0.6]), np.poly1d(np.polyfit(mvc_4_6, mfreq_4_6, 1))(np.unique([0.55,0.6])),
         label="K_VCO="+str(round(np.poly1d(np.polyfit(mvc_4_6, mfreq_4_6, 1))[1]/1e9,1))+"GHz/V")

plt.grid()
plt.legend()
#plt.yscale('log')
plt.ylabel("Frequency [Hz]")
plt.xlabel("VCO Control [V]")
plt.title("VCO Output vs Input")
plt.show()
plt.plot(mvc,mfreq)

plt.plot(np.unique(mvc), np.poly1d(np.polyfit(mvc, mfreq, 1))(np.unique(mvc)),
         label="K_VCO="+str(round(np.poly1d(np.polyfit(mvc, mfreq, 1))[1]/1e9,1))+"GHz/V")
plt.show()
