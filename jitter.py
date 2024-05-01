import numpy as np
import matplotlib.pyplot as plt
import csv

#csv filename, sets is number of signal paired with time data. time data is the same though from cadence
def get_data(filename, sets=1):
    data=[]
    for i in range(sets):
        data.append([])
        data.append([])
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            for i in range(sets):
                data[2*i].append(float(row[2*i]))
                data[2*i+1].append(float(row[2*i+1]))
    return np.asarray(data)

# returns:
# rising_edges as sample count
# rising_edge offset based on linear interpolation to next. time ith sample is too high.
# jitter
def get_freq(signal, time, threshold, tstart=0.0, tstop=99.9):
    """
    :param np.array signal: time domain voltage or current signal
    :param np.array time: time stamps for signal values
    :param float threshold: value for when a new cycle is started
    :param float tstart: time when to start using values from the data
    :param float tstop: time when to stop using values from the data
    """
    rising_edges = []
    rising_edge_offset = []
    period = []
    for i in range(1, (len(time))):
        if tstart < time[i] < tstop:
            if signal[i - 1] < threshold < signal[i]:
                rising_edges.append(i)
                proportion_above = (signal[i] - threshold) / (signal[i] - signal[i - 1])
                rising_edge_offset.append(proportion_above * (time[i] - time[i - 1]))
                if len(rising_edges) > 1:
                    period.append((time[rising_edges[-1]] - rising_edge_offset[-1])
                                  - (time[rising_edges[-2]] - rising_edge_offset[-2]))
    rising_edges = np.asarray(rising_edges)
    rising_edge_offset = np.asarray(rising_edge_offset)
    period = np.asarray(period)
    return rising_edges, rising_edge_offset, period

#returns phase jitter as offset from din rising edge. ie, 0.5(1/freq) is ideal.
def get_edges_and_phase(clkout, din, time, threshold_clk, threshold_din, tstart=0.0, tstop=99):
    clkout_rising_edges = []
    clkout_rising_edges_offset = []
    din_edges = []
    din_edges_offset = []
    data_period = []
    clk_period = []
    phase_ps = []
    for i in range(1, (len(time))):
        if tstart < time[i] < tstop:
            if clkout[i - 1] < threshold_clk < clkout[i] and len(din_edges) > 0:
                clkout_rising_edges.append(i)
                proportion_above = (clkout[i] - threshold_clk) / (clkout[i] - clkout[i - 1])
                clkout_rising_edges_offset.append(proportion_above * (time[i] - time[i - 1]))
                phase_ps.append(time[i]-clkout_rising_edges_offset[-1]
                             -(time[din_edges[-1]] - din_edges_offset[-1]))
                if len(clkout_rising_edges) > 1:
                    clk_period.append((time[clkout_rising_edges[-1]] - clkout_rising_edges_offset[-1])
                                  - (time[clkout_rising_edges[-2]] - clkout_rising_edges_offset[-2]))
            if din[i - 1] < threshold_din < din[i] or din[i - 1] > threshold_din > din[i]:
                din_edges.append(i)
                proportion_above = (din[i] - threshold_din) / (din[i] - din[i - 1])
                din_edges_offset.append(proportion_above * (time[i] - time[i - 1]))
                if len(din_edges) > 1:
                    data_period.append((time[din_edges[-1]] - din_edges_offset[-1])
                                  - (time[din_edges[-2]] - din_edges_offset[-2]))
    clkout_rising_edges = np.asarray(clkout_rising_edges)
    clkout_rising_edges_offset = np.asarray(clkout_rising_edges_offset)
    din_edges = np.asarray(din_edges)
    din_edges_offset = np.asarray(din_edges_offset)
    phase_ps = np.asarray(phase_ps)
    data_period = np.asarray(data_period)
    clk_period = np.asarray(clk_period)
    return clkout_rising_edges, clkout_rising_edges_offset, din_edges, din_edges_offset, phase_ps, data_period, clk_period

threshold_clk = 0.55
threshold_din = 0

tstart01 = 5e-9 #get histogram after settling
VOUT01 = get_data("VOUT_575mV_noNoise.csv")
rising_edges01, rising_edges_offset01, period01 = get_freq(VOUT01[1], VOUT01[0], threshold_clk, tstart=tstart01)
mean_period01 = np.mean(period01)
period_jitter01 = period01 - mean_period01
rms_period_jitter01 = np.std(period_jitter01)
print("periods = " + str(len(period01)))
print("mean period = " + str(mean_period01))
print("rms period jitter = " + str(rms_period_jitter01))

tstart02 = 100e-9  #5e-9 #100e-9 get histogram after settling for 600ns file
#VOUT02 = get_data("VOUT_575mV_100GHZnoise.csv")
VOUT02 = get_data("VCO_VOUT_noisy_600ns.csv")
rising_edges02, rising_edges_offset02, period02 = get_freq(VOUT02[1], VOUT02[0], threshold_clk, tstart=tstart02)
mean_period02 = np.mean(period02)
period_jitter02 = period02 - mean_period02
rms_period_jitter02 = np.std(period_jitter02)
print("periods = " + str(len(period02)))
print("mean period = " + str(mean_period02))
print("rms period jitter = " + str(rms_period_jitter02))
DIN02=np.zeros((2,np.size(VOUT02[0])))
DIN02[0]=np.linspace(0,600e-9,np.size(VOUT02[0]))
DIN02[1]=np.sin(np.pi/mean_period02*DIN02[0]) #no 2 in 2pi since data is half the hz
clk_rising_edges02b, clk_rising_edges_offset02b, din_edges02b, din_edges_offset02b, phase_ps02b, data_period02b, clk_period02b = get_edges_and_phase(VOUT02[1], DIN02[1], VOUT02[0], threshold_clk, threshold_din, tstart=tstart02)
mean_phase02b = np.mean(phase_ps02b)
phase_jitter02b = phase_ps02b - mean_phase02b
rms_phase_jitter02b = np.std(phase_jitter02b)
print("sim mean phase = " + str(mean_phase02b))
print("sim rms phase jitter = " + str(rms_phase_jitter02b))


tstart03 = 120e-9
tstop03 = 300e-9
VOUT03 = get_data("VCLK_noiseless_120ns_to_300ns_stable.csv")
DIN03 = get_data("DIN_noiseless_120ns_to_300ns_stable.csv")
tstart03 = 120e-9
tstop03 = 300e-9
VOUT03 = get_data("VCLK_noiseless_200ns_lock.csv")
DIN03 = get_data("DIN_noiseless_200ns_lock.csv")
rising_edges03, rising_edges_offset03, period03 = get_freq(VOUT03[1], VOUT03[0], threshold_clk, tstart=tstart03)
clk_rising_edges03, clk_rising_edges_offset03, din_edges03, din_edges_offset03, phase_ps03, data_period03, clk_period03 = get_edges_and_phase(VOUT03[1], DIN03[1], VOUT03[0], threshold_clk, threshold_din, tstart=tstart03)
clk_rising_edges03b, clk_rising_edges_offset03b, din_edges03b, din_edges_offset03b, phase_ps03b, data_period03b, clk_period03b = get_edges_and_phase(VOUT03[1], DIN03[1], VOUT03[0], threshold_clk, threshold_din, tstart=0)

mean_period03 = np.mean(period03)
period_jitter03 = period03 - mean_period03
rms_period_jitter03 = np.std(period_jitter03)
mean_phase03 = np.mean(phase_ps03)
phase_jitter03 = phase_ps03 - mean_phase03
rms_phase_jitter03 = np.std(phase_jitter03)
print("periods = " + str(len(period03)))
print("mean period = " + str(mean_period03))
print("rms period jitter = " + str(rms_period_jitter03))
print("mean phase = " + str(mean_phase03))
print("rms phase jitter = " + str(rms_phase_jitter03))

tstart04a = 0
tstop04a = 200e-9
data04ax = get_data("data_noisy_140ns_to_300ns_stable.csv", 3)
VOUT04a = data04ax[0:2]
DIN04a = data04ax[2:4]
rising_edges04a, rising_edges_offset04a, period04a = get_freq(VOUT04a[1], VOUT04a[0], threshold_clk, tstart=tstart04a)
clk_rising_edges04a, clk_rising_edges_offset04a, din_edges04a, din_edges_offset04a, phase_ps04a, data_period04a, clk_period04a = get_edges_and_phase(VOUT04a[1], DIN04a[1], VOUT04a[0], threshold_clk, threshold_din, tstart=tstart04a)
clk_rising_edges04a, clk_rising_edges_offset04a, din_edges04a, din_edges_offset04a, phase_ps04a, data_period04a, clk_period04a = get_edges_and_phase(VOUT04a[1], DIN04a[1], VOUT04a[0], threshold_clk, threshold_din, tstart=0)


tstart04 = 100e-9  #tstart04 = 140e-9
tstop04 = 600e-9 #tstop04 = 300e-9
data04 = get_data("data_noisy_600ns.csv",3)
VOUT04 = data04[0:2] #[4:]
DIN04 = data04[2:4]
rising_edges04, rising_edges_offset04, period04 = get_freq(VOUT04[1], VOUT04[0], threshold_clk, tstart=tstart04)
clk_rising_edges04, clk_rising_edges_offset04, din_edges04, din_edges_offset04, phase_ps04, data_period04, clk_period04 = get_edges_and_phase(VOUT04[1], DIN04[1], VOUT04[0], threshold_clk, threshold_din, tstart=tstart04)
clk_rising_edges04b, clk_rising_edges_offset04b, din_edges04b, din_edges_offset04b, phase_ps04b, data_period04b, clk_period04b = get_edges_and_phase(VOUT04[1], DIN04[1], VOUT04[0], threshold_clk, threshold_din, tstart=0)
mean_period04 = np.mean(period04)
period_jitter04 = period04 - mean_period04
rms_period_jitter04 = np.std(period_jitter04)
mean_phase04 = np.mean(phase_ps04)
phase_jitter04 = phase_ps04 - mean_phase04
rms_phase_jitter04 = np.std(phase_jitter04)
print("periods = " + str(len(period04)))
print("mean period = " + str(mean_period04))
print("rms period jitter = " + str(rms_period_jitter04))
print("mean phase = " + str(mean_phase04))
print("rms phase jitter = " + str(rms_phase_jitter04))


tstart05 = 75e-9
tstop05 = 250e-9
data05 = get_data("data_noiseless_300ns_62p5MHz_step_both_directions.csv", 3)
VOUT05 = data05[0:2] #  [4:]
DIN05 = data05[2:4]
rising_edges05, rising_edges_offset05, period05 = get_freq(VOUT05[1], VOUT05[0], threshold_clk, tstart=tstart05)
clk_rising_edges05, clk_rising_edges_offset05, din_edges05, din_edges_offset05, phase_ps05, data_period05, clk_period05 = get_edges_and_phase(VOUT05[1], DIN05[1], VOUT05[0], threshold_clk, threshold_din, tstart=tstart05)
clk_rising_edges05b, clk_rising_edges_offset05b, din_edges05b, din_edges_offset05b, phase_ps05b, data_period05b, clk_period05b = get_edges_and_phase(VOUT05[1], DIN05[1], VOUT05[0], threshold_clk, threshold_din, tstart=tstart05)

tstart06 = 75e-9
tstop06 = 300e-9
data06 = get_data("data_noiseless_300ns_250MHz_step_both_directions.csv", 3)
VOUT06 = data06[0:2] #  [4:]
DIN06 = data06[2:4]
rising_edges06, rising_edges_offset06, period06 = get_freq(VOUT06[1], VOUT06[0], threshold_clk, tstart=tstart06)
clk_rising_edges06, clk_rising_edges_offset06, din_edges06, din_edges_offset06, phase_ps06, data_period06, clk_period06 = get_edges_and_phase(VOUT06[1], DIN06[1], VOUT06[0], threshold_clk, threshold_din, tstart=tstart06)
clk_rising_edges06b, clk_rising_edges_offset06b, din_edges06b, din_edges_offset06b, phase_ps06b, data_period06b, clk_period06b = get_edges_and_phase(VOUT06[1], DIN06[1], VOUT06[0], threshold_clk, threshold_din, tstart=tstart06)


tstart07 = 70e-9
tstop07 = 1e1
data07 = get_data("data_noiseless_300ns_125MHz_step_both_directions.csv",3)
VOUT07 = data07[0:2] #[4:]
DIN07 = data07[2:4]
rising_edges07, rising_edges_offset07, period07 = get_freq(VOUT07[1], VOUT07[0], threshold_clk, tstart=tstart07)
clk_rising_edges07, clk_rising_edges_offset07, din_edges07, din_edges_offset07, phase_ps07, data_period07, clk_period07 = get_edges_and_phase(VOUT07[1], DIN07[1], VOUT07[0], threshold_clk, threshold_din, tstart=tstart07)
clk_rising_edges07b, clk_rising_edges_offset07b, din_edges07b, din_edges_offset07b, phase_ps07b, data_period07b, clk_period07b = get_edges_and_phase(VOUT07[1], DIN07[1], VOUT07[0], threshold_clk, threshold_din, tstart=tstart07)


def plot_phase_jitter(data,clk,data_edges,data_edges_offset,title):# plot phase jitter time overlay
    tminus_samples = 600
    tplus_samples = 700
    fig, ax1 = plt.subplots()

    color1 = 'tab:red'
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'tab:blue'
    ax2.tick_params(axis='y', labelcolor=color2)
    num_samples = len(data_edges) - 1  # in case no clock after last edge
    rel_samples_fully_dark = 0.01
    for i in range(
            num_samples):
        ax1.plot(1e12 * (data[0][data_edges[i] - tminus_samples:data_edges[i] + tplus_samples]
                         - data[0][data_edges[i]] + data_edges_offset[i]),
                 data[1][data_edges[i] - tminus_samples:data_edges[i] + tplus_samples],
                 alpha = 1/rel_samples_fully_dark/num_samples, color=color1)
        #clock data from the asociated rising edge, minusplus some data. offset by time
        ax2.plot(1e12 * (data[0][data_edges[i] - tminus_samples:data_edges[i] + tplus_samples]
                             - data[0][data_edges[i]] + data_edges_offset[i]),
                 clk[1][data_edges[i] - tminus_samples:data_edges[i] + tplus_samples],
                     alpha = 1/rel_samples_fully_dark/num_samples, color=color2)
    ax1.grid()
    ax1.set_xlim([-230,200])
    ax1.set_ylim([-1,1])
    ax1.set_xlabel('Time [ps]')
    ax1.set_ylabel('Data [V]', color=color1)#, color=color)  # we already handled the x-label with ax1
    ax2.set_ylabel('Clock [V]', color=color2)#, color=color)  # we already handled the x-label with ax1
    ax1.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# plot period jitter time overlay
def plot_period_jitter(data,edges,edge_offset,title):
    tminus_samples = 100
    tplus_samples = 700
    num_samples = len(edges)
    rel_samples_fully_dark = 0.01
    for i in range(
            len(edges)):  # start from index skipcount+1=20+1, so 22st rising edge. this is 21st period, so skippin first 20
        plt.plot(1e12 * (data[0][edges[i] - tminus_samples:edges[i] + tplus_samples]
                         - data[0][edges[i]] + edge_offset[i]),
                 data[1][edges[i] - tminus_samples:edges[i] + tplus_samples],
                 alpha=1/rel_samples_fully_dark/num_samples, color='C0')
    plt.grid()
    plt.xlim([-30, 200])
    plt.ylabel("Voltage")
    plt.xlabel("Time[ps]")
    plt.title(title)
    plt.show()
def plot_settling(data, clk_edges, phase_ps, data_period, clk_period, title):
    fig, ax1 = plt.subplots()
    color1 = 'tab:red'
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'tab:blue'
    ax2.tick_params(axis='y', labelcolor=color2)
    len_fewest = min(len(clk_edges), len(data_period), len(phase_ps)) #if clock start slow, multiple data recorded before clock. otherwise, dataperiod or phaseps could be shorter, depending on which comes last
    ax1.plot(1e9*data[0][clk_edges[:len_fewest]],phase_ps[:len_fewest]/data_period[:len_fewest]*360-180,label="CLK Offset",color=color1)
    ax2.plot(1e9*data[0][clk_edges[:len_fewest]],1/data_period[:len_fewest]*1e-9,label="Data Frequency",color=color2)
    ax2.plot(1e9 * data[0][clk_edges[:-1]], 1 / clk_period * 1e-9, label="Clock Frequency", color="C2",alpha=0.5)
    ax1.grid()
    ax1.set_ylim([-180,180])
    ax1.set_yticks(np.arange(-180, 180, step=45))
    ax1.set_ylabel("Phase Offset [degrees]")
    ax2.set_ylim([5.85,6.45])
    ax2.set_ylabel("Signal Frequency [GHz]")
    ax1.set_xlabel("Time[ns]")
    ax1.set_title(title)
    ax1.legend(loc='upper center')
    ax2.legend()
    plt.show()

plot_settling(VOUT03, clk_rising_edges03b, phase_ps03b, data_period03b, clk_period03b, "Turn-on Lock Behavior without Noise")
plot_settling(VOUT04, clk_rising_edges04a, phase_ps04a, data_period04a, clk_period04a, "Turn-on Lock Behavior with Noise")
#plot_settling(VOUT02, clk_rising_edges02b, phase_ps02b, data_period02b, clk_period02b, "VCO Behavior with Noise")
plot_settling(VOUT05, clk_rising_edges05b, phase_ps05b, data_period05b, clk_period05b, "62.5MHz Step Behavior without Noise")
plot_settling(VOUT06, clk_rising_edges06b, phase_ps06b, data_period06b, clk_period06b, "250MHz Step Behavior without Noise")
plot_settling(VOUT07, clk_rising_edges07b, phase_ps07b, data_period07b, clk_period07b, "125MHz Step Behavior without Noise")

#time domain plots
if 1:
    #plot_period_jitter(VOUT02, rising_edges02, rising_edges_offset02, "Transient Simulation With Noise - Period Jitter")
    #plot_period_jitter(VOUT03,clk_rising_edges03,clk_rising_edges_offset03, "Transient Simulation With Noise - Period Jitter")
    #plot_period_jitter(VOUT04,clk_rising_edges04,clk_rising_edges_offset04, "Transient Simulation With Noise - Period Jitter")

    plot_phase_jitter(DIN03, VOUT03, din_edges03, din_edges_offset03, "Transient Simulation Without Noise - \n Phase Jitter = 0.04 ps rms, offset = 6 ps")
    plot_phase_jitter(DIN04, VOUT04, din_edges04, din_edges_offset04, "Transient Simulation With Noise - \n Phase Jitter = 13.61 ps rms, offset = 6 ps")


# plot vco period jitter histogram
if 0:
    #bins=np.histogram(np.hstack((1e12*period_jitter01,1e12*period_jitter02)), bins=40)[1] #get the bin edges
    plt.hist(1e12 * period_jitter01, label="VCO Period Jitter Without Noise", alpha=1, density=False)
    plt.hist(1e12 * period_jitter02, label="VCO Period Jitter With Noise", alpha=0.5, density=False)
    plt.ylabel("Count")
    plt.xlabel("Jitter [ps]")
    plt.title("Transient Simulation With and Without Noise")
    plt.legend()
    plt.show()

#vco vs cdr period jitter
densityplot = True
bins = np.histogram(np.hstack((1e12 * period_jitter02, 1e12 * period_jitter04,)), bins=40)[1]  # get the bin edges
plt.hist(1e12 * period_jitter02, bins, label="VCO Period Jitter With Noise = 2.55 ps rms", alpha=0.5, density=densityplot)
plt.hist(1e12 * period_jitter04, bins, label="CDR Period Jitter With Noise = 2.63 ps rms", alpha=0.5, density=densityplot)
plt.ylabel("Probability")
plt.xlabel("Jitter [ps]")
plt.title("Transient Simulation With Noise - 3000 cycles")
plt.legend()
plt.show()

#vco vs cdr phase jitter
densityplot = True
#bins = np.histogram(np.hstack((1e12 * phase_jitter02b, 1e12 * phase_jitter04,)), bins=40)[1]  # get the bin edges
plt.hist(1e12 * phase_jitter02b, label="VCO Phase Jitter With Noise = 25.13 ps rms", alpha=0.5, density=densityplot)
plt.hist(1e12 * phase_jitter04, label="CDR Phase Jitter With Noise = 13.61 ps rms", alpha=0.5, density=densityplot)
plt.ylabel("Probability")
plt.xlabel("Jitter [ps]")
plt.title("Transient Simulation With Noise - 3000 cycles")
plt.legend()
plt.show()
#cdr phase jitter without and with noise
densityplot = True
bins = np.histogram(np.hstack((1e12 * (phase_ps03 - 0.5 / 6e9), 1e12 * (phase_ps04 - 0.5 / 6e9))), bins=40)[1]  # get the bin edges
#plt.hist(1e12 * period_jitter01, bins, label="VCO Period Jitter Without Noise", alpha=0.5, density=densityplot)
#plt.hist(1e12 * period_jitter02, bins, label="VCO Period Jitter With Noise = 2.57 ps", alpha=0.5, density=densityplot)
#plt.hist(1e12 * period_jitter03, bins, label="CDR Period Jitter Without Noise", alpha=0.5, density=densityplot)
#plt.hist(1e12 * period_jitter04, bins, label="CDR Period Jitter With Noise = 2.63 ps", alpha=0.5, density=densityplot)
plt.hist(1e12 * (phase_ps03 - 0.5 / 6e9), bins, label="CDR Phase Jitter Without Noise = 0.04 ps rms and 6 ps offset", alpha=0.5, density=densityplot)
plt.hist(1e12 * (phase_ps04 - 0.5 / 6e9), bins, label="CDR Phase Jitter With Noise = 11.5 ps rms and 6 ps offset", alpha=0.5, density=densityplot)
plt.ylabel("Probability")
plt.xlabel("Jitter [ps]")
plt.title("Transient Simulation With Noise")
plt.legend()
plt.show()

#cdr period vs phase jitter with noise
densityplot = False#True
bins = np.histogram(np.hstack((1e12 * (phase_ps03 - 0.5 / 6e9), 1e12 * (phase_ps04 - 0.5 / 6e9))), bins=40)[1]  # get the bin edges
bins = np.histogram(np.hstack((1e12 * period_jitter04,1e12 * (phase_ps04 - 0.5 / 6e9))), bins=40)[1]  # get the bin edges
#plt.hist(1e12 * period_jitter01, bins, label="VCO Period Jitter Without Noise", alpha=0.5, density=densityplot)
#plt.hist(1e12 * period_jitter02, bins, label="VCO Period Jitter With Noise = 2.57 ps", alpha=0.5, density=densityplot)
#plt.hist(1e12 * period_jitter03, bins, label="CDR Period Jitter Without Noise", alpha=0.5, density=densityplot)
plt.hist(1e12 * period_jitter04, bins, label="CDR Period Jitter With Noise = 2.63 ps", alpha=0.5, density=densityplot)
#plt.hist(1e12 * (phase03-0.5/6e9), bins, label="CDR Phase Jitter Without Noise including offset = 6 ps", alpha=0.5, density=densityplot)
plt.hist(1e12 * (phase_ps04 - 0.5 / 6e9), bins, label="CDR Phase Jitter With Noise = 13.61 ps rms and 6 ps offset", alpha=0.5, density=densityplot)
plt.ylabel("Probability")
plt.xlabel("Jitter [ps]")
plt.title("Transient Simulation With Noise")
plt.legend()
plt.show()

# plot period length over time
plt.plot(period02)
plt.plot(period01)
plt.plot(period03)
plt.plot(period04)
plt.show()

plt.hist(1e12 * period_jitter03, label="CDR Period Jitter Without Noise", alpha=0.5, density=True)
plt.hist(1e12 * phase_jitter03, label="CDR Phase Jitter Without Noise", alpha=0.5, density=True)
plt.legend()
plt.show()