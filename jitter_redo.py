import numpy as np
import matplotlib.pyplot as plt
import csv
import time


# csv filename, sets is number of signal paired with time data. time data is the same though from cadence
def get_data(filename, sets=1):
    data = []
    for i in range(sets):
        data.append([])
        data.append([])
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            for i in range(sets):
                data[2 * i].append(float(row[2 * i]))
                data[2 * i + 1].append(float(row[2 * i + 1]))
    return np.asarray(data)


def lin_interpolate(time1, time2, d1, d2, threshold):
    proportion_above = (d2 - threshold) / (d2 - d1)
    time_cross = time2 - proportion_above * (time2 - time1)
    return time_cross


def get_edges_period(time, signal, threshold, tstart=0.0, tstop=99.9):
    """
    :param np.array signal: time domain voltage or current signal
    :param np.array time: time stamps for signal values
    :param float threshold: value for when a new cycle is started
    :param float tstart: time when to start using values from the data
    :param float tstop: time when to stop using values from the data
    return edges with index stamp and seconds, period with time stamp and value
    """
    rising_edges = [[], []]
    period = [[], []]
    for i in range(1, (len(time))):
        if tstart < time[i] < tstop:
            if signal[i - 1] < threshold < signal[i]:
                rising_edges[0].append(lin_interpolate(time[i], time[i - 1], signal[i], signal[i - 1], threshold))
                rising_edges[1].append(i)
                if len(rising_edges[1]) > 1:
                    period[0].append(time[i])
                    period[1].append(rising_edges[1][-1] - rising_edges[1][-2])
    rising_edges = np.asarray(rising_edges)
    period = np.asarray(period)
    return rising_edges, period


def get_edges_period_phase(time, clk, din, threshold_clk, threshold_din, tstart=0.0, tstop=9.9):
    """
    return clk edges with time stamp and index
    return din edges with time stamp and index
    return period with time stamp and value
    return phase with time stamp and value
    """
    clk_rising_edges = [[], []]
    din_edges = [[], []]
    data_period = [[], []]
    clk_period = [[], []]
    phase = [[], []]

    for i in range(1, (len(time))):
        if tstart < time[i] < tstop:
            if clk[i - 1] < threshold_clk < clk[i]:
                clk_rising_edges[0].append(lin_interpolate(time[i], time[i - 1], clk[i], clk[i - 1], threshold_clk))
                clk_rising_edges[1].append(i)
                if len(din_edges[0]) > 0:
                    phase[0].append(time[i])
                    phase[1].append(clk_rising_edges[0][-1] - din_edges[0][-1])
                if len(clk_rising_edges[0]) > 1:
                    clk_period[0].append(time[i])
                    clk_period[1].append((clk_rising_edges[0][-1] - clk_rising_edges[0][-2]))
            if din[i - 1] < threshold_din < din[i] or din[i - 1] > threshold_din > din[i]:
                din_edges[0].append(lin_interpolate(time[i], time[i - 1], din[i], din[i - 1], threshold_din))
                din_edges[1].append(i)
                if len(din_edges[0]) > 1:
                    data_period[0].append(time[i])
                    data_period[1].append(din_edges[1][-1] - din_edges[1][-2])

    clk_rising_edges = np.asarray(clk_rising_edges, dtype=object)
    clk_rising_edges[1,:]=clk_rising_edges[1].astype(int)
    din_edges = np.asarray(din_edges)
    clk_period = np.asarray(clk_period)
    data_period = np.asarray(data_period)
    phase = np.asarray(phase)
    return clk_rising_edges, din_edges, clk_period, data_period, phase


def jitter(period, phase: float = 0):
    mean_period = np.mean(period)
    period_jitter = period - mean_period
    rms_period_jitter = np.std(period_jitter)
    mean_phase = np.mean(phase)
    phase_jitter = phase - mean_phase
    rms_phase_jitter = np.std(phase_jitter)
    return mean_period, period_jitter, rms_period_jitter, mean_phase, phase_jitter, rms_phase_jitter


clk_threshold = 0.55
din_threshold = 0

tstart01 = 5e-9  # get vco histogram after settling
VOUT01 = get_data("VOUT_575mV_noNoise.csv")
rising_edges01, period01 = get_edges_period(VOUT01[0], VOUT01[1], clk_threshold, tstart=tstart01)
mean_period01, period_jitter01, rms_period_jitter01 = jitter(period01[1])[:3]
print(rising_edges01)
print("periods = " + str(len(period01[0])))
print("mean period = " + str(mean_period01))
print("rms period jitter = " + str(rms_period_jitter01))

tstart02 = 100e-9  # 5e-9 #100e-9 get histogram after settling for 600ns file
VOUT02 = get_data("VCO_VOUT_noisy_600ns.csv")  # VOUT02 = get_data("VOUT_575mV_100GHZnoise.csv")
rising_edges02, period02 = get_edges_period(VOUT02[0], VOUT02[1], clk_threshold,
                                            tstart=tstart02)  # for mean_period to create dummy din data
mean_period02 = np.mean(period02)
DIN02 = np.zeros((2, np.size(VOUT02[0])))
DIN02[0] = np.linspace(0, 600e-9, np.size(VOUT02[0]))
DIN02[1] = np.sin(np.pi / mean_period02 * DIN02[0])  # no 2 in 2pi since data is half the hz
clk_rising_edges02, din_edges02, clk_period02, data_period02, phase02 = get_edges_period_phase(
    VOUT02[0], VOUT02[1], DIN02[1], clk_threshold, din_threshold, tstart=tstart02)
mean_period02, period_jitter02, rms_period_jitter02, mean_phase02, phase_jitter02, rms_phase_jitter02 \
    = jitter(clk_period02[1], phase02[1])
print("periods = " + str(len(clk_period02[0])))
print("mean period = " + str(mean_period02))
print("rms period jitter = " + str(rms_period_jitter02))
print("sim mean phase = " + str(mean_phase02))
print("sim rms phase jitter = " + str(rms_phase_jitter02))

# tstart03 = 120e-9
# tstop03 = 300e-9
# VOUT03 = get_data("VCLK_noiseless_120ns_to_300ns_stable.csv") #random step
# DIN03 = get_data("DIN_noiseless_120ns_to_300ns_stable.csv")
# 03 data includes startup. 03b is only after stabilizes
tstart03 = 0
tstop03 = 200e-9
tstart03b = 120e-9
tstop03b = 200e-9
VOUT03 = get_data("VCLK_noiseless_200ns_lock.csv")
DIN03 = get_data("DIN_noiseless_200ns_lock.csv")
clk_rising_edges03, din_edges03, clk_period03, data_period03, phase03 = get_edges_period_phase(
    VOUT03[0], VOUT03[1], DIN03[1], clk_threshold, din_threshold, tstart=tstart03, tstop=tstop03)
mean_period03, period_jitter03, rms_period_jitter03, mean_phase03, phase_jitter03, rms_phase_jitter03 \
    = jitter(clk_period03[1], phase03[1])

clk_rising_edges03b, din_edges03b, clk_period03b, data_period03b, phase03b = get_edges_period_phase(
    VOUT03[0], VOUT03[1], DIN03[1], clk_threshold, din_threshold, tstart=tstart03b, tstop=tstop03b)
mean_period03b, period_jitter03b, rms_period_jitter03b, mean_phase03b, phase_jitter03b, rms_phase_jitter03b \
    = jitter(clk_period03b[1], phase03b[1])
print("periods = " + str(len(clk_period03b[0])))
print("mean period = " + str(mean_period03b))
print("rms period jitter = " + str(rms_period_jitter03b))
print("sim mean phase = " + str(mean_phase03b))
print("sim rms phase jitter = " + str(rms_phase_jitter03b))

# CDR noisy for transient lock plot
tstart04t = 0
tstop04t = 200e-9
data04t = get_data("data_noisy_140ns_to_300ns_stable.csv", 3)
VOUT04t = data04t[0:2]
DIN04t = data04t[2:4]
clk_rising_edges04t, din_edges04t, clk_period04t, data_period04t, phase04t = get_edges_period_phase(
    VOUT04t[0], VOUT04t[1], DIN04t[1], clk_threshold, din_threshold, tstart=tstart04t, tstop=tstop04t)
print(clk_rising_edges04t)
# CDR noisy for histogram jitter
tstart04h = 100e-9  # tstart04 = 140e-9
tstop04h = 600e-9  # tstop04 = 300e-9
data04h = get_data("data_noisy_600ns.csv", 3)
VOUT04h = data04h[0:2]  # [4:]
DIN04h = data04h[2:4]
clk_rising_edges04h, din_edges04h, clk_period04h, data_period04h, phase04h = get_edges_period_phase(
    VOUT04h[0], VOUT04h[1], DIN04h[1], clk_threshold, din_threshold, tstart=tstart04h, tstop=tstop04h)
mean_period04h, period_jitter04h, rms_period_jitter04h, mean_phase04h, phase_jitter04h, rms_phase_jitter04h \
    = jitter(clk_period04h[1], phase04h[1])
print("periods = " + str(len(clk_period04h[0])))
print(clk_period04h[1])
print("mean period = " + str(mean_period04h))
print("rms period jitter = " + str(rms_period_jitter04h))
print("sim mean phase = " + str(mean_phase04h))
print("sim rms phase jitter = " + str(rms_phase_jitter04h))

# for step 62.5MHz
tstart05 = 75e-9
tstop05 = 250e-9
data05 = get_data("data_noiseless_300ns_62p5MHz_step_both_directions.csv", 3)
VOUT05 = data05[0:2]
DIN05 = data05[2:4]
clk_rising_edges05, din_edges05, clk_period05, data_period05, phase05 = get_edges_period_phase(
    VOUT05[0], VOUT05[1], DIN05[1], clk_threshold, din_threshold, tstart=tstart05, tstop=tstop05)

# for step 125MHz
tstart06 = 75e-9
tstop06 = 250e-9
data06 = get_data("data_noiseless_300ns_125MHz_step_both_directions.csv", 3)
VOUT06 = data06[0:2]
DIN06 = data06[2:4]
clk_rising_edges06, din_edges06, clk_period06, data_period06, phase06 = get_edges_period_phase(
    VOUT06[0], VOUT06[1], DIN06[1], clk_threshold, din_threshold, tstart=tstart06, tstop=tstop06)

# for step 250MHz
tstart07 = 75e-9
tstop07 = 250e-9
data07 = get_data("data_noiseless_300ns_250MHz_step_both_directions.csv", 3)
VOUT07 = data07[0:2]
DIN07 = data07[2:4]
clk_rising_edges07, din_edges07, clk_period07, data_period07, phase07 = get_edges_period_phase(
    VOUT07[0], VOUT07[1], DIN07[1], clk_threshold, din_threshold, tstart=tstart07, tstop=tstop07)


def plot_phase_jitter(time, clk, din, data_edges, tbefore, tafter, title):  # plot phase jitter time overlay
    tbefore_index = np.searchsorted(time, data_edges[0] - tbefore)
    tafter_index = np.searchsorted(time, data_edges[0] + tafter)
    # tminus_samples = 600
    # tplus_samples = 700
    fig, ax1 = plt.subplots()

    color1 = 'tab:red'
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'tab:blue'
    ax2.tick_params(axis='y', labelcolor=color2)
    num_samples = len(data_edges[0]) - 1  # in case no clock after last edge
    rel_samples_fully_dark = 0.02  # 1 means need 100% of trace to overlap to be fully dark. 0.01 means 1% needed to
    # be fully dark
    for i in range(num_samples):
        ax1.plot(1e12 * (time[tbefore_index[i]:tafter_index[i]] - data_edges[0][i]),
                 din[tbefore_index[i]:tafter_index[i]],
                 alpha=1 / rel_samples_fully_dark / num_samples, color=color1)
        # clock data from the associated din rising edge
        ax2.plot(1e12 * (time[tbefore_index[i]:tafter_index[i]] - data_edges[0][i]),
                 clk[tbefore_index[i]:tafter_index[i]],
                 alpha=1 / rel_samples_fully_dark / num_samples, color=color2)
    ax1.grid()
    ax1.set_xlim([-tbefore * 1e12, tafter * 1e12])
    ax1.set_ylim([-1, 1])
    ax1.set_xlabel('Time [ps]')
    ax1.set_ylabel('Data [V]', color=color1)  # , color=color)  # we already handled the x-label with ax1
    ax2.set_ylabel('Clock [V]', color=color2)  # , color=color)  # we already handled the x-label with ax1
    ax1.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# plot period jitter time overlay
def plot_period_jitter(data, edges, tbefore, tafter, title):
    tbefore_index = np.searchsorted(data[0], edges[0] - tbefore)
    tafter_index = np.searchsorted(data[0], edges[0] + tafter)
    num_samples = len(edges[0])
    rel_samples_fully_dark = 0.01
    for i in range(
            len(edges[
                    0])):  # start from index skipcount+1=20+1, so 22st rising edge. this is 21st period, so skippin first 20
        plt.plot(1e12 * (data[0][tbefore_index[i]:tafter_index[i]] - edges[0][i]),
                 data[1][tbefore_index[i]:tafter_index[i]],
                 alpha=1 / rel_samples_fully_dark / num_samples, color='C0')
    plt.grid()
    plt.ylabel("Voltage")
    plt.xlabel("Time[ps]")
    plt.title(title)
    plt.show()


def plot_settling(data, clk_edges, phase, data_period, clk_period, title):
    fig, ax1 = plt.subplots()
    color1 = 'tab:red'
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.tick_params(axis='y', labelcolor=color2)

    len_fewest = min(len(clk_edges[1]), len(data_period[1]), len(phase[1])) #if clock start slow, multiple data recorded before clock. otherwise, dataperiod or phaseps could be shorter, depending on which comes last
    print(type(clk_edges[1][:len_fewest]))
    print(type(clk_edges[1][:len_fewest][0]))
    print(data[0][clk_edges[1][:len_fewest]])
    ax1.plot(1e9*data[0][clk_edges[1][:len_fewest]],phase[1][:len_fewest]/data_period[1][:len_fewest]*360-180,label="CLK Offset",color=color1)
    ax2.plot(1e9*data[0][clk_edges[1][:len_fewest]],1/data_period[1][:len_fewest]*1e-9,label="Data Frequency",color=color2)
    ax2.plot(1e9 * data[0][clk_edges[1][:-1]], 1 / clk_period[1] * 1e-9, label="Clock Frequency", color="C2",alpha=0.5)
    ax1.grid()
    ax1.set_ylim([-180,180])
    ax1.set_yticks(np.arange(-180, 180, step=45))
    ax1.set_ylabel("Phase Offset [degrees]")
    #ax2.set_ylim([5.85,6.45])
    ax2.set_ylabel("Signal Frequency [GHz]")
    ax1.set_xlabel("Time[ns]")
    ax1.set_title(title)
    ax1.legend(loc='upper center')
    ax2.legend()
    plt.show()

# time domain plots
if 1:
    # plot_period_jitter(VOUT02, rising_edges02, rising_edges_offset02, "Transient Simulation With Noise - Period Jitter")
    # plot_period_jitter(VOUT03,clk_rising_edges03,clk_rising_edges_offset03, "Transient Simulation With Noise - Period Jitter")
    # plot_period_jitter(VOUT04,clk_rising_edges04,clk_rising_edges_offset04, "Transient Simulation With Noise - Period Jitter")

    plot_period_jitter(VOUT04h, clk_rising_edges04h, 30e-12, 200e-12,
                       "Transient Simulation With Noise - \n Period Jitter = 2.63 ps rms, offset = 6 ps")
    plot_phase_jitter(VOUT03[0], VOUT03[1], DIN03[1], din_edges03b, 30e-12, 200e-12,
                      "Transient Simulation Without Noise - \n Phase Jitter = 0.04 ps rms, offset = 6 ps")
    plot_phase_jitter(VOUT04h[0], VOUT04h[1], DIN04h[1], din_edges04h, 30e-12, 200e-12,
                      "Transient Simulation With Noise - \n Phase Jitter = 13.61 ps rms, offset = 6 ps")
    t0 = time.time()
    plot_settling(VOUT04t, clk_rising_edges04t, phase04t, data_period04t, clk_period04t,
                      "Transient Simulation With Noise - \n Phase Jitter = 13.61 ps rms, offset = 6 ps")
