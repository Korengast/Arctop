from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, sosfilt, argrelextrema
import pandas as pd


def print_ft_cdf(df, channels, sampling_rate):
    N = df.shape[0]  # Number of samples
    for c in channels:
        if c not in 'Y':
            y = df[c]
            yf = fft(y)
            xf = np.linspace(0.0, 1.0 / (2.0 * sampling_rate), N // 2)
            plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
            plt.show()


def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def determine_window_size(df):
    y = list(df['Y'])
    w_sizes = []
    counter = 0
    last_val = 3
    is_new = True
    for y_val in y:
        if is_new:
            last_val = y_val
            counter += 1
            is_new = False
        else:
            if last_val == y_val:
                counter += 1
            else:
                w_sizes.append(counter)
                counter = 0
                is_new = True
    return min(w_sizes) - 2


def fourier_features(df, channels, sampling_rate):
    features = dict()
    N = df.shape[0]
    for c in channels:
        y = df[c]
        yf = fft(y)
        yft = 2.0 / N * np.abs(yf[0:N // 2])
        xf = np.linspace(0.0, 1.0 / (2.0 * sampling_rate), N // 2)
        peaks = xf[argrelextrema(yft, np.greater)]
        peaks_val = yft[argrelextrema(yft, np.greater)]
        features[c + '_#peaks'] = [peaks.shape[0]]
        for i in range(4):
            num = str(i + 1)
            try:
                features[c + '_ft_peak' + num] = [peaks[i]]
                features[c + '_ft_peak_val' + num] = [peaks_val[i]]
            except:
                features[c + '_ft_peak' + num] = [0]
                features[c + '_ft_peak_val' + num] = [0]
    return features


def reg_features(df, channels):
    features = dict()
    for c in channels:
        features[c + '_min'] = [df[c].min()]
        features[c + '_max'] = [df[c].max()]
        features[c + '_mean'] = [df[c].mean()]
        features[c + '_gap'] = [df[c].max() - df[c].min()]
    return features


def diff_features(df, channels):
    features = dict()
    diff_df = df.diff()
    for c in channels:
        features[c + '_min_diff'] = [diff_df[c].min()]
        features[c + '_max_diff'] = [diff_df[c].max()]
        features[c + '_mean_diff'] = [diff_df[c].mean()]
        features[c + '_gap_diff'] = [diff_df[c].max() - diff_df[c].min()]
    return features


def build_features(df, channels, sampling_rate):
    features = dict()
    features.update(reg_features(df, channels))
    features.update(diff_features(df, channels))
    features.update(fourier_features(df, channels, sampling_rate))
    features['Y'] = df['Y'].mode()[0]
    return features


def build_features_by_window(df, window_size, features_list, channels, sampling_rate, is_train=True):
    full_features_list = []
    for c in channels:
        for f in features_list:
            full_features_list.append(c + '_' + f)
    features_df = pd.DataFrame(columns=full_features_list)
    data_length = df.shape[0]
    i = 0
    while i < data_length - window_size + 1:
        window = df.iloc[i:i + window_size]
        if is_train:
            while len(window['Y'].unique()) != 1 and i < data_length - window_size + 1:
                i += 1
                window = df.iloc[i:i + window_size]
            row_features = pd.DataFrame(build_features(window, channels, sampling_rate))
            features_df = features_df.append(row_features)
            i += window_size
        else:
            row_features = pd.DataFrame(build_features(window, channels, sampling_rate))
            features_df = features_df.append(row_features)
            i += 1
    return features_df