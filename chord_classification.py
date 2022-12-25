#import package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import IPython

# importing packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


path = r"C:\Users\Davon\Desktop\University\Semester 3\Lab on Python for Data Science and Machine Learning\Final Project\audio"
#label frequency with correspnding note
curr_freq = 55
freq_list = []

# I want to calculate 8 octaves of notes. Each octave has 12 notes. Looping for 96 steps:
for i in range(96): 
    freq_list.append(curr_freq)
    curr_freq *= np.power(2, 1/12) # Multiplying by 2^(1/12)

#reshaping and creating dataframe
freq_array = np.reshape(np.round(freq_list,1), (8, 12))
cols = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
df_note_freqs = pd.DataFrame(freq_array, columns=cols)
# print("NOTE FREQUENCIES IN WESTERN MUSIC")
# print(df_note_freqs.head(10))

def find_harmonics(path):
    fs, X = wavfile.read(path)
    N = len(X)
    X_F = fft(X)
    X_F_onesided = 2.0/N * np.abs(X_F[0:N//2])
    freqs = fftfreq(N, 1/fs)[:N//2]
    freqs_50_index = np.abs(freqs - 50).argmin()
    
    h = X_F_onesided.max()*5/100
    peaks, _ = find_peaks(X_F_onesided, distance=10, height = h)
    peaks = peaks[peaks>freqs_50_index]
    harmonics = np.round(freqs[peaks],2)
    
    return harmonics

# Another example to check if method is working correctly
data = []
max_harm_length = 0 # i will keep track of max harmonic length for naming columns



from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

path = "/kaggle/input/musical-instrument-chord-classification/Audio_Files"
data = []
max_harm_length = 0 # i will keep track of max harmonic length for naming columns

filename = r"C:\Users\Davon\Desktop\University\Semester 3\Lab on Python for Data Science and Machine Learning\Final Project\audio\Major_2.wav"

freq_peaks = find_harmonics(filename)

max_harm_length = max(max_harm_length, len(freq_peaks))

chordType = filename[filename.index('_')-5:filename.index('_')]
cur_data = [chordType, filename]
cur_data.extend([freq_peaks.min(), freq_peaks.max(), len(freq_peaks)])
cur_data.extend(freq_peaks)

data.append(cur_data)
# Column Names for DataFrame:
cols = ["Chord Type", "File Name", "Min Harmonic", "Max Harmonic", "# of Harmonics"]
for i in range(max_harm_length):
    cols.append("Harmonic {}".format(i+1))

# Creating DataFrame
df = pd.DataFrame(data, columns=cols)

df_original = df.copy()
df["Interval 1"] = df["Harmonic 2"].div(df["Harmonic 1"], axis=0)

df = df_original.copy() # refreshing df

for i in range(1,21):
    curr_interval = "Interval {}".format(i)
    curr_harm = "Harmonic {}".format(i+1)
    prev_harm = "Harmonic {}".format(i)
    df[curr_interval] = df[curr_harm].div(df[prev_harm], axis=0)

for i in range(2,14):
    curr_interval = "Interval {}_1".format(i)
    curr_harm = "Harmonic {}".format(i)
    df[curr_interval] = df[curr_harm].div(df["Harmonic 1"], axis=0)

print("Our dataset: ")
print(df.head())

df["Chord Type"] = df["Chord Type"].replace("Major", 1)
df["Chord Type"] = df["Chord Type"].replace("Minor", 0)

columns = ["Interval 1", "Interval 2", "Interval 3", "Interval 4"]
columns.extend(["Interval 4_1", "Interval 5_1", "Interval 6_1"])

print(df[columns])
train_X, val_X, train_y, val_y = train_test_split(df[columns], df["Chord Type"], test_size=0.40, random_state=0)
train_X.head()

# rf_classifier = joblib.load('/kaggle/input/pre-trained-models/random_forest.joblib')
# kn_classifier = joblib.load('/kaggle/input/pre-trained-models/k_neighbours.joblib')
# dt_classifier = joblib.load('/kaggle/input/pre-trained-models/decision_tree.joblib')