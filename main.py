# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:11:48 2021

@author: moore
"""

import os
import glob

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import utils
import json




#if the song is shorter than the target_duration in seconds, it will pad with zeros
def path_to_mfcc(filename, target_duration=31):

    x, sr = librosa.load(filename, sr=22050, mono=True)
    target_samples = target_duration*sr
    curr_samples = x.shape[-1]
    if target_samples > curr_samples:
        x = np.concatenate((x,np.zeros(target_samples-curr_samples)))
    else:
        print("id: " + str(id) + " has more samples than the target (" + str(curr_samples) + " > " + str(target_samples) + "), cutting...")
        x = x[0:target_samples]

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)  
    return mfcc

def get_mfccs(folder):
    mfccs = []
    paths = glob.glob(folder + "\\*.wav")
    total = len(paths)
    print("Calculating MFCCS for " + str(total) + " files...")
    done = 0    
    for path in paths:
        mfccs.append(path_to_mfcc(path))
        done += 1
        print(str(100*done/total) + "% done")
    return np.array(mfccs)

def save_mfccs(mfccs, save_path):
    np.save(save_path,mfccs)

def calc_and_save_mfccs(save_path, audio_dir = 'data\\fma_small\\'):
    paths = glob.glob(audio_dir + "\\*")
    for path in paths:
        mfcc = get_mfccs(path)
        if len(mfcc) > 0:
            folder = path[path.rfind('\\'):]
            save_mfccs(mfcc, save_path + folder)
            print("Finished folder " + folder)
            

calc_and_save_mfccs('data\\fma_small_mfcc\\')
# ## 4 Audio
# 
# You can load the waveform and listen to audio in the notebook itself.

# In[ ]:


#filename = utils.get_audio_path(AUDIO_DIR, 2)
#print('File: {}'.format(filename))

#x, sr = librosa.load(filename, sr=22050, mono=True)
#print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

#start, end = 7, 17
#ipd.Audio(data=x[start*sr:end*sr], rate=sr)


# And use [librosa](https://github.com/librosa/librosa) to compute spectrograms and audio features.

# In[ ]:


#librosa.display.waveplot(x, sr, alpha=0.5);
#plt.show()
#plt.vlines([start, end], -1, 1)

#start = len(x) // 2
#plt.figure()
#plt.plot(x[start:start+2000])
#plt.ylim((-1, 1));


# In[ ]:


#stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
#mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
#log_mel = librosa.logamplitude(mel)
#log_mel = librosa.amplitude_to_db(mel) #amplitude_to_db replaced logamplitude

#librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel');
#plt.show()


# In[ ]:


#mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
#mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
#print(len(mfcc))
#print(len(mfcc[0]))
#librosa.display.specshow(mfcc, sr=sr, x_axis='time');
#plt.show()


