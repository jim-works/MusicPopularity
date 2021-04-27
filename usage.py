#!/usr/bin/env python
# coding: utf-8

# # [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)
# 
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.
# 
# ## Usage
# 
# 1. Go through the [paper] to understand what the data is about.
# 1. Download some datasets from <https://github.com/mdeff/fma>.
# 1. Uncompress the archives, e.g. with `unzip fma_small.zip`.
# 1. Load and play with the data in this notebook.
# 
# [paper]: https://arxiv.org/abs/1612.01840

# In[1]:


import os

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

plt.rcParams['figure.figsize'] = (17, 5)


# In[ ]:


# Directory where mp3 are stored.
AUDIO_DIR = 'data/fma_small/'

# Load metadata and features.
tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')
features = utils.load('data/fma_metadata/features.csv')
echonest = utils.load('data/fma_metadata/echonest.csv')

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, genres.shape, features.shape, echonest.shape


# ## 1 Metadata
# 
# The metadata table, a CSV file in the `fma_metadata.zip` archive, is composed of many colums:
# 1. The index is the ID of the song, taken from the website, used as the name of the audio file.
# 2. Per-track, per-album and per-artist metadata from the Free Music Archive website.
# 3. Two columns to indicate the subset (small, medium, large) and the split (training, validation, test).

# In[ ]:


ipd.display(tracks['track'].head())
ipd.display(tracks['album'].head())
ipd.display(tracks['artist'].head())
ipd.display(tracks['set'].head())


# ### 1.1 Subsets
# 
# The small and medium subsets can be selected with the below code.

# In[ ]:


small = tracks[tracks['set', 'subset'] <= 'small']
small.shape


# In[ ]:


medium = tracks[tracks['set', 'subset'] <= 'medium']
medium.shape


# ## 2 Genres
# 
# The genre hierarchy is stored in `genres.csv` and distributed in `fma_metadata.zip`.

# In[ ]:


print('{} top-level genres'.format(len(genres['top_level'].unique())))
genres.loc[genres['top_level'].unique()].sort_values('#tracks', ascending=False)


# In[ ]:


genres.sort_values('#tracks').head(10)


# ## 3 Features
# 
# 1. Features extracted from the audio for all tracks.
# 2. For some tracks, data colected from the [Echonest](http://the.echonest.com/) API.

# In[ ]:


print('{1} features for {0} tracks'.format(*features.shape))
columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast']
columns.append(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'])
columns.append(['rmse', 'zcr'])
for column in columns:
    ipd.display(features[column].head().style.format('{:.2f}'))


# ### 3.1 Echonest features

# In[ ]:


print('{1} features for {0} tracks'.format(*echonest.shape))
ipd.display(echonest['echonest', 'metadata'].head())
ipd.display(echonest['echonest', 'audio_features'].head())
ipd.display(echonest['echonest', 'social_features'].head())
ipd.display(echonest['echonest', 'ranks'].head())


# In[ ]:


ipd.display(echonest['echonest', 'temporal_features'].head())
x = echonest.loc[2, ('echonest', 'temporal_features')]
plt.plot(x);


# ### 3.2 Features like MFCCs are discriminant

# In[ ]:


small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
genre2 = tracks['track', 'genre_top'] == 'Hip-Hop'

X = features.loc[small & (genre1 | genre2), 'mfcc']
X = skl.decomposition.PCA(n_components=2).fit_transform(X)

y = tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]
y = skl.preprocessing.LabelEncoder().fit_transform(y)

plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.5)
X.shape, y.shape


# ## 4 Audio
# 
# You can load the waveform and listen to audio in the notebook itself.

# In[ ]:


filename = utils.get_audio_path(AUDIO_DIR, 2)
print('File: {}'.format(filename))

x, sr = librosa.load(filename, sr=None, mono=True)
print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

start, end = 7, 17
ipd.Audio(data=x[start*sr:end*sr], rate=sr)


# And use [librosa](https://github.com/librosa/librosa) to compute spectrograms and audio features.

# In[ ]:


librosa.display.waveplot(x, sr, alpha=0.5);
plt.vlines([start, end], -1, 1)

start = len(x) // 2
plt.figure()
plt.plot(x[start:start+2000])
plt.ylim((-1, 1));


# In[ ]:


stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
#log_mel = librosa.logamplitude(mel)
log_mel = librosa.amplitude_to_db(mel) #amplitude_to_db replaced logamplitude

librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel');


# In[ ]:


mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
mfcc = skl.preprocessing.StandardScaler().fit_transform(mfcc)
librosa.display.specshow(mfcc, sr=sr, x_axis='time');


# ## 5 Genre classification

# ### 5.1 From features

# In[ ]:


small = tracks['set', 'subset'] <= 'small'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
X_train = features.loc[small & train, 'mfcc']
X_test = features.loc[small & test, 'mfcc']

print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))


# In[ ]:


# Be sure training samples are shuffled.
X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

# Standardize features by removing the mean and scaling to unit variance.
scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)

# Support vector classification.
clf = skl.svm.SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('Accuracy: {:.2%}'.format(score))


# ### 5.2 From audio
