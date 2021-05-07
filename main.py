# -*- coding: utf-8 -*-
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import librosa
import librosa.display

import utils
from ClassifierArray import ClassifierArray

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
            
def load_transform_mfccs(path):
    mfccs = np.load(path)
    fmfccs = []
    for m in mfccs:
        fmfccs.append(m.flatten())
    return fmfccs

def get_listens(folder):
    paths = glob.glob(folder + "\\*.wav")
    l = []
    for path in paths:
        l.append(all_listens[int(path[path.rfind('\\')+1:-4])])
    return l

#4012 songs out of 8000 with at least 2500 plays
def get_listen_categories(folder, cutoff=2500):
    paths = glob.glob(folder + "\\*.wav")
    l = []
    overcutoff = 0
    undercutoff = 0
    for path in paths:
        value = 1 if all_listens[int(path[path.rfind('\\')+1:-4])] >= cutoff else 0
        l.append(value)
        overcutoff += value
        undercutoff += 1-value
    return (l, overcutoff, undercutoff)

#median for our dataset
def get_listen_categories_MyTracks(folder, cutoff=93365877):
    paths = glob.glob(folder + "\\*.wav")
    l = []
    overcutoff = 0
    undercutoff = 0
    for path in paths:
        value = 1 if int(all_listens[path[path.rfind('\\')+1:-4]]) >= cutoff else 0
        l.append(value)
        overcutoff += value
        undercutoff += 1-value
    return (l, overcutoff, undercutoff)

def get_train_test_sets(test_interval=3,folder_ids=range(0,156),mfcc_folder='data\\fma_small_mfcc\\', audio_folder='data\\fma_small\\', cutoff=2500):
    X_trn = []
    y_trn = []
    X_test = []
    y_test = []
    positives = 0
    negatives = 0
    for folderid in folder_ids:
        foldername = "%03d" % folderid
        fmfccs = load_transform_mfccs('data\\fma_small_mfcc\\' + foldername + '.npy') 
        listens, pos, neg = get_listen_categories('data\\fma_small\\' + foldername,cutoff)
        positives += pos
        negatives += neg
        for i in range(0,len(listens)):
            if i % test_interval == 0:
                X_test.append(fmfccs[i])
                y_test.append(listens[i])
            else:
                X_trn.append(fmfccs[i])
                y_trn.append(listens[i])
                
    return (np.array(X_trn),np.array(y_trn),np.array(X_test),np.array(y_test))

def get_train_test_sets_MyTracks(test_interval=3,mfcc_folder='data\\MyTracks_mfcc\\', audio_folder='data\\MyTracks\\', cutoff=2500):
    X_trn = []
    y_trn = []
    X_test = []
    y_test = []
    fmfccs = load_transform_mfccs('data\\MyTracks_mfcc\\000.npy')
    listens, pos, neg = get_listen_categories_MyTracks('data\\MyTracks')
    for i in range(0,len(listens)):
            if i % test_interval == 0:
                X_test.append(fmfccs[i])
                y_test.append(listens[i])
            else:
                X_trn.append(fmfccs[i])
                y_trn.append(listens[i])
    return (np.array(X_trn),np.array(y_trn),np.array(X_test),np.array(y_test))
    
def train(model, X_trn_list, y_trn_list, X_trn, y_trn):
    model.fit(X_trn, X_trn_list,y_trn_list)
    return test(model,X_trn,y_trn)
def train_MLP(model, X_trn, y_trn):
    model.fit(X_trn, y_trn)
    return test(model,X_trn,y_trn)

def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return skl.metrics.confusion_matrix(y_test, y_pred)

def avg_confusion(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    acc = (tn+tp)/(tn+fp+fn+tp)
    return acc
def print_confusion(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    print("true positives: " + str(tp) + ", true negatives: " + str(tn))
    print("false positives: " + str(fp) + ", false negatives: " + str(fn))
    acc = (tn+tp)/(tn+fp+fn+tp)
    print("accuracy: " + ("%0.4f" % acc))
    return acc

def save_model(model, path):
    dump(model, path)

def create_test(iteration_min, iteration_max, iteration_step, save_path, classifier_count=20, hidden_layer_sizes = (40,20), director_sizes=(40,40)):
    accs_trn = []
    accs_test = []
    confusions = []
    classifier = ClassifierArray(director_hidden_sizes = director_sizes, hidden_sizes = hidden_layer_sizes, random=1, count=classifier_count, max_iterations=iteration_step, warm=True)
    X_trn,y_trn,X_test,y_test = get_train_test_sets(folder_ids=range(0,156))
    X_trn_list = []
    y_trn_list = []
    length = len(y_trn)
    for i in range(classifier_count):
        X_trn_list.append(X_trn[int(i*length/classifier_count):int((i+1)*length/classifier_count)])
        y_trn_list.append(y_trn[int(i*length/classifier_count):int((i+1)*length/classifier_count)])
    
    for it in range(iteration_min,iteration_max,iteration_step):
        #print("training with max iter=%d" % it)
        acc_trn = avg_confusion(train(classifier,X_trn_list,y_trn_list, X_trn, y_trn))
        print("testing with max iter=%d (training acc: %.2f)" % (it, acc_trn))
        conf = test(classifier,X_test,y_test)
        confusions.append(conf)
        acc_test = avg_confusion(conf)
        accs_trn.append(acc_trn)
        accs_test.append(acc_test)
        print()
        #save_model(classifier, save_path + str(hidden_sizes) + str(it) + ' epochs.joblib')
    return (classifier, accs_trn,accs_test, confusions)

def create_test_MyTracks(iteration_min, iteration_max, iteration_step, save_path, hidden_sizes = (40,20)):
    accs_trn = []
    accs_test = []
    confusions = []
    classifier = MLPClassifier(hidden_layer_sizes = hidden_sizes, random_state=1, max_iter=iteration_step, warm_start=True)
    X_trn,y_trn,X_test,y_test = get_train_test_sets_MyTracks()
    
    for it in range(iteration_min,iteration_max,iteration_step):
        #print("training with max iter=%d" % it)
        acc_trn = avg_confusion(train_MLP(classifier, X_trn, y_trn))
        print("testing with max iter=%d (training acc: %.2f)" % (it, acc_trn))
        conf = test(classifier,X_test,y_test)
        confusions.append(conf)
        acc_test = avg_confusion(conf)
        accs_trn.append(acc_trn)
        accs_test.append(acc_test)
        print()
        #save_model(classifier, save_path + str(hidden_sizes) + str(it) + ' epochs.joblib')
    return (classifier, accs_trn,accs_test, confusions)


def load_MyTracks(path_to_csv='data/MyTracks.csv'):
    tracks = pd.read_csv(path_to_csv, index_col=0, usecols=[0,1])
    return tracks['Plays']

#all_tracks = utils.load('data/fma_metadata/tracks.csv')
#all_listens = all_tracks[all_tracks['set', 'subset'] <= 'small'][('track', 'listens')]
all_listens = load_MyTracks()

it_min =    5
it_max=     100
it_step =   5

train_epochs = range(it_min, it_max, it_step)
model, accs_trn, accs_test, confusions = create_test_MyTracks(it_min,it_max,it_step, 'data\\models\\')

print()
print('done')
print()
max_test_i = np.argmax(accs_test)
print('maximum test accurracy: %.3f with %d epochs' % (accs_test[max_test_i], train_epochs[max_test_i]))
print_confusion(confusions[max_test_i])
plt.show()
plt.figure()
plt.xlim([0,it_max])
plt.ylim([0,1])

plt.plot(train_epochs, accs_trn, 'bo-', linewidth=2, label='train accuracy')
plt.plot(train_epochs, accs_test, 'go-', linewidth=2, label='test accuracy')

plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('Training epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)


#save_model(regressor, 'data\\MLPregressor.joblib')
#print('model saved!')
#loaded = load('data\\MLPregressor.joblib')

#print(test_folder(loaded,1))


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


