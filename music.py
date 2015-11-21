# Classifying music based on genre
# Author: Kaushik Balakrishnan, PhD
# Email: kaushikb258@gmail.com
# Based on GTZAN dataset
# Logistic Regression
# Classify based on FFT
# Also classify based on CEPS
# Mel Frequency Cepstral Coefficients (MFCC)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import random
from matplotlib.pyplot import specgram
import scipy.io
from scipy.io import wavfile
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
import os
import glob
from scikits.talkbox.features import mfcc


base_dir = '/home/kaushik/Canopy/kaushik_py/music_wav/genres/'
subdirs = ['classical', 'country', 'jazz', 'metal', 'pop', 'rock']

# Randomly read 18 files and create specgram
for i in range(0,18):
  j = random.randint(0,5)
  k = random.randint(0,99)  
  filename = base_dir + subdirs[j] + '/' + subdirs[j] + '.' "%05d" % k + '.au.wav'
  sample_rate, X = wavfile.read(filename)
  print sample_rate, X.shape
  plt.subplot(6,3,i)
  gg = subdirs[j] + '.' "%05d" % k
  plt.title(gg)
  specgram(X, Fs=sample_rate, xextent=(0,30))
  plt.show()
  plt.savefig('/home/kaushik/Canopy/kaushik_py/music_wav/music_specgram.pdf')


# First time you run, set this to =1
# For subsequent runs, set this to =0
is_it_first_time = 0


# Create FFT
def create_fft(fn):
   sample_rate, X = wavfile.read(fn)
   fft_features = abs(scipy.fft(X)[:1000])
   base_fn, ext = os.path.splitext(fn)
   data_fn = base_fn + ".fft"
   np.save(data_fn, fft_features)


if (is_it_first_time == 1):
 for i in range(0,6):
  for j in range(0,100):
    filename = base_dir + subdirs[i] + '/' + subdirs[i] + '.' "%05d" % j + '.au.wav'
    create_fft(filename)

# Read FFT
def read_fft(genre_list):
   X = []
   y = []
   for label, genre in enumerate(genre_list):
     genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
     file_list = glob.glob(genre_dir)
     for fn in file_list:
       fft_features = np.load(fn)
       X.append(fft_features[:1000])
       y.append(label)
   return np.array(X), np.array(y)    


X, y = read_fft(subdirs)


# Separate into training and test sets
def train_and_test(X,y):
   x_train = []
   y_train = []
   x_test = []
   y_test = []
   for i in range(0,X.shape[0]):
       r = np.random.rand()
       if(r<0.7):
           x_train.append(X[i,:])
           y_train.append(y[i])
       else:
           x_test.append(X[i,:])
           y_test.append(y[i])     
   return x_train, y_train, x_test, y_test
   
   
x_train, y_train, x_test, y_test = train_and_test(X,y)    


# Apply Logistic Regression on the training data
model = LogisticRegression()
model.fit(x_train, y_train)
print 'model: ', model

# Make predictions on test set
predicted = model.predict(x_test)
cm = confusion_matrix(y_test, predicted)
print 'cm: ', cm


# Confusion matrix
def plot_confusion_matrix(cm, genre_list, name, title, filename):
  pylab.clf()
  pylab.matshow(cm, fignum=False)
  ax = pylab.axes()
  ax.set_xticks(range(len(genre_list)))
  ax.set_xticklabels(genre_list)
  ax.xaxis.set_ticks_position("bottom")
  ax.set_yticks(range(len(genre_list)))
  ax.set_yticklabels(genre_list)
  pylab.title(title)
  pylab.colorbar()
  pylab.grid(False)
  pylab.xlabel('Predicted class')
  pylab.ylabel('True class')
  pylab.grid(False)
  pylab.show()
  filename = '/home/kaushik/Canopy/kaushik_py/music_wav/' + filename
  plt.savefig(filename) 


genre_list = subdirs
name = 0
title = 'Confusion matrix of an FFT based classifier' 
filename = 'music_cm.pdf'
plot_confusion_matrix(cm, genre_list, name, title, filename)


# Write ceps
def write_ceps(ceps, fn):
  base_fn, ext = os.path.splitext(fn)
  data_fn = base_fn + ".ceps"
  np.save(data_fn, ceps)
  print "written %s" % data_fn

# MFCC
def create_ceps(fn):
  sample_rate, X = wavfile.read(fn)
  ceps, mspec, spec = mfcc(X)
  write_ceps(ceps, fn)

# Read ceps
def read_ceps(genre_list):
  X, y = [], []
  for label, genre in enumerate(genre_list): 
    for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
        ceps = np.load(fn)
        num_ceps = len(ceps)
        X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))   
        y.append(label)
  return np.array(X), np.array(y)
 
 
if (is_it_first_time == 1):
 for i in range(0,6):
  for j in range(0,100):
    filename = base_dir + subdirs[i] + '/' + subdirs[i] + '.' "%05d" % j + '.au.wav'
    create_ceps(filename)
 
X, y = read_ceps(subdirs)

x_train, y_train, x_test, y_test = train_and_test(X,y)

# Apply Logistic Regression on training set (CEPS data)
model = LogisticRegression()
model.fit(x_train, y_train)
print 'model: ', model

# Make predictions on test set
predicted = model.predict(x_test)
cm = confusion_matrix(y_test, predicted)
print 'cm: ', cm

genre_list = subdirs
name = 0
title = 'Confusion matrix of a CEPS based classifier' 
filename = 'music_ceps_cm.pdf'
plot_confusion_matrix(cm, genre_list, name, title, filename)

