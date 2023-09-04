"""
paderborn_data_loader.py

a class to handle loading and splitting the paderborn bearing data

we will use the windowed training data for cross validation and then the 
held out data (x_test) for blind fold testing

note: files need to be downloaded and extracted into the following structure

healthy/normal/*.mat
artifical/ir/*.mat
artificial/or/*.mat
real/ir/*.mat
real/or/*.mat

for the real data KB23, KB24, and KB27 have been excluded as they have both
inner and outer race damage

author: alex shenfield
date:   17/04/2020
"""

import re 

import numpy as np
import scipy.io as sio

from tqdm import tqdm
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# implementation of robust z-score normalisation - where we center and scale
# to have median = 0, and median absolute deviation = 1
def normalise(x, axis=0):
    
    # calculate the center and scale
    x_center = np.median(x, axis=axis)
    x_scale = np.median(np.abs(x - x_center), axis=axis)
    
    # normalise our data and return
    x_norm = (x - x_center) / x_scale
    return x_norm
    

# dictionary mapping class names (i.e. faults) to integers
#faults_idx = {
    # normal healthy bearings
    #'K001': 0,
    #'K002': 0,
    # 'K003': 0,
    # 'K004': 0,
    # 'K005': 0,
    # 'K006': 0,
    # artificial damage
    #'KA01': 1,
    #'KA03': 1,
    #'KA05': 1,
    # 'KA06': 1,
    #'KA07': 1,
    # 'KA08': 1,
    # 'KA09': 1,
    # 'KI01': 2,
    # 'KI03': 2,
    # 'KI05': 2,
    # 'KI07': 2,
    # 'KI08': 2,
    # real damage
    #'KI04': 1,
    #'KI14': 1,
    # 'KI16': 1,
    # 'KI17': 1,
    # 'KI18': 1,
    # 'KI21': 1,
    # 'KA04': 2,
    # 'KA15': 2,
    #'KA16': 2,
    #'KA22': 2,
    # 'KA30': 2,
    #'KB23': 'IROR',
    #'KB24': 'IROR',
    #'KB27': 'IROR',
#}


# this is the main class for preprocessing and loading the data
class PaderbornData:
    
    """Load the Paderborn bearing data.
    
    This class loads the specified bearing data matlab files, pulls out the 
    variables we are interested in, and then divides the data into train and 
    test sets.
    
    Arguments:
        root_dir       - the base directory where the data can be found
        experiment     - the subset of the data to use (e.g. artificial, 
                         artificial/ir, real, etc.)
        normalisation  - which normalisation method to use - can be: 
                         'robust-zscore' (zero median and unit m.a.d)
                         'standard-scaler' (zero mean and unit std)
                         'robust-scaler' (uses the robust scaler from scikit-learn)
                         None (no normalisation applied)
    """
    
    # data specific magic numbers - this requires understanding what the mat files
    # look like inside
    mcs_1 = 1
    mcs_2 = 2
    vibration = 6
    
    def __init__(self, root_dir, experiment='artificial', datastream='vibration',
                 normalisation='robust-zscore'):
        
        # get the paths
        self.root_dir = root_dir
        self.normal_dir = Path(root_dir, 'healthy/normal')   
        self.damage_dir = Path(root_dir, experiment)   

        # build a list of all the files
        filelist = list()
        
        # get the healthy normal bearing files
        for p in self.normal_dir.rglob("*.mat"):
            filelist.append(p)
        
        # get the damaged bearing files
        for p in self.damage_dir.rglob("*.mat"):
            filelist.append(p)
        
        filelist.sort()
        self.files = filelist
        
        # store the datastream we are interested in
        self.datastream = datastream
        
        # apply specified normalisation
        if normalisation:
            if normalisation == 'robust-zscore':
                self.transform = lambda x : normalise(x)
            elif normalisation == 'standard-scaler':
                self.scaler = StandardScaler()
                self.transform = lambda x : self.scaler.fit_transform(x)
            elif normalisation == 'robust-scaler':
                self.scaler = RobustScaler()
                self.transform = lambda x : self.scaler.fit_transform(x)
        else:
            self.transform = lambda x : x
                
    
    # retrieve the label from the file name
    def _get_class(self, f, faults_idx):        
        c = None
        for k in faults_idx.keys():
            if k in str(f):
                c = k
                return faults_idx[c]
        
        return None
        
        
    # extract the data and split it into train and test
    def split_data(self, data_length, train_fraction=0.5, 
                   window_length=2048, window_step=64, 
                   faults_idx={}, verbose=False):        
                #store faults
        self.faults_idx = faults_idx
        print(f"faults:{self.faults_idx}")
        
        # get the training end point and testing start point
        train_end = np.int_(np.around(data_length * train_fraction))
        
        # get the train and test splits using a fraction of the data for 
        # training (which we will window) and half for testing (which we wont)
        train_splits = np.arange(0, train_end, window_step)
        test_splits  = np.arange(train_end, (data_length - window_length), window_length)
        
        # information
        if verbose:
            print('we are using {0} training samples and {1} testing samples '
                  'from each fault category'.format(len(train_splits), 
                                                    len(test_splits)))
        
        # get the timeseries dimension based on the type of data we are 
        # using
        if self.datastream == 'vibration':
            ts_dim = 1
        elif self.datastream == 'motor':
            ts_dim = 2
        
        # initialise arrays for our training and testing data
        x_train = np.zeros((0, window_length, ts_dim))
        x_test  = np.zeros((0, window_length, ts_dim))      
        y_train = list()
        y_test  = list()

        # if we are not printing a load of output for each file then use a tqdm 
        # progress bar
        if verbose:
            file_iter = self.files
        else:
            file_iter = tqdm(self.files) 
        
        # for all the data files we are interested in
        for f in file_iter:
            if self._get_class(f,self.faults_idx) != None:
              try:
                # print the file we are processing
                if verbose:
                    print(str(f))
                print(f"file:{f}")
                # load the raw data from the matlab file
                data = sio.loadmat(f)
                data = data[list(data.keys())[-1]]
                
                # pull out the variables of interest and turn the data into a 'n' 
                # dimensional timeseries
                if self.datastream == 'vibration':
                    ts_data = data['Y'][0][0][0][self.vibration][2].T
                elif self.datastream == 'motor':
                    data_1 = data['Y'][0][0][0][self.mcs_1][2].T
                    data_2 = data['Y'][0][0][0][self.mcs_2][2].T
                    ts_data = np.hstack([data_1, data_2])
                
                # normalise using specified scaling / normalisation method
                ts_data = self.transform(ts_data)
                
                # really we should fit the scaler to the training set and then 
                # use those calculated values to transform the data - but tbh the
                # training set and the testing set look pretty much identical as 
                # they are gathered the same way from different parts of the same 
                # signal
                
                # print the timeseries shape so we can see how many samples and 
                # how many variables we have
                if verbose:
                    print(ts_data.shape)
                
                # now window the data for our training set (checking we are
                # actually supposed to be producing training data ...)
                if train_splits.size:
                    samples = list()
                    for start in train_splits:
                        samples.append(ts_data[start:start+window_length])        
                    x_train = np.vstack((x_train, np.stack(samples, axis=0)))
                
                # now get our test set in much the same way but without overlaps
                # in our windows (first checking we are actually supposed to be 
                # producing test data ...)
                if test_splits.size:
                    samples = list()
                    for start in test_splits:
                        samples.append(ts_data[start:start+window_length])        
                    x_test = np.vstack((x_test, np.stack(samples, axis=0)))
                
                if train_splits.size:
                    y_train.extend([self._get_class(f, self.faults_idx)] * len(train_splits))
                
                if test_splits.size: 
                    y_test.extend([self._get_class(f, self.faults_idx)] * len(test_splits))
              
              except Exception as e:
                print(f"error: {e}")
                print(f"failed to get get data for {f}")

        # return our training and testing data sets
        y_train = np.array(y_train)
        y_test  = np.array(y_test)
        return x_train, y_train, x_test, y_test