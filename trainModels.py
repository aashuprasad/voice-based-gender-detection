#IMPORT REQUIRED LIBRARIES

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    
    features = mfcc.mfcc( audio , sr , 0.025 , 0.01 , 13 , appendEnergy = False )
    
    features = preprocessing.scale(features)

    return features

source = "C:\\Users\\aashu\\OneDrive\\Desktop\\ashu\\Voice-Based-Gender-Detection-master\\pygender\\train_data\\youtube\\male"

dest = "C:\\Users\\aashu\\OneDrive\\Desktop\\ashu\\Voice-Based-Gender-Detection-master\\Modules\\"

files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]

features = np.asarray(())

for f in files:
    
    sr,audio = read(f)
    
    vector  = get_MFCC(sr,audio)

    if features.size == 0:
        features = vector
    else:

        features = np.vstack((features, vector)) 

gmm = GaussianMixture(n_components = 8, covariance_type='diag', max_iter = 200 , n_init = 3 )

gmm.fit(features)

pickleFile = f.split("\\")[-2].split(".wav")[0]+".gmm"

cPickle.dump(gmm,open(dest + pickleFile,'wb'))

print ("Modeling Completed for Gender : " + pickleFile)
