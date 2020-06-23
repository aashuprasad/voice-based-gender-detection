import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    
    features = mfcc.mfcc(audio,sr,0.025,0.01,13,appendEnergy=False)
    
    feat = np.asarray(())

 
    for i in range(features.shape[0]):
        
        temp = features[i,:]

        if np.isnan(np.min(temp)):
            continue
        else:
            
  
            if feat.size == 0 :
                feat = temp
            else:
                
                    feat = np.vstack((feat,temp))
    features = feat
    

    features = preprocessing.scale(features)
    
    return features                


sourcePath = "C:\\Users\\aashu\\OneDrive\\Desktop\\ashu\\Voice-Based-Gender-Detection-master\\pygender\\test_data\\AudioSet\\male_clips"


modelPath = "C:\\Users\\aashu\\OneDrive\\Desktop\\ashu\\Voice-Based-Gender-Detection-master\\Modules\\"


gmmFiles = [os.path.join(modelPath,fname) for fname in os.listdir(modelPath) if fname.endswith('.gmm')]

models = [cPickle.load(open(fname,'rb')) for fname in gmmFiles]

genders = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmmFiles]

files = [os.path.join(sourcePath,f) for f in os.listdir(sourcePath) if f.endswith(".wav")]

for f in files:

    print(f.split("\\")[-1])
   
    sr,audio = read(f)


    features = get_MFCC(sr,audio)

    scores = None

 
    log_likelihood = np.zeros(len(models))

  
    for i in range(len(models)):
        
        gmm = models[i]
        
   
        scores = np.array(gmm.score(features))
        
    
        log_likelihood[i] = scores.sum()
    
        
    winner = np.argmax(log_likelihood)
    
    print(genders[winner])

    
