"""
The data is provided by 
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable
reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

    Code		Description
    N		Normal beat (displayed as . by the PhysioBank ATM, LightWAVE, pschart, and psfd)
    L		Left bundle branch block beat
    R		Right bundle branch block beat
    B		Bundle branch block beat (unspecified)
    A		Atrial premature beat
    a		Aberrated atrial premature beat
    J		Nodal (junctional) premature beat
    S		Supraventricular premature or ectopic beat (atrial or nodal)
    V		Premature ventricular contraction
    r		R-on-T premature ventricular contraction
    F		Fusion of ventricular and normal beat
    e		Atrial escape beat
    j		Nodal (junctional) escape beat
    n		Supraventricular escape beat (atrial or nodal)
    E		Ventricular escape beat
    /		Paced beat
    f		Fusion of paced and normal beat
    Q		Unclassifiable beat
    ?		Beat not classified during learning
"""

from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
from utils import *
from config import get_config

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report
import os

#-*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

def get_config():
        
    misc_arg = add_argument_group('misc')
    misc_arg.add_argument('--split', type=bool, default = True)
    misc_arg.add_argument('--input_size', type=int, default = 256, 
                        help='multiplies of 256 by the structure of the model') 
    misc_arg.add_argument('--use_network', type=bool, default = False)

    data_arg = add_argument_group('data')
    data_arg.add_argument('--downloading', type=bool, default = False)

    graph_arg = add_argument_group('graph')
    graph_arg.add_argument('--filter_length', type=int, default = 32)
    graph_arg.add_argument('--kernel_size', type=int, default = 16)
    graph_arg.add_argument('--drop_rate', type=float, default = 0.2)

    train_arg = add_argument_group('train')
    train_arg.add_argument('--feature', type=str, default = "MLII",
                        help='one of MLII, V1, V2, V4, V5. Favorably MLII or V1')
    train_arg.add_argument('--epochs', type=int, default = 80)
    train_arg.add_argument('--batch', type=int, default = 256)
    train_arg.add_argument('--patience', type=int, default = 10)
    train_arg.add_argument('--min_lr', type=float, default = 0.00005)
    train_arg.add_argument('--checkpoint_path', type=str, default = None)
    train_arg.add_argument('--resume_epoch', type=int)
    train_arg.add_argument('--ensemble', type=bool, default = False)
    train_arg.add_argument('--trained_model', type=str, default = None, 
                        help='dir and filename of the trained model for usage.')

    predict_arg = add_argument_group('predict')
    predict_arg.add_argument('--num', type=int, default = None)
    predict_arg.add_argument('--upload', type=bool, default = False)
    predict_arg.add_argument('--sample_rate', type=int, default = None)
    predict_arg.add_argument('--cinc_download', type=bool, default = False)
    config, unparsed = parser.parse_known_args()

    return config

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)



def plot_confusion_matrix(y_true, y_pred, classes, feature,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """Modification from code at https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"""
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    mkdir_recursive('results')
    fig.savefig('results/confusionMatrix-'+feature+'.eps', format='eps', dpi=1000)
    return ax


def add_noise(config):
    noises = dict()
    noises["trainset"] = list()
    noises["testset"] = list() 
    import csv
    try:
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv')))
    except:
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv')))
    for i, label in enumerate(testlabel):
      if label[1] == '~':
        filename = 'training2017/'+ label[0] + '.mat'
        from scipy.io import loadmat
        noise = loadmat(filename)
        noise = noise['val']
        _, size = noise.shape
        noise = noise.reshape(size,)
        noise = np.nan_to_num(noise) # removing NaNs and Infs
        from scipy.signal import resample
        noise= resample(noise, int(len(noise) * 360 / 300) ) # resample to match the data sampling rate 360(mit), 300(cinc)
        from sklearn import preprocessing
        noise = preprocessing.scale(noise)
        noise = noise/1000*6 # rough normalize, to be improved 
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(noise, distance=150)
        choices = 10 # 256*10 from 9000
        picked_peaks = np.random.choice(peaks, choices, replace=False)
        for j, peak in enumerate(picked_peaks):
          if peak > config.input_size//2 and peak < len(noise) - config.input_size//2:
              start,end  = peak-config.input_size//2, peak+config.input_size//2
              if i > len(testlabel)/6:
                noises["trainset"].append(noise[start:end].tolist())
              else:
                noises["testset"].append(noise[start:end].tolist())
    return noises

def preprocess(data, config):
    sr = config.sample_rate
    if sr == None:
      sr = 300
    data = np.nan_to_num(data) # removing NaNs and Infs
    from scipy.signal import resample
    data = resample(data, int(len(data) * 360 / sr) ) # resample to match the data sampling rate 360(mit), 300(cinc)
    from sklearn import preprocessing
    data = preprocessing.scale(data)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(data, distance=150)
    data = data.reshape(1,len(data))
    data = np.expand_dims(data, axis=2) # required by Keras
    return data, peaks

# predict 
def uploadedData(filename, csvbool = True):
    if csvbool:
      csvlist = list()
      with open(filename, 'r') as csvfile:
        for e in csvfile:
          if len(e.split()) == 1 :
            csvlist.append(float(e))
          else:
            csvlist.append(e)
    return csvlist

def preprocess(split):
    nums = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

    removed_patients = ['102', '104', '107', '217']
    # features = ['MLII', 'V1', 'V2', 'V4', 'V5'] 
    features = ['MLII']

    if split :
        testset = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
        trainset = ['101', '106', '108', '109','112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
    def dataSaver(dataSet, datasetname, labelsname):
        # classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
        AAMI = ['N','L','R','B','A','a','j','S','V','r','F','e','j','n','E','f','/','Q','?']
        Nclass = len(AAMI)
        datadict, datalabel= dict(), dict()

        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()

        def dataprocess():
          input_size = config.input_size 
          for num in tqdm(dataSet):
            from wfdb import rdrecord, rdann
            record = rdrecord('dataset/'+ num, smooth_frames= True)
            from sklearn import preprocessing
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signals0, distance=150)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            global lppened0, lappend1, dappend0, dappend1 
            lappend0 = datalabel[feature0].append
            lappend1 = datalabel[feature1].append
            dappend0 = datadict[feature0].append
            dappend1 = datadict[feature1].append
            # skip a first peak to have enough range of the sample 
            for peak in tqdm(peaks[1:-1]):
              start, end =  peak-input_size//2 , peak+input_size//2
              ann = rdann('dataset/'+ num, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
              
              def to_dict(chosenSym):
                y = [0]*Nclass
                y[AAMI.index(chosenSym)] = 1
                lappend0(y)
                lappend1(y)
                dappend0(signals0[start:end])
                dappend1(signals1[start:end])

              annSymbol = ann.symbol
              # remove some of "N" which breaks the balance of dataset 
              if len(annSymbol) == 1 and (annSymbol[0] in AAMI) and (annSymbol[0] != "N" or np.random.random()<0.15):
                to_dict(annSymbol[0])
 
        dataprocess()
        # noises = add_noise(config)
        for feature in ["MLII", "V1"]: 
            d = np.array(datadict[feature])
            if len(d) > 15*10**3:
                n = np.array(noises["trainset"])
            else:
                n = np.array(noises["testset"]) 
            datadict[feature]=np.concatenate((d,n))
            size, _  = n.shape 
            l = np.array(datalabel[feature])
            noise_label = [0]*Nclass
            noise_label[-1] = 1
            
            noise_label = np.array([noise_label] * size) 
            datalabel[feature] = np.concatenate((l, noise_label))
        import deepdish as dd
        dd.io.save(datasetname, datadict)
        dd.io.save(labelsname, datalabel)

    if split:
        dataSaver(trainset, 'dataset/train.hdf5', 'dataset/trainlabel.hdf5')
        dataSaver(testset, 'dataset/test.hdf5', 'dataset/testlabel.hdf5')
    else:
        dataSaver(nums, 'dataset/targetdata.hdf5', 'dataset/labeldata.hdf5')

def main(config):

    return preprocess(config.split)

if __name__=="__main__":
    config = get_config()
    main(config)