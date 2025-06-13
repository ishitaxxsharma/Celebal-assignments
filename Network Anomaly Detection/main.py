import numpy as np
import os, random, math, sys, signal
from shutil import copyfile
from importlib import reload
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# for debugging
#import ptvsd
#ptvsd.enable_attach(log_dir= os.path.dirname(__file__))
#ptvsd.wait_for_attach(timeout=15)

from util import options, data_processing
from ml_models import autoencoder

def train(args):
    print(args.model_name)
    
    #Prepare the data for training
    process_data = data_processing.processData(args.data_path)
    X_train, Y_train, X_val, Y_val = process_data.prepareTrainingData()
    
    # Pre-train AutoEncoder with normal data
    model = autoencoder.AutoEncoder(X_train.shape[1])
    compiled_model = model.compile_model() 
    print("model compiled")
    
    # We need to rescale the data
    x_scale = process_data.dataScaling(X_train)
    x_norm, x_fraud = x_scale[Y_train == 0], x_scale[Y_train == 1]
    
    
    #Let's pretrain the compiled autoencoder with normal data, we don't need to train with every data point!
    compiled_model.fit(x_norm[0:8000], x_norm[0:8000], 
                batch_size = 256, epochs = 50, 
                shuffle = True, validation_split = 0.20);
    
    save_path = os.path.join(args.ckpt_path, args.model_name)
    model.save_load_models(path=save_path, model=compiled_model)
    del(compiled_model)
    compiled_model = model.save_load_models(path=save_path, mode="load")

    # Now Let's try to get latent representation of the trained model
    hidden_representation = model.getHiddenRepresentation(compiled_model)

    norm_hid_rep = hidden_representation.predict(x_norm[9000:15000])
    fraud_hid_rep = hidden_representation.predict(x_fraud[9000:15000])
    rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
    y_n = np.zeros(norm_hid_rep.shape[0])
    y_f = np.ones(fraud_hid_rep.shape[0])
    rep_y = np.append(y_n, y_f)

    #Finally we can train a classifier on learnt representations

    train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
    clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    #Let's also save this classifier for future
    filename = os.path.join(save_path, "linear_regression_Classifier.pkl")
    s = pickle.dump(clf, open(filename, 'wb'))
