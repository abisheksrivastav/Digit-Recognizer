import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
import config

def predict(test_data_path, model):
    sample = pd.read_csv("../input/sample_submission.csv")
    sampleimage = sample["ImageId"].values
    #predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        
        
        clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model}_{FOLD}.bin"))
        
        
        preds = clf.predict(df)

       

    sub = pd.DataFrame(np.column_stack((sampleimage, preds)), columns=["ImageId", "Label"])
    return sub
    

if __name__ == "__main__":
    submission = predict(test_data_path="../input/test.csv", 
                         model="rf")
   
    submission.to_csv("../models/rf_submission.csv", index=False)