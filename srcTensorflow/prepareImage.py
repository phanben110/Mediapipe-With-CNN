import cv2 
import os
import json 
import numpy as np 

#DATASET_PATH = "/home/ben/ai/image/eyeAI/dataV2" 

#JSON_PATH = "dataImageV1.json"

DATASET_PATH = ".././data/raw" 

JSON_PATH = ".././data/processed/dataHandSize50.json"

labels = ["Ok", "Silent", "Dislike", "Like", "Hi" , "hello" , "stop" ] 


def preprocess_dataset(dataset_path, JSON_PATH ):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
       "image": [],
        "labels": [],
        "mapping": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            #label = dirpath.split("/")[-1]
            label = labels[i-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                print ( file_path )  
                feature = cv2.imread( file_path , 0 ) 
                feature = cv2.resize( feature , (50, 50)) 

                print(feature.shape ) 

                
                data["image"].append(feature.T.tolist() )
                data["labels"].append(i-1)
                print("{}: {}".format(file_path, i-1))

    # save data in json file
    #print(data) 
    print ( "==========================") 

    with open(JSON_PATH, 'w' ) as json_file:
        json.dump(data, json_file, indent =3)
      


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
