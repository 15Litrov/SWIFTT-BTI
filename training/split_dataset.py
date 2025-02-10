import os, sys, getopt
import pandas as pd
import numpy as np
import cv2

def main(argv):
    test_year = '2021'
    train_percent = 80
    data_dir = ''
    save_to = 'split.csv'

    try:
      opts, args = getopt.getopt(argv, "hi:o:", ["test-year=", "train-percent="])
    except getopt.GetoptError:
      print ('split_dataset.py -i <datadir> -o <saveto> test-year=<test_year> train-percent=<train_percent>')
      sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print ('split_dataset.py -i <datadir> -o <saveto> test-year=<test_year> train-percent=<train_percent>')
            sys.exit()
        elif opt == '-i':
            data_dir = arg
        elif opt == '-o':
            save_to = arg
        elif opt == 'test-year':
            test_year = arg
        elif opt == 'train-percent':
            train_percent = int(arg)

    dataset_df = pd.DataFrame(columns=["image_name", "image_path", "mask_path", "split_name"])
    image_names_set = set()
    image_names = []
    dir_names = []
    split_names = []

    for file_name in os.listdir(data_dir):
        file_type = os.path.splitext(file_name)[0].split('_')[-1]
        image_name = file_name[0:file_name.find('_VV')] if file_type == 'VV' else file_name[0:file_name.find('_GT')]

        if image_name in image_names_set:
            continue

        mask = cv2.imread(os.path.join(data_dir, f"{image_name}_GT_LABELS.tif"), cv2.IMREAD_UNCHANGED)
        if not (np.unique(mask).tolist() == [0, 1]):
            continue

        if file_name.startswith(test_year):
            split_names.append("TEST")
        else:
            split_names.append("TRAIN" if np.random.uniform(0, 100) < train_percent else "VALID")

        image_names_set.add(image_name)
        image_names.append(image_name)
        dir_names.append(data_dir)

    dataset_df["image_name"] = image_names
    dataset_df["dir_name"] = dir_names
    dataset_df["split_name"] = split_names

    dataset_df.to_csv(save_to)

if __name__ == "__main__":
   main(sys.argv[1:])