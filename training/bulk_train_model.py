import subprocess

import sys, getopt
import yaml

if __name__ != "__main__":
   exit()

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "hc:")
except getopt.GetoptError:
    print('bulk_train_model.py -c <config_path>')
    sys.exit(2)

CONFIG_PATH = None

for opt, arg in opts:
    if opt == '-h':
        print('bulk_train_model.py -c <config_path>')
        sys.exit()
    elif opt == '-c':
        CONFIG_PATH = arg

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

for model_config in config["models"]:
    print(f"STARTING TRAINING OF MODEL BY {model_config}")
    subprocess.run(["python", "train_model.py", "-r", config["root_dir"], "-c", model_config, "-s", config["split_path"]])