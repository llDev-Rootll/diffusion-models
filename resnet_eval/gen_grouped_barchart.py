from cProfile import label
import matplotlib.pyplot as plt
import os
import json
import argparse
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('-f','--folders', nargs='+', help='Path to results folder for one or multiple experiments', required=True)
args = vars(parser.parse_args())
files = []
for folder in args["folders"]:
    files+=os.listdir(folder)


result_jsons = [file for file in files if ("results" in file and ".json" in file)]
width=3
# plt.figure("Comparitive plot")
df = {}
class_names = None
for idx, (json_file, base_path) in enumerate(zip(result_jsons, args['folders'])):
    json_file_path = os.path.join(base_path, json_file)
    f = open(json_file_path)
    results = json.load(f)
    # print(results)
    dataset_name = json_file.split("_")[:-1]
    dataset_name.remove("results")
    dataset_name = "_".join(dataset_name)
    class_names = list(results.keys())

    df[dataset_name] = list(results.values())

df = pd.DataFrame(df, index=class_names)
ax = df.plot.bar(rot=0)
plt.savefig("resnet_eval/outputs/"+"_".join(df.columns)+".png")