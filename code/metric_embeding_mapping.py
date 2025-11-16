import numpy as np
import pickle as pkl
import pandas as pd

# Load data
metric_names=pd.read_json(r"D:\code\repo\M.tech\sem1\DA\LAB\contest\dataset\metric_names.json").to_numpy()
metric_def=np.load(r"D:\code\repo\M.tech\sem1\DA\LAB\contest\dataset\metric_name_embeddings.npy")

mapping={}

if __name__=="__main__":
    print(metric_names.shape)
    print(metric_def.shape)

    for i in range(metric_names.shape[0]):
        mapping[metric_names[i,0]]=metric_def

    with open(r"D:\code\repo\M.tech\sem1\DA\LAB\contest\dataset\defination_embedding1.pkl","wb") as file:
        pkl.dump(mapping,file)
