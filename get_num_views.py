import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

files=glob.glob("../camo-data/*/good_cams.txt")
result=[]
for file in files:
    with open(file,'r') as f:
        good_views=f.readlines()
    #if len(good_views)<9:
    #    print(file,", num good views:",len(good_views))
    seed= sum([ord(x) for x in file.replace("good_cams.txt","")[-5:]])
    print(file,"seed", seed)
    train_views,test_views=train_test_split(np.arange(len(good_views)),test_size=max(1,min(3,len(good_views)-6)),random_state=seed,)
    with open(file.replace("good_cams.txt",'train_val_v3.pkl'), 'wb+') as f:
        pickle.dump((train_views,test_views), f,protocol=2)
    result.append({'scene':file.split('/')[-2],'n_good_views':len(good_views),'train':train_views,'test':test_views})
pd.DataFrame(result).sort_values('scene').to_csv("Num_good_views.csv",index=False)