
# coding: utf-8

# In[1]:


import os
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import json
import pandas as pd
import random
import numpy as np


# In[49]:


morph_file = os.path.join('data', 'raw', 'kt.voc.morpheme.585k.train')
class_file = os.path.join('data', 'raw', 'ktvoc_cls.txt')
class_df = pd.read_csv(class_file, sep='\t', names=['label'])


# In[62]:





# In[ ]:


labels = class_df['label'].unique()
label_dic = {label : 0 for label in labels}
df_reader = pd.read_csv(morph_file, sep='\t', chunksize=1000)
train_df = pd.DataFrame(columns=['label', 'document'])
test_df = pd.DataFrame(columns=['label', 'document'])
for index, df in enumerate(df_reader):
    print(f'index*1000={index*1000}, len(test_df)={len(test_df)}, len(train_df)={len(train_df)}')
    for i, row in df.iterrows(): 
        curr_label = row['label']
        curr_count = label_dic.get(curr_label, 0)
        if curr_count % 10 == 0:
            test_df.loc[len(test_df)] = df.loc[i]
        else:
            train_df.loc[len(train_df)] = df.loc[i]
        label_dic[curr_label] = curr_count + 1
#    break


# In[87]:


#len(train_df)


# In[89]:


#len(test_df)
#label_dic

