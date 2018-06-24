
# coding: utf-8

# In[15]:


import os
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import json
import pandas as pd
import random
import numpy as np
from datetime import datetime


# In[4]:


morph_file = os.path.join('data', 'morph', 'kt.voc.morpheme.585k.train')
class_file = os.path.join('data', 'raw', 'ktvoc_cls.txt')
class_df = pd.read_csv(class_file, sep='\t', names=['label'])


# In[110]:


labels = class_df['label'].unique()
label_dic = {label : 0 for label in labels}
df_reader = pd.read_csv(morph_file, sep='\t', chunksize=1000)
test_file = os.path.join('data', 'morph', 'kt.voc.test')
train_file = os.path.join('data', 'morph', 'kt.voc.train')
if os.path.isfile(test_file):
    os.remove(test_file)
    print('remove test!')
if os.path.isfile(train_file):
    os.remove(train_file)
    print('remove train')

#train_df = pd.DataFrame(columns=['label', 'document'])
#test_df = pd.DataFrame(columns=['label', 'document'])
for index, df in enumerate(df_reader):
    print(f'index*1000={index*1000}, datetime={datetime.now()}')
    for i, row in df.iterrows(): 
        curr_label = row['label']
        curr_count = label_dic.get(curr_label, 0)
        if curr_count % 10 == 0:
            pd.DataFrame(row).transpose().to_csv(test_file, mode='a', header=False, sep='\t')
            #row.transpose().to_csv(test_file, mode='a', header=False, sep='\t')
        else:
            pd.DataFrame(row).transpose().to_csv(train_file, mode='a', header=False, sep='\t')
            #row.transpose().to_csv(train_file, mode='a', header=False, sep='\t')
        label_dic[curr_label] = curr_count + 1
    #break


# In[111]:


train_df = pd.read_csv(train_file, sep='\t', names=['label', 'document'])
test_df = pd.read_csv(test_file, sep='\t', names=['label', 'document'])


# In[124]:


len(test_df)+len(train_df)


# In[112]:


test_df.head(10), train_df.head(10)

