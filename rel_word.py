
# coding: utf-8

# In[18]:

import sys
import codecs
import os
import glob
import multiprocessing
import gensim
from collections import namedtuple
from gensim.models import word2vec
from gensim.models import doc2vec
from collections import Counter
from gensim.models import Phrases

from gensim.models.keyedvectors import KeyedVectors

from konlpy.tag import Mecab
from konlpy.tag import Kkma
from konlpy.utils import pprint
from konlpy.tag import Twitter
from nltk.stem import PorterStemmer

import sys, getopt

class RelWordsSystem:
    
    def __init__(self):
        self.config = {
            'min_count': 5,  # 등장 횟수가 5 이하인 단어는 무시
            'size': 200,  # ???차원짜리 벡터스페이스에 embedding
            'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용한다
            'batch_words': 10000,  # 사전을 구축할때 한번에 읽을 단어 수
            'iter': 50,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수
            'window':5, # 윈도우 사이즈
            'workers': multiprocessing.cpu_count(),
        }
        #self.projectdir = os.path.dirname(os.path.dirname(__file__))
        
        #self.tagger = Mecab(dicpath=self.projectdir+"/install/mecab-ko-dic/dic")
        self.tagger = Mecab()
        #self.tagger = Mecab("/home/mini/work/chatbot2017/install/mecab-ko-dic/dic")
        self.twitter_tagger = Twitter()
        self.ps = PorterStemmer()
        self.title_dict ={}
        self.docs = []
        self.texts_ko =[]
        self.sentences = []
        self.doc_file_names = []
        
    def read_title(self,title_file_nm):
        #title_file_nm = "/home/mini/data/papertitle.txt"
        f = open(title_file_nm, 'r')
        for line in f:
            f_name,title = line.split('\t')
            start_num = f_name.find('.pdf')
            f_name = f_name[:start_num+4] 
            #self.title_dict[f_name]=unicode(title)
            self.title_dict[f_name]= title
            
    def read_stemmed_data(self,directory):
        path_name = directory +"*.txt"
        self.doc={}
        for file in glob.glob(path_name):
            with open(file, "r") as paper:
                head, tail = os.path.split(file)
                start_num = tail.find('.pdf')
                f_name = tail[:start_num+4] 
                text = paper.read().replace('\n',' ').lower()
                #print (text)
                word_list = [xx.strip() for xx in text.split()]
                self.texts_ko.append(word_list)
                #print (word_list)
                self.sentences.append(text)
                try:
                    value = self.doc[f_name]
                    value += " "
                    value += text
                    self.doc[f_name] = value
                except KeyError:
                    # Key is not present
                    self.doc[f_name] = text
                    self.doc_file_names.append(f_name)
                    pass

        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        
        for i,j in enumerate(self.doc.keys()):
            words = self.doc[j].lower().split()
            tags=[j]
            #print (tags,j,self.doc[j])
            self.docs.append(analyzedDocument(words, tags))
            
        #for idx, doc in enumerate(self.doc_list):
        #    yield LabeledSentence(words=doc.split(),labels=[self.labels_list[idx]])
        """
        for i, text in enumerate(self.sentences):
            words = text.lower().split()
            tags = [i]
            self.docs.append(analyzedDocument(words, tags))
       """
    def stop_word_check(self, word):
        stop_word = ['keykeykeykey']  # stopword 등록 
        for _filter_word in stop_word:
            res = _filter_word.get(word,0)
            return res
    
    def read_data(self, directory):
        
            
        path_name = directory +"*.txt"
        self.doc={}
        
        filelist = []
        for (path, dir, files) in os.walk(directory):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.txt':
                    name = path+"/"+filename
                    #print (name) 
                    filelist.append(name)
        for file in filelist:
            with open(file, "r") as paper:
                head, tail = os.path.split(file)
                start_num = tail.find('.pdf')
                f_name = tail[:start_num+4] 
                text = paper.read().replace('\n',' ').lower()
                
                #print (text)
                #word_list = [xx.strip() for xx in text.split()]
                tagged = self.twitter_tagger.pos(text)
                index_list = [s for s, t in tagged if(t=='Noun' or (t=='Alpha' and s.isalpha()))]
                stem_list = [self.ps.stem(xx) for xx in index_list]
                #print (tagged)
                #print (index_list)
                #print (stem_list)
                self.texts_ko.append(stem_list)
                #print (word_list)
                self.sentences.append(text)
                try:
                    value = self.doc[f_name]
                    value += " "
                    value += text
                    self.doc[f_name] = value
                except KeyError:
                    # Key is not present
                    self.doc[f_name] = text
                    self.doc_file_names.append(f_name)
                    pass

        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        
        for i,j in enumerate(self.doc.keys()):
            words = self.doc[j].lower().split()
            tags=[j]
            #print (tags,j,self.doc[j])
            self.docs.append(analyzedDocument(words, tags))
        
    def word2vec_train(self,model_f_name):
        _bigram =0
        import gensim, logging
        if(_bigram):
            bigram = Phrases(RW.texts_ko)
            self.model= word2vec.Word2Vec(bigram[self.texts_ko], **self.config)
        else :
            self.model= word2vec.Word2Vec(self.texts_ko, **self.config)
        
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        """
        outv = KeyedVectors()
        outv.vocab = self.model.wv.vocab
        outv.index2word = self.model.wv.index2word
        outv.syn0 = self.model.syn1neg
        """
        #inout_sim = outv.most_similar('navi')
        #print (inout_sim)
        #fname = str(self.config['size'])+'_'+str(self.config['window'])+ '_'+model_f_name
        self.model.save(model_f_name)#test
        self.model.init_sims(replace=True)
        
    def word2vec_model_load(self,model_path):
        self.model = word2vec.Word2Vec.load(model_path)
        
    def doc2vec_model_load(self,model_path):
        self.doc2vec_model = doc2vec.Doc2Vec.load(model_path)

    
    def doc2vec_train(self, model_f_name):
        #reload(sys)
        #sys.setdefaultencoding('utf-8')
        cores = multiprocessing.cpu_count()
        #doc2vec parameters
        vector_size = 300
        window_size = 10  #가까이 있는것을 중요하게 봄
        word_min_count = 2
        sampling_threshold = 1e-5
        negative_size = 5
        train_epoch = 100
        dm = 0 #0 = dbow; 1 = dmpv
        #hs =1
        worker_count = cores #number of parallel processes
        
        word2vec_file = model_f_name + ".word2vec_format"
        #sentences=doc2vec.TaggedLineDocument(inputfile)
        #build voca 
        doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count, size=vector_size, alpha=0.025, min_alpha=0.025, seed=1234, workers=worker_count)
        doc_vectorizer.build_vocab(self.docs)
        # Train document vectors!
        for epoch in range(12):
            print (epoch,doc_vectorizer.alpha)
            #doc_vectorizer.train(docs,total_examples=doc_vectorizer.corpus_count)
            doc_vectorizer.train(self.docs,total_examples=doc_vectorizer.corpus_count,epochs=doc_vectorizer.iter)
            doc_vectorizer.alpha -= 0.002 # decrease the learning rate
            doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
        # To save
        doc_vectorizer.save(model_f_name)
        doc_vectorizer.save_word2vec_format(word2vec_file, binary=False)
    def rel_word_search(self,q,N):
        res_list ={}
        try:
            #res = self.model.most_similar(q,topn=N)
            res = self.model.most_similar(positive=[self.model.syn1neg[self.model.wv.vocab[q].index]]) #negative sample
            rel_word_list = []
            for i in res:
                rel_word_list.append(i[0])
            res_list[q]=rel_word_list
        except KeyError:
            resl_list ={}
            
        return res_list
    
    def rel_word_search2(self,q,N):
        res_list ={}
        _q = self.model[q]
        try:
            res = self.model.most_similar(positive=[self.model.syn1neg[self.model.wv.vocab[_q].index]]) #negative sample
            #res = self.model.most_similar([_q],topn=N)
            rel_word_list = []
            for i in res:
                rel_word_list.append(i[0])
            res_list[q]=rel_word_list
        except KeyError:
            resl_list ={}
            
        return res_list
    def rel_word_with_wgt_search2(self,q,N):
        tagged = self.tagger.pos(q)
        self.qry_list = [s for s, t in tagged if(t=='NNG' or t=='NNP' or t=='SL' or t=='UNKNOWN')]
        self.qry_list = [self.ps.stem(xx) for xx in self.qry_list]
        res_list ={}
        relWD_set ={}
        for _q in self.qry_list:
            try:
                res = self.model.most_similar(positive=[self.model.syn1neg[self.model.wv.vocab[_q].index]])
                for i in res:
                    relWD_set[i[0]]=i[1]
                res_list[q]=relWD_set
            except KeyError:
                continue;
        return res_list
    
    def rel_word_with_wgt_search(self,q,N):
       
        res_list ={}
        relWD_set ={}
        try:
            res = self.model.most_similar(positive=[self.model.syn1neg[self.model.wv.vocab[q].index]]) #negative sample
            for i in res:
                relWD_set[i[0]]=i[1]
            res_list[q]=relWD_set
        except KeyError:
            return res_list
        return res_list
    def sim_doc_search(self,q,N):
        org_q = q
        tagged = self.tagger.pos(q)
        qry_list = [s for s, t in tagged if(t=='NNG' or t=='NNP' or t=='SL' or t=='UNKNOWN')]
        q = ' '.join(qry_list)
        self.doc2vec_model.random.seed(0)
        _infer_vector = self.doc2vec_model.infer_vector(q)
        
        #_infer_vector=[str(x) for x in RW.doc2vec_model.infer_vector(q, alpha=0.01, steps=1000)]
        #print (q)
        #print (_infer_vector)
        sim_doc_set=[]
        res = {}
        #print (_infer_vector)
        try:
            similar_documents = self.doc2vec_model.docvecs.most_similar([_infer_vector], topn = N)
            #similar_documents = self.doc2vec_model..most_similar([_infer_vector], topn = N)
        except KeyError:
            sim_doc_set=[]
            ir_stat = {}
            doc_cnt =0 
            
        doc_cnt = len(similar_documents)
        ir_stat = {}
        print (similar_documents)
        for i in range(len(similar_documents)):
            cell_dict={}
            fname =similar_documents[i][0]
            #inx = similar_documents[i][0]
            #value = self.doc_file_names[inx]
            cell_dict['docurl']=fname     
            cell_dict['title']= self.title_dict[fname]
            sim_doc_set.append(cell_dict)
        
        ir_stat['doc_cnt'] = doc_cnt
        res['TOTAL']= ir_stat
        res['SEARCH_TITLE'] = org_q
        res['document_list']= sim_doc_set
        return res
        
   
        
def main(argv):
    
    RW = RelWordsSystem()
    
    """
    
    #검색엔진모델에서 사용한 색인어 파일을 로딩
    RW.read_stemmed_data("/home/mini/data/mobis_data/stemmed/")
    # 제목 데이터 로딩
    RW.read_title("/home/mini/data/mobistitle.txt")
    #model train
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #RW.word2vec_train('paper_word2vec.model')
    doc_model = RW.doc2vec_train()
    model = RW.word2vec_model_load("/home/mini/mobis_word2vec.model")
    doc2vecmodel = RW.doc2vec_model_load("/home/mini/mobis_doc2vec.model")

    query ='자동차 배터리와 유사한 문서'
    #res=RW.rel_word_search(query,20)
    res2 = RW.rel_word_with_wgt_search(query,20)
    res3 = RW.sim_doc_search(query,3)
    #RW.model.wv.vocab
    #for i in RW.model.wv.vocab.keys():
    #    print (i,RW.model.wv.vocab[i].count)
    #RW.model.predict_output_word("차량 오디오", topn=10)
    """
    try:
        opts, args = getopt.getopt(argv,"w:d:f:")
    except getopt.GetoptError:
        print (" ex >>>>>>>>>>>>python rel_word.py -w /home/mini/data/mobis_data/stemmed/ -f word2vec.model")
        print (" ex >>>>>>>>>>>>python rel_word.py -d /home/mini/data/mobis_data/stemmed/ -f doc2vec.model")
        sys.exit(2)
        
    mode = ''
    stemmed_dir = ''
    for opt, arg in opts:
        
        if opt in "-w":
            mode = 'w'
            stemmed_dir = arg
            
        elif opt in "-d":
            mode = 'd'
            stemmed_dir = arg
            
        elif opt in "-f":
            fname = arg
            
    if(mode == 'w'):
        RW.read_stemmed_data(stemmed_dir)
        word_model = RW.word2vec_train(fname)
    elif (mode =='d'):
        RW.read_stemmed_data(stemmed_dir)
        doc_model = RW.doc2vec_train(fname)
        
if __name__ == "__main__":
    
    if(len(sys.argv)!=5):
        print (" ex >>>>>>>>>>>>python rel_word.py -w /home/mini/data/mobis_data/stemmed/ -f word2vec.model")
        print (" ex >>>>>>>>>>>>python rel_word.py -d /home/mini/data/mobis_data/stemmed/ -f doc2vec.model")
    args = sys.argv[1:]
    main(args)
    
#Glove 함수 
"""
import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
corpus = Corpus()
corpus.fit(RW.texts_ko, window=5)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.most_similar('내비게이션')

"""


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[39]:

"""
RW = RelWordsSystem()
stemmed_dir ='/home/mini/data/mobis_data/stemmed/'
fname = 'word2vec.model'

RW.read_stemmed_data(stemmed_dir)
word_model = RW.word2vec_train(fname)
model = RW.word2vec_model_load(fname)
RW.rel_word_with_wgt_search('블루투스',5)

RW.read_title("/home/mini/data/mobistitle.txt")
RW.doc2vec_model_load('doc2vec.model')
RW.sim_doc_search('블루투스',3)
"""

