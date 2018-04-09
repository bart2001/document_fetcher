import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import multiprocessing
from gensim.models import doc2vec
from gensim.models import Doc2Vec
from collections import namedtuple
import glob
from module.morph_analyzer import morph_analyzer
from pprint import pprint

def read_stemmed_data(directory):
    path_name = directory + "*.txt"
    doc = {}
    docs = []
    texts_ko = []
    sentences = []
    doc_file_names = []

    #print("len(glob.glob(path_name))={}".format(len(glob.glob(path_name))))

    for file in glob.glob(path_name):
        with open(file, "r", encoding='utf-8') as paper:
            head, tail = os.path.split(file)
            #print(head, tail)
            start_num = tail.find('.pdf')
            #print(start_num)
            f_name = tail[:start_num + 4]
            #print(f_name)
            text = paper.read().replace('\n', ' ').lower()
            #print (text)
            word_list = [xx.strip() for xx in text.split()]
            #print(word_list)
            texts_ko.append(word_list)

            #print(text)
            sentences.append(text)
            try:
                value = doc[f_name]
                value += " "
                value += text
                doc[f_name] = value
            except KeyError:
                # Key is not present
                doc[f_name] = text
                doc_file_names.append(f_name)
                pass
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

    for i, j in enumerate(doc.keys()):
        pprint('i={}, j={}'.format(i, j))
        words = doc[j].lower().split()
        tags = [j]
        # print (tags,j,self.doc[j])
        docs.append(analyzedDocument(words, tags))

    #pprint('words={}, tags={}'.format(words, tags))
    print('docs[0]={}'.format(docs[0]))
    print('docs[-1]={}'.format(docs[-1]))
    return docs, doc_file_names

def doc2vec_train(model_f_name, docs):
    # reload(sys)
    # sys.setdefaultencoding('utf-8')
    cores = multiprocessing.cpu_count()
    # doc2vec parameters
    vector_size = 300
    window_size = 10  # 가까이 있는것을 중요하게 봄
    word_min_count = 2
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 0  # 0 = dbow; 1 = dmpv
    # hs =1
    worker_count = cores  # number of parallel processes
    #worker_count = 1

    word2vec_file = model_f_name + ".word2vec_format"
    #sentences=doc2vec.TaggedLineDocument(inputfile)
    # build voca
    doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count, size=vector_size, alpha=0.025, min_alpha=0.025, seed=1234, workers=worker_count)
    doc_vectorizer.build_vocab(docs)
    #doc_vectorizer.save_word2vec_format(word2vec_file, binary=False)

    # Train document vectors!
    #for epoch in range(12):
    #일단 1번만 반복
    for epoch in range(12):
        #print(epoch, doc_vectorizer.alpha)
        #doc_vectorizer.train(docs,total_examples=doc_vectorizer.corpus_count)
        doc_vectorizer.train(docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
        doc_vectorizer.alpha -= 0.002 # decrease the learning rate
        doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay

    # To save
    doc_vectorizer.save(model_f_name)

# 모델파일 로드
def doc2vec_model_load(model_path):
    return doc2vec.Doc2Vec.load(model_path)

# 유사문서 찾기
#def sim_doc_search(q, N, doc2vec_model):
def sim_doc_search(q, N, doc2vec_model, title_dict):
    org_q = q
    #tagged = tagger.pos(q)
    result_list = morph_analyzer.analyze(q)
    tagged = result_list['pos']

    qry_list = [s for s, t in tagged if (t == 'NNG' or t == 'NNP' or t == 'SL' or t == 'UNKNOWN')]
    print("qry_list=", qry_list)
    q = ' '.join(qry_list)
    print("q=",q)
    doc2vec_model.random.seed(0)
    _infer_vector = doc2vec_model.infer_vector(q)

    # _infer_vector=[str(x) for x in RW.doc2vec_model.infer_vector(q, alpha=0.01, steps=1000)]
    # print (q)
    # print (_infer_vector)
    sim_doc_set = []
    res = {}
    # print (_infer_vector)
    try:
        similar_documents = doc2vec_model.most_similar([_infer_vector], topn=N)
    except KeyError:
        sim_doc_set = []
        ir_stat = {}
        doc_cnt = 0

    doc_cnt = len(similar_documents)
    print('similar_documents=', similar_documents)
    ir_stat = {}
    print(similar_documents)
    for i in range(len(similar_documents)):
        cell_dict = {}
        fname = similar_documents[i][0]
        # inx = similar_documents[i][0]
        # value = self.doc_file_names[inx]
        cell_dict['docurl'] = fname
        #cell_dict['title'] = title_dict[fname]
        sim_doc_set.append(cell_dict)

    ir_stat['doc_cnt'] = doc_cnt
    res['TOTAL'] = ir_stat
    res['SEARCH_TITLE'] = org_q
    res['document_list'] = sim_doc_set
    return res

def read_title(title_file_nm):
    title_dict = {}
    #title_file_nm = "/home/mini/data/papertitle.txt"
    f = open(title_file_nm, 'r', encoding='utf-8')
    for line in f:
        f_name,title = line.split('\t')
        start_num = f_name.find('.pdf')
        f_name = f_name[:start_num+4]
        #self.title_dict[f_name]=unicode(title)
        title_dict[f_name]= title
    return title_dict

if __name__ == '__main__':
    #list_a = [1, 2, 3, 4]
    #list_b = [5, 6, 7, 8]
    #a= list(zip(list_a, list_b))
    #print(a)
    #exit()

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stemmed_dir = PROJECT_DIR + '/data/associated_doc/mobis_data/stemmed/'
    title_dict_path = PROJECT_DIR + '/data/associated_doc/mobistitle.md'
    docs, doc_file_names = read_stemmed_data(stemmed_dir)
    #pprint('docs={}'.format(docs))
    #pprint('doc_file_names={}'.format(doc_file_names))
    #exit()
    model_dir = PROJECT_DIR + '/model/associated_doc/'
    model_path = PROJECT_DIR + '/model/associated_doc/doc2vec.md'
    #doc2vec_train(model_path, docs)
    doc2vec_model = doc2vec_model_load(model_path)
    title_dict = read_title(title_dict_path)
    pprint("title_dict={}".format(title_dict))
    result = sim_doc_search('데이터베이스', 3, doc2vec_model, title_dict)
    print(result)

    #print(len(docs))
