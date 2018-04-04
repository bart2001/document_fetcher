import json
import os
from pprint import pprint
#import pdb
#pdb.set_trace()
from collections import Counter
from konlpy.tag import Mecab
import random
import shutil
from wordcloud import WordCloud

from .mchatcommon import makeJSON
from .tfidf_scikit import IRSystem
from .rel_word import RelWordsSystem
class IRAgent:

	def __init__(self, name, doc_path, title_path, w2v_path, d2v_path):

		self.name = name
		projectdir = os.path.dirname(os.path.dirname(__file__))
		self.font_path = projectdir + '/install/pytagcloud/NANUMGOTHIC.TTF'
		self.irsys = IRSystem()
		self.irsys.read_data(projectdir + doc_path)
		self.irsys.read_title(projectdir + title_path)
		self.irsys.tfidf()	
		self.rwsys = RelWordsSystem()
		self.rwsys.read_title(projectdir + title_path)
		#self.rwsys.read_data(projectdir + doc_path+'/stemmed/')
		self.rwsys.word2vec_model_load(projectdir + w2v_path)
		self.rwsys.doc2vec_model_load(projectdir + d2v_path)
		self.wordclouddir = projectdir + "/mobis_chatbot/static/mobis_chatbot/img/"
		self.requestAgent = ''
		self.responseAgent = self.name

		self.result = self.name + '_RESULT'

		return

	def contextMgr(self, action):
		''' context를 수정할 부분 '''
		if action == 'init':
			keylist = self.variables.keys()
			for key in list(self.variables):
				# 이전에 이 에이전트에서 추가했다면 clear
				if key.find(self.name) == 0:
					del(self.variables[key])
			# agent 결과 구조 초기화
			self.variables[self.result] = {}


		return
		
	def interface(self, request):
		self.variables = json.loads(request)
		self.contextMgr(action='init')
		self.requestAgent = self.variables['REQUEST_AGENT']

		self.processing(self.requestAgent)

		self.contextMgr(action='update')
		
		return makeJSON(self.name, self.requestAgent, self.name, self.variables)

	def processing(self, sender):

		intent = self.variables[sender + '_RESULT']['INTENT']
		if intent == 'request_search':
			self.processing_request_search(sender)
		elif intent == 'find_similar_document':
			self.processing_find_similar_document(sender)
		elif intent == 'recommend_document':
			self.processing_recommend_document(sender)

		return 

	def processing_find_similar_document(self, sender):
		rwsys_topN=3
		msg = self.variables[sender + '_RESULT']['QUERY']
		response = self.rwsys.sim_doc_search(msg,rwsys_topN)
		self.variables[self.result]['IR_RESULT'] = response
		
	def processing_recommend_document(self, sender):
		rwsys_topN=3
		msg = self.variables['USER_CONTEXT']
		msg = ' '.join(msg)
		response = self.rwsys.sim_doc_search(msg,rwsys_topN)
		self.variables[self.result]['IR_RESULT'] = response

	def processing_request_search(self, sender):
		rwsys_topN=100

		msg = self.variables[sender + '_RESULT']['QUERY']
		response = self.irsys.response_generation(msg)
		#nounlist = self.postagger.nouns(msg)
		
		#rel_word_response = self.rwsys.rel_word_with_wgt_search(msg,rwsys_topN)
		#rel_doc_response = self.rwsys.sim_doc_search(msg,rwsys_topN)
		
		"""
		nounlist = response['nounlist']
		relwordDict = {}
		for noun in nounlist:
			rel_word_response = self.rwsys.rel_word_with_wgt_search(noun,rwsys_topN)
			# 면사가 연관어 대상에 없으면 skip
			if noun not in rel_word_response:
				continue
			for relword in rel_word_response[noun]:
				if relword not in relwordDict:
					relwordDict[relword] = 0
				relwordDict[relword] += int(rel_word_response[noun][relword]*8)

		# wordcloud name 설정
		randname = str(random.randrange(1,1000000))
		wordcloudname = "wordcloud_" + self.variables['USER'] + '_' + \
			self.variables['SESSION'] + '_' + randname +'.png'

		relwordList = []
		for n in relwordDict:
			for idx in range(relwordDict[n]):
				relwordList.append(n)

		count = Counter(relwordList)
		tag2 = count.most_common(20)

		if len(relwordList) > 0 :
			random.shuffle(relwordList)
			wordcloud = WordCloud(font_path=self.font_path,width=400, height=200, background_color='black').generate(' '.join(relwordList))
			wordcloud.to_file(self.wordclouddir + wordcloudname)
		else:
			pass
		"""
		wordcloudname = ""
		nounlist = response['nounlist']
		relwordDict = {}
		for noun in nounlist:
			rel_word_response = self.rwsys.rel_word_with_wgt_search(noun,rwsys_topN)
                        # 면사가 연관어 대상에 없으면 skip
			if noun not in rel_word_response:
				continue
			randname = str(random.randrange(1,1000000))

			relwordDict.update(rel_word_response[noun])
                # wordcloud name 설정
		if len(relwordDict) > 0 :
			randname = str(random.randrange(1,1000000))	
			wordcloudname = "wordcloud_" + self.variables['USER'] + '_' + self.variables['SESSION'] + '_' + randname +'.png'	
                        #random.shuffle(relwordList)
			wordcloud = WordCloud(font_path=self.font_path,width=400, height=200, background_color='black').generate_from_frequencies(relwordDict)
			wordcloud.to_file(self.wordclouddir + wordcloudname)
		else:
			pass

		response['document_page_list'] = response['document_page_list'][0:3]
			
		self.variables[self.result]['KEYWORDS'] = nounlist
		self.variables[self.result]['IR_RESULT'] = response
		self.variables[self.result]['RW_RESULT'] = relwordDict
		self.variables[self.result]['WORDCLOUD'] = wordcloudname


#if __name__ == "__main__":
#	main()
