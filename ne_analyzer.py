import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import marisa_trie, html, hgtk, csv
from module.morph_analyzer import morph_analyzer
from api.base_config import BaseConfig
from api import data_helpers
import time

LOGGER = BaseConfig().getLogger()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NE_DIC_DIR = PROJECT_DIR + '/data/ne_analyzer/dic'
TXT_NE_DIC_PATH = NE_DIC_DIR + '/named_entity.txt'
TXT_NE_DIC_EXTENDED_PATH = NE_DIC_DIR + '/named_entity_extended.txt'
COMPILED_NE_DIC_PATH = NE_DIC_DIR + '/named_entity.marisa'

# 공백이 있는 한글의 경우 공백없는 경우도 포함해서 확장해서 새로운 파일 만들어주기
def extend_dic():
    LOGGER.info('extending TXT_NE_DIC to EXTENDED_NE_DIC for removing spaces')
    ne_dic = []
    with open(TXT_NE_DIC_PATH, 'r', encoding='utf-8') as txt_ne_dic_file:
        lines = csv.reader(txt_ne_dic_file)
        for line in lines:
            # 잘못된 줄이 있는 경우 경우 continue
            # 주석인 경우 continue
            if len(line) < 2 or line[0][0] == '#':
                continue
            # key값을 대문자화하기
            key = str(line[0]).upper()
            value = html.unescape(line[1])
            ne_dic.append((key, value))

    #확장된 사전구조 만들어주기
    extended_ne_dic = []
    for ne in ne_dic:
        '''
        extended_ne_dic.append(ne)
        word = str(ne[0])
        ne_tag = str(ne[1])
        if hgtk.checker.is_hangul(word.replace(' ', '')) and ' ' in word:
            words_without_spaces = data_helpers.get_words_without_space(word)
        '''
        word = str(ne[0])
        ne_tag = str(ne[1])
        words_without_spaces = data_helpers.get_words_without_space(word)
        for word_without_space in words_without_spaces:
            extended_ne_dic.append((word_without_space, ne_tag))
    # 확장된 txt 사전 파일에 쓰기
    with open(TXT_NE_DIC_EXTENDED_PATH, 'w', encoding='utf-8') as txt_ne_dic_extended_file:
        for key_and_value in extended_ne_dic:
            write_line = '"{}","{}"'.format(key_and_value[0], key_and_value[1]) + '\n'
            txt_ne_dic_extended_file.write(write_line)
    return

# txt 사전 -> compiled 사전: .txt -> .marisa
def compile_dic():

    #사전파일이 존재하지 않을 경우
    if not os.path.isfile(TXT_NE_DIC_PATH):
        msg = "named_entity txt file does not exist"
        LOGGER.error(msg)
        return {'result': False, 'msg': msg}

    try:
        # 컴파일 중인 상태를 알려주는 파일 생성
        open(NE_DIC_DIR + '/is_compiling', 'a').close()

        #사전 확장
        extend_dic()

        keys = []
        values = []
        with open(TXT_NE_DIC_EXTENDED_PATH, 'r', encoding='utf-8') as dic_file:
            lines = csv.reader(dic_file)
            for line in lines:
                # 잘못된 줄이 있는 경우 경우 continue
                # 주석인 경우 continue
                if len(line) < 2 or line[0][0] == '#':
                    continue
                value = str.encode(html.unescape(line[1]))
                keys.append(line[0])
                values.append(value)

        trie = marisa_trie.BytesTrie(zip(keys, values))
        trie.save(COMPILED_NE_DIC_PATH)
        LOGGER.debug("length of keys={}".format(len(values)))
        LOGGER.debug("length of values={}".format(len(values)))
    # 예외가 생길경우 False 반환하고 메세지도 반환하기
    except Exception as e:
        return {'result': False, 'msg': str(e)}
    finally:
        # 컴파일 중인 상태를 알려주는 파일 삭제
        if os.path.exists(NE_DIC_DIR + '/is_compiling'):
            os.remove(NE_DIC_DIR + '/is_compiling')
    return {'result': True, 'msg': 'ne entity dic extend and compile success'}

def is_compiling():
    # 컴파일 중인 상태를 알려주는 파일 존재 여부 체크
    result = os.path.exists(NE_DIC_DIR + '/is_compiling')
    return result

# 개체명 사전 형성을 위한 검색 키워드 생성
def gen_prefix_array(input):
    search_iter = []
    search_index = []
    morph_result = morph_analyzer.analyze(input)
    morphs = morph_result['morphs']
    tags = morph_result['tags']
    LOGGER.debug('morphs={}'.format(morphs))
    LOGGER.debug('tags={}'.format(tags))

    try:
        for i in range(len(tags)):
            if tags[i][0] == 'E' or tags[i][0] == 'J':
                LOGGER.debug('tag is E or J, so continue')
                continue
            one_morph = ''
            for j in range(i, len(morphs)):
                one_morph += morphs[j]
                #분석을 위해 공백 추가 (마지막인 경우 제외)
                if j != len(morphs) - 1:
                    one_morph += ' '
            search_iter.append(one_morph)
            search_index.append((i, j))

            '''
            #공백이 제거한 경우도 추가
            one_morph = ''
            for j in range(i, len(morphs)):
                one_morph += morphs[j]
            search_iter.append(one_morph)
            search_index.append((i, j))
            '''

        # searchIndex에 idx,jdx대신 길이 정보를 넣어야 함
        # 이후 겹치는 것을 찾기 위한 처리
    except:
        pass

    LOGGER.debug('search_iter={}'.format(search_iter))
    LOGGER.debug('search_index={}'.format(search_index))
    return morphs, search_iter, search_index, morph_result


# 개체명 분석
def analyze(input):

    #만약 사전 컴파일 중이면 5초 기다려주기
    if is_compiling():
        time.sleep(5)

    compile_dic = marisa_trie.BytesTrie().load(COMPILED_NE_DIC_PATH)
    morphs, search_iter, search_index, morph_result = gen_prefix_array(input)

    # resultlist = []
    ne_list = []
    ne_tag_list = []
    longest = True
    for i in range(len(search_iter)):
        candi = search_iter[i]
        # print (self.searchIndex[idx])
        findlist = compile_dic.prefixes(candi)
        LOGGER.info('candi={}, findlist={}'.format(candi, findlist))
        if len(findlist) == 0:
            continue
        # 최장일치
        if longest:
            item = max(findlist, key=len)
            # print("item=", item)
            # for tag in compile_dic:
            if item in compile_dic and item not in ne_list:
                # values = [value.decode('utf-8')  for value in compile_dic[item]]
                # resultlist.append((item, values))
                for value in compile_dic[item]:
                    ne_list.append(item)
                    ne_tag_list.append(value.decode('utf-8'))

        # 최단일치
        else:
            for item in findlist:
                # print("item1=", item)
                # for tag in compile_dic:
                #    resultlist.append((item, tag))
                if item in compile_dic:
                    # values = [value.decode('utf-8') for value in compile_dic[item]]
                    # resultlist.append((item, values))
                    for value in compile_dic[item]:
                        ne_list.append(item)
                        ne_tag_list.append(value.decode('utf-8'))
    # 개체명 분석 결과 구조체
    ne_result = {
        'ne': ne_list
        , 'neTags': ne_tag_list
        , 'neNum': len(ne_list)
        , 'morphResult': morph_result
    }
    return ne_result

# 바이너리 사전에 등록된 개체명 추출, 없을 경우 None으로 돌려주기
def get_value(input):
    compile_dic = marisa_trie.BytesTrie().load(COMPILED_NE_DIC_PATH)
    results = compile_dic.get(input, [b'None'])
    results = [result.decode('utf-8') for result in results]
    return results


# for test
if __name__ == '__main__':
    #word = "인문학 공부법 강좌 학원"
    #words_without_spaces = get_words_without_space(word)
    #print(word)
    #print(words_without_spaces)
    extend_dic()


