import os, sys, platform
os_type = platform.system()
if (os_type == 'Windows'):
    import MeCab as Mecab_windows
elif (os_type == 'Linux'):
    from konlpy.tag import Mecab as Mecab_linux

def pos_analysis(input):
    result = []
    # 윈도우에서 분석할 경우
    if os_type == 'Windows':
        tagger = Mecab_windows.Tagger()
        node = tagger.parseToNode(input)
        while node:
            key = node.surface
            value = node.feature.split(',')[0]
            pair = (key, value)
            if not 'BOS/EOS' in value:
                result.append(pair)
            node = node.next
    #리눅스에서 분석할 경우
    elif os_type == 'Linux':
        result = Mecab_linux.pos(input)
    return result


if __name__ == '__main__':
    result = pos_analysis('무궁화 꽃이 피었습니다.')
    print(result)
