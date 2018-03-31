# 2어절 이상의 단어를 넣을 경우 모든 경우의 수대로 공백을 제거한 리스트까지 제공해주는 함수
#  공부 방법 -> [공부 방법, 공부방법]

import re
import itertools


if __name__ == '__main__':

    word = "인문학 공부법 강좌 학원"
    spaces = [m.start() for m in re.finditer(' ', word)]

    #print(spaces)

    combinations = []
    for i in range(1, len(spaces) + 1):
        combination = list(itertools.combinations(spaces, i))
        #print(combination)
        combinations.extend(combination)

    #print(combinations)
    words = [word]


    new_words = []
    for spaces in combinations:
        new_word = ''
        for i in range(len(word)):
            if (i in spaces):
                continue
            else:
                new_word += word[i]
        #print(spaces)
        #print(new_word)
        new_words.append(new_word)

    print(word)
    print(new_words)








