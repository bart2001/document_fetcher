import pymysql
import multiprocessing
import os

PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATE_DIR = os.path.join(PROJECT_HOME, 'data', 'keyword')
SELECT_QUERY = """
    SELECT 'ID' AS ID, 'CONTENT' AS CONTENT
    FROM DUAL
"""

def fetch_and_write(select_query, file_name):

    print(f'fetching and writing starting={select_query},{file_name}')

    conn = pymysql.connect(
        host='localhost'
        , user='root'
        , password='system'
        , db='mobi_chat'
        , charset='utf8'
    )
    curs = conn.cursor(pymysql.cursors.DictCursor)
    curs.execute(select_query)

    #print(type(curs))

    with open(file_name, 'w', encoding='utf-8') as file:
        row = curs.fetchone()
        for i in range(1000):
            doc_id = row['ID']
            doc_content = row['CONTENT']
            index = i
            line_to_write = f'{file_name},{index}\n'
            file.write(line_to_write)

    return

def make_file_ranges(start, end, gap):
    file_ranges = []
    #size = end - start

    for i in range(start, end, gap):
        file_ranges.append(i)

    last = file_ranges[-1]
    file_ranges.append(last + gap)
    return file_ranges

def make_file_names(file_dir, prefix, file_ranges):
    file_names = []
    for i in range(0, len(file_ranges)-1):
        start = file_ranges[i]
        end = file_ranges[i+1]
        file_name = '{}_{}_{}.keyword'.format(prefix, start, end)
        file_path = os.path.join(file_dir, file_name)
        file_names.append((start, end, file_path))
    return file_names

def extract_keyword(start, end, file_path):

    fetch_and_write()


    return


if __name__ == '__main__':

    #print(PROJECT_HOME)
    print('fetching and writing start')

    file_ranges = make_file_ranges(0, 10000, 1000)

    file_names = make_file_names(os.path.join(PROJECT_HOME, 'data', 'keyword', 'edims'), 'edims', file_ranges)

    for start, end, file_name in file_names:
        t = multiprocessing.Process(target=fetch_and_write, args=(SELECT_QUERY, file_name))
        t.start()
        t.join()




