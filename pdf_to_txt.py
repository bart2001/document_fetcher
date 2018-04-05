# coding: utf-8


#python 2.7.13
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import sys, getopt

#converts pdf, returns its text content as a string
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)
    #print "---------------",pagenums
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 
def search(dirname):
    filenames = os.listdir(dirname)
    pdfDirList = []
    for filename in filenames:
        #full_filename = os.path.join(dirname, filename)
        #pdfDirList += (full_filename)+('\n')
        pdfDirList.append((filename))
        #fileNameList.append((filename)+('/'))
    return pdfDirList


# In[ ]:




# # pdfPath:변환대상문서 (.pdf) 경로 지정
# # txtPath: 변환 출력 문서(.txt)경로지정
# #출력파일형태: .txt.#pagenum
# 

# In[13]:

#converts all pdfs in directory pdfDir, saves all resulting txt files to txtdir
from PyPDF2 import PdfFileWriter, PdfFileReader

def convertMultipleFile(pdfDir, txtDir):
    if pdfDir == "": pdfDir = os.getcwd() + "\\" #if no pdfDir passed in 
    for pdf in os.listdir(pdfDir): #iterate through pdfs in pdf directory
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = pdfDir + pdf 
            infile = PdfFileReader(open(pdfFilename, 'rb'))
            #of pages
            for i in xrange(infile.getNumPages()): 
                text = convert(pdfFilename,[i]) #get string of text content of pdf
                textFilename = txtDir + pdf +"."+str(i+1)+ ".txt"
                textFile = open(textFilename, "w") #make text file
                textFile.write(text) #write text to text file
def convertMultipleFileofDoc(pdfDir, txtDir):
    if pdfDir == "": pdfDir = os.getcwd() + "\\" #if no pdfDir passed in 
    for pdf in os.listdir(pdfDir): #iterate through pdfs in pdf directory
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = pdfDir + pdf 
            infile = PdfFileReader(open(pdfFilename, 'rb'))
            #of pages
            textFilename = txtDir + pdf +".txt"
            textFile = open(textFilename, "w") #make text file
            contents =""
            for i in xrange(infile.getNumPages()): 
                text = convert(pdfFilename,[i]) #get string of text content of pdf
                contents +=text
            textFile.write(contents) #write text to text file
            textFile.close()


# In[20]:

def main(argv):
    pdfPath = ''
    txtPath = ''
    
    try:
        opts, args = getopt.getopt(argv,"i:o:",["idir=","odir="])
    except getopt.GetoptError:
        print 'PDF_TO_TEXT.py -i <input_directory> -o <output_directory>'
        sys.exit(2)
    print opts
    for opt, arg in opts:
        if opt == '-h':
            print 'PDF_TO_TEXT.py -i <input_directory> -o <output_directory>'
            sys.exit()
        elif opt in ("-i", "--idir"):
            pdfPath = arg
        elif opt in ("-o", "--odir"):
            txtPath = arg
    
    #pdfPath = "/home/mini/data/KSAE_200701-201709/"
    #txtPath = "/home/mini/data/new_txt/"
    print pdfPath, txtPath
    if (os.path.isdir(txtPath)==False):
        print "folder created:",txtPath
        os.mkdir(txtPath)
    pdfDir_list = search(pdfPath)
    for dir_name in pdfDir_list:
        print dir_name
        pdfDir = pdfPath+str(dir_name)+'/'
        txtDir = txtPath
        convertMultipleFile(pdfDir, txtDir)
if __name__ == "__main__":
    if(len(sys.argv)!=5):
        print " ex >>>>>>>>>>>>python PDF_TO_TEXT.py -i /home/mini/data/KSAE_200701-201709/ -o /home/mini/txt/"
    args = sys.argv[1:]
    main(args)

