import pandas as pd

wordList = {"beg"}
commonWords = {}

def toAlphaNum(text):
    docText = text.split()
    docText = [x.lower().strip() for x in docText if x.isalpha()]
    for word in docText:
        wordList.add(word)
    return docText

def lowercase(text):
    return text.lower()

def parseText(textFile):
    documents = pd.read_csv(textFile)
    documentText = documents[['ABSTRACT']]
    documentText = documentText.applymap(toAlphaNum)
    minDocLength = 10000
    for list in documentText:
        if(len(list)<minDocLength):
            minDocLength = len(list)
    return [documentText,wordList,minDocLength]

#print(parseText('test.csv')[0])
#print(len(parseText('test.csv')[1]))
