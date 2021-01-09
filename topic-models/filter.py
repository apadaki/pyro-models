import pandas as pd
import nltk

wordList = {}
commonWords = {}
wordCount = 0
minDocLength = 140
#nltk.download('stopwords')
#nltk.download('wordnet')

en_stop = set(nltk.corpus.stopwords.words('english'))

def get_lemma(word):
    lemma = nltk.corpus.wordnet.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def toAlphaNum(text):
    docText = text.split()
    docText = [x.lower().strip() for x in docText if x.isalpha()]
    docText = [word for word in docText if word not in en_stop]
    docText = [get_lemma(word) for word in docText]
    return docText

def hash(text):
    global wordCount
    for word in text:
        if word not in wordList:
            wordCount += 1
            wordList[word] = wordCount
    text = [int(wordList[word]) for word in text]
    return text

def parseText(textFile):
    global minDocLength
    documents = pd.read_csv(textFile)
    documentText = documents[['ABSTRACT']]
    documentText = documentText.applymap(toAlphaNum)
    documentText = [list for list in documentText['ABSTRACT'] if len(list) >= minDocLength]
    documentText = [hash(list) for list in documentText]
    documentText = [list[0:minDocLength] for list in documentText]
    return [documentText,wordList,minDocLength]

def printResults(textFile):
    data = parseText(textFile)
    #print(data[0]['ABSTRACT'][1337])
    #print(len(data[0]['ABSTRACT'][1337]))
    print(len(data[0]))
    print(len(data[1]))
    print(data[2])

#printResults('test.csv')
