import requests
from contextlib import closing
import csv
import re
import numpy as np
from nltk import word_tokenize
from scipy.optimize._tstutils import f1
from sklearn import ensemble, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

ls=[]
url="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"

def read_tsv():
    with closing(requests.get(url, stream=True)) as r:
        f = (line.decode('utf-8') for line in r.iter_lines())
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            x=row
            if x !=[]:
                ls.append(x)
    #ls.pop()
    print(ls)
    train_ls_text=[]
    train_ls_names=[]
    test_ls_text=[]
    test_ls_names=[]
    validation_ls_text=[]
    validation_ls_names=[]
    for j in ls:
        if j[1]=='training':
            train_ls_text.append(j[3])
            train_ls_names.append(j[2])
        elif j[1]=='testing':
            test_ls_text.append(j[3])
            test_ls_names.append(j[2])
        elif j[1]=='validation':
            validation_ls_text.append(j[3])
            validation_ls_names.append(j[2])

    for i in validation_ls_text:
        train_ls_text.append(i)
    for j in validation_ls_names:
        train_ls_names.append(j)
    #print(len(train_ls_text))
    for k in test_ls_text:
        train_ls_text.append(k)
    for g in test_ls_names:
        train_ls_names.append(g)
    return train_ls_text,train_ls_names,test_ls_text
    #print(len(train_ls_names))

def features(train_ls_text,train_ls_names):

    ls1=[]
    redacted_name_count=[]
    for i in train_ls_text:
         redacted_names=re.findall(r'\█+' , i)
         ls1.append(redacted_names)
    #print(ls1)
    for i in range(len(ls1)):
        if len(ls1[i])==1:
            for j in ls1[i]:
                #print(len(ls1[i]))
                count=len(j)
                redacted_name_count.append(count)
        elif len(ls1[i]) >1:
            train_ls_text.remove(train_ls_text[i])
            train_ls_names.remove(train_ls_names[i])

    ## Feature
    space_count=[]
    sp_list=[]
    for i in train_ls_names:
        p=i.split(" ")
        #print(p)
        space_count.append(len(p)-1)
    #print(len(space_count))

    word_count=[]
    for i in train_ls_text:
        words=word_tokenize(i)
        word_count.append(len(words))

    ## Feature
    sentiment=[]
    for i in train_ls_text:
        blob=TextBlob(i)
        #print(blob.sentiment)
        if blob.sentiment[0] >= 0:
            sentiment.append(1)
        else:
            sentiment.append(0)

    dict={ }
    features_list=[]

    for i in range(len(train_ls_text)):
        dict={'names_count': len(train_ls_names[i]),'word_space_count':space_count[i],'words_count':word_count[i],'sentiment': sentiment[i],'review':train_ls_text[i]}
        features_list.append(dict)
    return features_list

def vectorization(list,test_ls_text,train_ls_text,train_ls_names):

    DV = DictVectorizer()
    feature_array = DV.fit_transform(list)
    X=feature_array.toarray()
    length1=len(train_ls_text)-len(test_ls_text)
    X_train=X[0:length1]
    test=X[-(len(test_ls_text)):]
    y=np.asarray(train_ls_names)
    y_train=y[0:length1]
    y_test=y[-(len(test_ls_text)):]
    return X_train,test,y_train,y_test

def train_data(X_train,test,y_train,y_test):

    model = GaussianNB()
    #model = ensemble.RandomForestClassifier()
    m=model.fit(X_train,y_train)
    pred=m.predict(test)

    #print(pred)
    precision_score=metrics.precision_score(y_test, pred, average='weighted', labels=np.unique(pred))
    print("Precision_score:",precision_score)
    print("Recall_score:",metrics.recall_score(y_test,pred,average='weighted',labels=np.unique(pred)))
    print("f1 score:", metrics.f1_score(y_test,pred,average='weighted',labels=np.unique(pred)))

    return int(precision_score)


if __name__ == '__main__':
    train_ls_text, train_ls_names,test_ls_text = read_tsv()
    features_list=features(train_ls_text,train_ls_names)
    X_train, X_test, y_train, y_test= vectorization(features_list,test_ls_text,train_ls_text,train_ls_names)
    train_data(X_train,X_test,y_train,y_test)




