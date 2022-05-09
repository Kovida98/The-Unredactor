import numpy as np
import pytest
import unredactor
train_ls_text, train_ls_names, test_ls_text =unredactor.read_tsv()
features_list = unredactor.features(train_ls_text,train_ls_names)
X_train,test, y_train, y_test= unredactor.vectorization(features_list,test_ls_text,train_ls_text,train_ls_names)
precision_score=unredactor.train_data(X_train,test,y_train,y_test)
def test_readtsv():

    assert train_ls_text is not None
    assert train_ls_names is not None
    assert test_ls_text is not None

def test_features():

    assert features_list is not None
    assert type(features_list) == type([ ])
    for i in features_list:
        assert type(i['names_count']) == int

def test_vectorization():

    assert X_train is not None
    type(X_train).__module__
    assert type(X_train).__module__ == np.__name__
    #print(type(X_train))

def test_train_data():
    assert type(precision_score) == int