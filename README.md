## THE UNREDACTOR

Whenever sensitive information is shared with the public, the data must
go through a redaction process. That is, all sensitive names, places, 
and other sensitive information must be hidden. Documents such as police
reports, court transcripts, and hospital records all containing sensitive 
information. Redacting this information is often expensive and time 
consuming.

For project 3,  i am creating an unredactor. The unredactor will 
take redacted documents and return the most likely candidates to fill in
the redacted location. The unredactor only needs to unredact people
names.In this project my unredactor only unredacts names.

As you can guess, discovering names is not easy. To discover the best 
names, i trained a model to predict missing words. 
For this assignment, i am using the unredactor.tsv file as input.
unredactor.tsv file contains the data of Movie reviews and the redacted person names.
For this project, we will only use the reviews for their textual content.

## AUTHOR:

Kovida Mothukuri

Author Email: kovida.mothukuri-1@ou.edu

## PACKAGES USED:
1. csv
2. re
3. numpy
4. nltk
5. sklearn
6. pytest
7. warnings
8. TextBlob
9. import requests
10. from contextlib import closing

## FUNCTIONS:

1. **read_tsv():**

In this function, i took the unredactor.tsv url as input. I separated the
data which is in unredactor.tsv file. I have splitted the whole data in to 
train and test data.The data which has label as training and validation,
i used that data for training. The data which has label as testing , i used that 
data for testing.

I took all the training and validation data in to `train_ls_text` and `train_ls_names`.
train_ls_text has all the reviews and train_ls_names contains all the names.
Then, I appended all the test data reviews and names at the end of this train_ls_text list and 
train_ls_names list respectively. This function returns three lists named
`train_ls_text`,`train_ls_names`,`test_ls_text`.

2. **features(train_ls_text,train_ls_names):**

This function takes train_ls_text and train_ls_names as input parameters.
I have considered few features to implement this project.One is the length of the
person names,space count between the person names and the count of the words
in review sentence.Other feature is i calculated sentiment polarity for each review
sentence and took that if the polarity is greater than zero then i took the sentence
as positive and took as 1.Else i took the sentence as negative and took as 0. I appended
all these person names length, space count, words count in review text and sentiment
polarities in to seperate lists. I kept all values of those lists in to dictionary 
and appended all those dictionaries in to a list named `features_list`.This function
returns this `features_list`.

3. **vectorization(list,test_ls_text,train_ls_text,train_ls_names):**

This function takes the features_list,test_ls_text,train_ls_text,train_ls_names
as input parameters.In this function, i used dict vectorizer to vectorize the
data. I vectorized the features_list which contains the vectorized data for
training,validation,testing data.Now, i splitted the data in to two parts 
according to the length of test data which results in one trained vectorized 
data and test vectorized data named `X_train`,`test`.So now, X_train contains the
vectorized form of training and validation data and test contains 
vectorized form of test data. I changed the `train_ls_names` in to array
by using numpy named `y`.Again I splitted `y` in to two parts according to the
length of test data named `y_train`,`y_test` where y_train contains array 
of the person names of training,validation data and y_test contains the
array of the person names of test data. This function returns `X_train`,`test`,
`y_train`,`y_test`.

4. **train_data(X_train,test,y_train,y_test):**

This function takes X_train,test,y_train,y_test as input parameters.
I used Gaussian Naive Bayes model to train the data. I passed X_train and y_train
data in to `model.fit()` function to fit the data.To predict the person names,
i used `.predict` function and passed `test` in to it. This will give you the
list of predicted person names named as `pred`. I calculated precision,
recall,f1 scores for y_test and pred. This function prints Precision score, recall
score, f1 score.

## TEST CASES:

For test cases, i created a file named `test_all.py`.In that file,
I wrote four functions which tests all the four functions which are 
written in unredactor.py file.

1. **test_readtsv():**

This function tests the `read_tsv()` function which returns the train_ls_text, train_ls_names, test_ls_text.
I wrote an assert statement to check whether the train_ls_text, train_ls_names, test_ls_text
is not none.

2. **test_features():**

This function tests the `features(train_ls_text,train_ls_names)` 
function which returns the features_list. I wrote an assert statement 
to check whether the features_list is not None, checks the type of
features_list is a dictionary and checks whether the names_count in the 
features_list is an integer or not.

3. **test_vectorization():**

This function tests the `vectorization(features_list,test_ls_text,train_ls_text,train_ls_names)` 
function which returns the X_train,test, y_train, y_test.
I wrote an assert statement to check whether the X_train is not None and
checks the type of X_train is numpy array or not.

4. **test_train_data():**

This function tests the `train_data(X_train,test,y_train,y_test)` 
function which returns the precision score. I wrote an assert statement
to check whether the type of precision score is integer or not.

## COMMANDS TO RUN:

Here to run the file we have to use below command:

`pipenv run python unredactor.py `

To run test cases we can use any one of the following commands

`pipenv run python -m pytest`

**Expected Output:**

```
Precision_score: 0.2358974358974359

Recall_score: 0.5384615384615384

f1 score: 0.3090418353576248
```

## DIRECTIONS TO INSTALL:

You can create folders and files using mkdir and touch commands.
Here in this project we will be using python 3.10.2 version. to install that use this command.

`pipenv install --python 3.10.2`

After downloading the project from github, go to that directory using cd.Install pipenv by using
command. `pip install pipenv`. After that by checking requirements.txt file, you have to install all
required packages.  you need to install pytest using this command `pipenv install pytest`.Once the installation of pytest is done, you will be able to
run the unittests using `pipenv run python -m pytest`. 

you can run the code using
`pipenv run python unredactor.py`.

## ASSUMPTIONS AND BUGS:

I am assuming that the input file that is unredactor.tsv file  is cleaned data.

I am assuming that the review text contains only one redacted word in the sentence.

The second column values in unredactor.tsv file should be training, validation,testing 
only. training,validation,testing always should be in lower case otherwise my code will not
work much efficiently.


## EXTERNAL LINKS USED:

[https://www.codegrepper.com/code-examples/python/how+to+read+a+csv+file+from+a+url+with+python](https://www.codegrepper.com/code-examples/python/how+to+read+a+csv+file+from+a+url+with+python)


[https://www.programcreek.com/python/example/92914/sklearn.feature_extraction.DictVectorizer](https://www.programcreek.com/python/example/92914/sklearn.feature_extraction.DictVectorizer)


[https://www.codespeedy.com/naive-bayes-algorithm-in-python/](https://www.codespeedy.com/naive-bayes-algorithm-in-python/)


[https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi](https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi)












