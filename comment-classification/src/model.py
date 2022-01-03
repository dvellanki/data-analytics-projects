import numpy as np
import pandas as pd
import pickle

from my_features import PoS_TagFeatures, BadWords_Features, Symbol_Features, TextFeatures
from data_process import Data, get_classes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from scipy import interp
import matplotlib.pyplot as plt

def feature_extraction(data, flag):
    # Word Vectorizer
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 1),
        max_features=20000)

    # N-gram Character Vectorizer
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(1, 4),
        max_features=30000)

    # Pipelining Parts of Speech Tag Features with DictVectorizer for processing
    posTag_vectorizer = Pipeline([
        ('parts_of_speech', PoS_TagFeatures()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Bad Word Features with DictVectorizer for processing
    badWord_vectorizer = Pipeline([
        ('bad_words', BadWords_Features()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Symbol based Features with DictVectorizer for processing
    symbol_vectorizer = Pipeline([
        ('symbols', Symbol_Features()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Text Features with DictVectorizer for processing
    text_vectorizer = Pipeline([
        ('texts', TextFeatures()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    print ("Extracting features...")
    # combined_features = FeatureUnion(
    #     [("word", word_vectorizer), ("char", char_vectorizer), ("pos_tags", posTag_vectorizer),
    #      ("bad_word", badWord_vectorizer), ("symbol", symbol_vectorizer), ("text", text_vectorizer)])


    combined_features = FeatureUnion(
         [("word", word_vectorizer), ("char", char_vectorizer)])


    if(flag == 'train'):
        features = combined_features.fit(data.train_text).transform(data.train_text)
        print ("Saving features")
        feature_pkl_filename = '../model/features.pkl'
        feature_pkl = open(feature_pkl_filename, 'wb')
        pickle.dump(combined_features, feature_pkl)
        feature_pkl.close()
        print ("Features saved")

    if (flag == 'test'):
        print ("Loading features")
        feature_pkl = open('../model/features.pkl', 'rb')
        loaded_features = pickle.load(feature_pkl)
        print ("Loaded features :: ", loaded_features)
        features = loaded_features.transform(data.test_text)
    return features

def create_and_save():
    data = Data()
    train_features = feature_extraction(data, "train")
    scores = []
    # print (train_features.shape)
    # kbest = SelectKBest(chi2, k=1000)
    for i in range(len(data.classes)):
        print ("Processing "+data.classes[i])
        train_target = data.train[data.classes[i]]
        # x_feature = kbest.fit_transform(train_features, train_target)
        # print (x_feature)
        classifier = LogisticRegression(solver='sag')
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        # cv_score = np.mean(cross_val_score(classifier, x_feature, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(data.classes[i], cv_score))

        # Calculate ROC_AUC
        roc_auc(train_features, np.array(train_target), data.classes[i])

        print ("Creating model for class "+data.classes[i])
        classifier.fit(train_features, train_target)
        # classifier.fit(x_feature, train_target)

        print ("Saving model logistic_regression_%s" %data.classes[i])
        lr_pkl_filename = '../model/logistic_regression_%s.pkl' %data.classes[i]
        lr_model_pkl = open(lr_pkl_filename, 'wb')
        pickle.dump(classifier, lr_model_pkl)
        lr_model_pkl.close()
        print ("Model saved")
    print('Total CV score is {}'.format(np.mean(scores)))
    print ("Successfully created and saved all models!")


def predict_score():
    data = Data()
    test_features = feature_extraction(data, "test")
    submission = pd.DataFrame.from_dict({'id': data.test['id']})
    for i in range(len(data.classes)):
        print ("Processing "+data.classes[i])
        lr_model_pkl = open('../model/logistic_regression_%s.pkl' %data.classes[i], 'rb')
        lr_model = pickle.load(lr_model_pkl)
        print ("Loaded Logistic Regression Model for class %s :: " %data.classes[i], lr_model)
        submission[data.classes[i]] = lr_model.predict_proba(test_features)[:, 1]
    print (submission.head(5))
    print ("Saving output")
    submission.to_csv('../data/output.csv', index=False)
    print ("Output saved")


def predict_individual_score(comment):
    print ("Loading features")
    feature_pkl = open('../model/features.pkl', 'rb')
    loaded_features = pickle.load(feature_pkl)
    print ("Loaded features :: ", loaded_features)
    comment_list = []
    comment_list.append(comment)
    comment_features = loaded_features.transform(comment_list)
    prediction = pd.DataFrame()
    classes = get_classes()
    for i in range(len(classes)):
        print ("Processing "+classes[i])
        lr_model_pkl = open('../model/logistic_regression_%s.pkl' %classes[i], 'rb')
        lr_model = pickle.load(lr_model_pkl)
        print ("Loaded Logistic Regression Model for class %s :: " %classes[i], lr_model)
        prediction[classes[i]] = lr_model.predict_proba(comment_features)[:, 1]
    # print ("Prediction:")
    # print (prediction)
    return prediction


def roc_auc(X, Y, clas):
    classifier = LogisticRegression(solver='sag')
    cv = StratifiedKFold(Y, n_folds=5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    # print(X.shape, Y.shape)
    for i, (tran, tet) in enumerate(cv):
        # print(tran, tet)
        probas_ = classifier.fit(X[tran, :], Y[tran]).predict_proba(X[tet, :])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y[tet], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics - %s' %clas)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('testplot%s.png' %clas)
    plt.close()

© 2022 GitHub, Inc.
