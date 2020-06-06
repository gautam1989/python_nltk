from statistics import mode

import nltk
import random
from nltk.corpus import names
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC

class VoteClassifier(nltk.ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def gender_features(word):
    return {'last_letter': word[-1]}


labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = nltk.SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set)) * 100)

# print(classifier.classify(gender_features('Neo')))
# print(MNB_classifier.classify(gender_features('Neo')))

BernoulliNB_classifier = nltk.SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set)) * 100)

LogisticRegression_classifier = nltk.SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, test_set)) * 100)

SGDClassifier_classifier = nltk.SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier_classifier accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, test_set)) * 100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(train_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

LinearSVC_classifier = nltk.SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set)) * 100)

NuSVC_classifier = nltk.SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)

voted_classifier = VoteClassifier(
    NuSVC_classifier,
    LinearSVC_classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set)) * 100)

print(voted_classifier.classify(gender_features("gaga")))
print(voted_classifier.confidence(gender_features("gaga")))