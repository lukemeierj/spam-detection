from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from binning import *


def runNBs(features, labels):
    classifiers = [
              ("Guassian", GaussianNB()), 
              ("Multinomial", MultinomialNB()),
              ("Compliment", ComplementNB())
            ]

    for name, nb in classifiers:
        print("\tRunning with {} Naive Bayes\n".format(name))
        for binning_fn in [linear_bin, logarithmic_bin, None]:
            if binning_fn is not None: 
                features = binned_data(features, range(0, 48), binning_fn)

            bin_name = binning_fn.__name__ if binning_fn is not None else "no"
            print("\t\tusing {} binning".format(bin_name))
            runNB(nb, features, labels, scoringFns=[accuracy_score, precision_score, recall_score])
            print()


def runNB(nb, features, labels, scoringFns=[accuracy_score]):
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, random_state=6)

    # Fit
    nb.fit(feature_train, label_train)
    

    # Predict
    label_pred = nb.predict(feature_test)

    results = [ (fn.__name__, fn(label_test, label_pred)) for fn in scoringFns ] 

    # Accuracy
    for metric, score in results:
        print("\t\t\t{}: {}".format(metric, score))

    return results

