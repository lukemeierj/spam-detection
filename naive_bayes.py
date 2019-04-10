from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def runNBs(features, labels):
    classifiers = [
                ("Guassian", GaussianNB()), 
                ("Multinomial", MultinomialNB()),
                ("Compliment", ComplementNB())
              ]

    for name, nb in classifiers:
        print("Running with {} Naive Bayes".format(name))
        runNB(nb, features, labels, scoringFns=[accuracy_score, precision_score, recall_score])
        print()


def runNB(nb, features, labels, scoringFns=[accuracy_score]):
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, random_state=6)

    # Fit
    nb.fit(feature_train, label_train)

    # Predict
    label_pred = nb.predict(feature_test)

    # Accuracy
    for fn in scoringFns:
        print("{}: {}".format(fn.__name__, fn(label_test, label_pred)))