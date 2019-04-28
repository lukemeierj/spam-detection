from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from binning import *

best_log_bin_size = 7

classifiers = [
              #("Guassian", GaussianNB()), 
              ("Multinomial", MultinomialNB()),
              ("Compliment", ComplementNB())
            ]

def nb_optimize_bin_num(nb, features, labels, scoringFn):
    best = (-1, 0)
    for i in range(1, 30):    
        data = features.copy()
        data = binned_data(data, range(0, 48), logarithmic_bin, num_bins = i)
        results = runNB(nb, data, labels, scoringFns=[scoringFn])
        if best[1] < results[0][1]:
            best = (i, results[0][1])
    return best

def optimizeNBs(features, labels):
    for name, nb in classifiers:
        best = nb_optimize_bin_num(nb, features, labels, precision_score)
        print("Best bin size for {} is {} with {} precision".format(name, best[0], best[1]))


def runNBs(features, labels):
    
    for name, nb in classifiers:
        print("\tRunning with {} Naive Bayes\n".format(name))
        for binning_fn in [linear_bin, logarithmic_bin, None]:
            data = features.copy()
            if binning_fn is not None: 
                data = binned_data(data, range(0, 48), binning_fn, num_bins = best_log_bin_size)

            bin_name = binning_fn.__name__ if binning_fn is not None else "no"
            print("\t\tusing {} binning".format(bin_name))
            runNB(nb, data, labels, scoringFns=[accuracy_score, precision_score, recall_score])
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


