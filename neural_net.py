from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def runNNs(features, labels):
    
    scoringFns=[accuracy_score, precision_score, recall_score]
    
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 8), random_state=1)
    
    trainFeature, testFeature, trainLabel, testLabel = train_test_split(features, labels, random_state=6)
    
    nn.fit(trainFeature, trainLabel)
    
    predLabel = nn.predict(testFeature)
    
    for fn in scoringFns:
        print("{}: {}".format(fn.__name__, fn(testLabel, predLabel)))

