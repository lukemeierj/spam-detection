import tensorflow
import pandas as pd
import numpy
import naive_bayes
import neural_net

def main():
    df = pd.read_csv("./spambase/spambase.data")

    target = df["spam"].values
    data = df.drop("spam", 1).values
    
    naive_bayes.runNBs(data, target)
    neural_net.runNNs(data, target)


if __name__ == "__main__":
    main()