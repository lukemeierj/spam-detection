import tensorflow
import pandas as pd
import numpy
import naive_bayes

def main():
    df = pd.read_csv("./spambase/spambase.data")

    target = df["spam"].values
    data = df.drop("spam", 1).values
    
    naive_bayes.runNBs(data, target)


if __name__ == "__main__":
    main()