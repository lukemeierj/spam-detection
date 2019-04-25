import tensorflow
import pandas as pd
import numpy
import naive_bayes

def main():
    df = pd.read_csv("./spambase/spambase.data")

    # Do we want to use a range 0,100 or 0,1?
    # for col in df.columns:
    #     if "freq" in col:
    #         df["word_freq_people"] *= 100

    target = df["spam"].values
    data = df.drop("spam", 1).values
    
    

    naive_bayes.runNBs(data, target)

    
    


if __name__ == "__main__":
    main()