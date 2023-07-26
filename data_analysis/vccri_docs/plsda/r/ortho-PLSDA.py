import pandas as pd
import subprocess
import os


df = pd.read_csv("two_samples.csv", sep=",", index_col="unique_id")

# df.T.to_csv("two_t.csv")

df.drop(labels="label").to_csv("in.csv")
df[:1].T.to_csv("in_g.csv", )

subprocess.run("Rscript plsda.r", stdout= open(os.devnull, "w"), shell=True)
subprocess.run("Rscript random_forest.r", stdout= open(os.devnull, "w"), shell=True)

def getScoreDf():
    # x = p1 y = 01
    return pd.read_csv("score.csv", sep=',')

def getVipDf():
    # x = G y = VIP
    return pd.read_csv("vip.csv")

if __name__ == "__main__":
    pass
    # print(getScoreDf())
    # print(getVipDf())