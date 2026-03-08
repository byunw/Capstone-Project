from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/Users/byeon-useog/desktop/training2017/REFERENCE.csv",header=None,names=["record","label"])

assert df["record"][0] == "A00001"
assert df["record"][8527] == "A08528"
assert df["label"][0]=="N"
assert df["label"][8527]=="N"

train_df, temp_df = train_test_split(
    df, test_size = 0.30,
    stratify=df["label"],random_state=42
)

val_df, test_df = train_test_split(
    temp_df,test_size = 0.50,
    stratify=temp_df["label"],random_state=42
)

#test code
assert len(df["label"])==8528
assert len(train_df["label"])==5969
assert len(temp_df["label"])==2559
assert len(val_df["label"])==1279
assert len(test_df["label"])==1280

#pick up from here these 3 lines of code
    # train_df.to_csv("train_split.csv",index=False)
    # val_df.to_csv("val_split.csv",index=False)
    # test_df.to_csv("test_split.csv",index=False)

















