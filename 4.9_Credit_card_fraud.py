import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

card_df = pd.read_csv("/MLguide/creditcard.csv")
card_df.head()