# from lightgbm.sklearn import LGBMRegressor
#
# from lightgbm.callback import early_stopping
# earlyStopping = early_stopping(stopping_rounds=200, verbose=True)
#
#
#
#
# model = LGBMRegressor()
#
#
# model.fit(callbacks=[earlyStopping])
#
#
#
# import os, cv2, matplotlib.pyplot as plt
# for dirname, _, filenames in os.walk('../input/aptos2019-blindness-detection/train_images'):
#     for filename in filenames:
#         if filename == 'f58d37d48e42.png':
#             img = cv2.imread(os.path.join('../input/aptos2019-blindness-detection/train_images', filename))
#             plt.imshow(img)
#
#





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#
# # df_sub = pd.read_csv('../input/submission-e-net-best-weight/submission_E_Net_best_weight.csv')
# # df_sub = pd.read_csv('../input/sample-submission-0796/submission_0_796.csv')
# df_sub = pd.read_csv('../input/submission/submissionAptos.csv')
# sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
#
# df_merged = pd.merge(sample_submission, df_sub, on='id_code', how='left')
# df_merged.fillna(0, inplace=True)
# df_merged['diagnosis'] = df_merged['diagnosis_y']
# df_merged = df_merged[['id_code', 'diagnosis']]
# df_merged['diagnosis'] = df_merged['diagnosis'].astype('int64')
#
# df_merged.to_csv('submission.csv', index=False)

#
# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
#
# bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)


%reload_ext autoreload
%autoreload 2
%matplotlib inline
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from torchvision import models as md
from torch import nn
from torch.nn import functional as F
import re
import math
import collections
from functools import partial
from torch.utils import model_zoo
from sklearn import metrics
from collections import Counter
import json


learn = Learner(data,
                md_ef,
                metrics = [qk],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '/content',
                                      folder='test_images',
                                      suffix='.png'))


learn.fit_one_cycle

