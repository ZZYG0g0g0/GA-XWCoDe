"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/12
Last Updated: 2023/11/21
Version: 1.0.0
"""

from imblearn.over_sampling import SMOTE


#SMOTE过采样
def smote(x,y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(x, y)
    return X_res, y_res



