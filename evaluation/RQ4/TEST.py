import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Data for Precision, Recall, and F1 Score
    precision = [0.645, 0.6254, 0.6185, 0.5401, 0.5715, 0.5561, 0.5375, 0.4588, 0.3819]
    recall = [0.5701, 0.5859, 0.5346, 0.5012, 0.4473, 0.4309, 0.4366, 0.4165, 0.3072]
    f1 = [0.6052, 0.605, 0.5735, 0.5199, 0.5018, 0.4856, 0.4818, 0.4366, 0.3405]

    # Convert lists to numpy arrays for variance calculation
    precision_array = np.array(precision)
    recall_array = np.array(recall)
    f1_array = np.array(f1)

    # Calculate variances
    precision_variance = np.var(precision_array)
    recall_variance = np.var(recall_array)
    f1_variance = np.var(f1_array)

    print(precision_variance)
    print(recall_variance)
    print(f1_variance)