"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/3/25
Last Updated: 2024/3/25
Version: 1.0.0
"""
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../../'))

from ML.machine_learning import xgboost_classification_kfold, XWCoDe, calculate_dependency_importance, XGB_CD, get_dependency_matrix
from find_parameter.best_param_optim.best_parameter_train_2 import  best_parameter_test_kfold, best_parameter_test_kfold_all_results

if __name__ == '__main__':
    # 初始化一个空的DataFrame用于存储结果
    results_list = []

    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    for dataset in datasets:
        ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        best_individual_eTour = [0.356118737, 7.609326525, 0.888211172, 0.124011689, 0.224363367, 0.217800669]
        best_individual_eANCI = [0.337776485, 7.349362195, 0.961965044, 0.257936369, 0.305167627, 0.470519603]
        best_individual_iTrust = [0.086766218, 7.902444804, 0.63931166, 0.079436777, 0.081968587, 0.206560853]
        best_individual_smos = [0.27895015, 9.247654594, 0.961642787, 0.069235715, 0.339326463, 0.317987252]
        if dataset == 'eTour':
            results_df = best_parameter_test_kfold_all_results(best_individual_eTour, dataset, "selected_features", ci, mi, wi,dependency_matrix)
        elif dataset == 'eANCI':
            results_df = best_parameter_test_kfold_all_results(best_individual_eANCI, dataset, "selected_features", ci, mi, wi,dependency_matrix)
        elif dataset == 'iTrust':
            results_df = best_parameter_test_kfold_all_results(best_individual_iTrust, dataset, "selected_features", ci, mi, wi, dependency_matrix)
        elif dataset == 'smos':
            results_df = best_parameter_test_kfold_all_results(best_individual_smos, dataset, "selected_features", ci, mi, wi, dependency_matrix)

        # 将列表转换为DataFrame
        results_df.to_excel(f'AC_XWCoDe_results1_{dataset}.xlsx', index=False)
