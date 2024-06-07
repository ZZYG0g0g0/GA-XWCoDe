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

from ML.machine_learning import  calculate_dependency_importance
from find_parameter.best_param_optim.best_parameter_train_2 import best_parameter_test_kfold

if __name__ == '__main__':
    # 初始化一个空的DataFrame用于存储结果
    results_list = []

    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    dataset1 = [ 'eTour']
    for dataset in dataset1:
        # p_XGB, r_XGB, f1_XGB = xgboost_classification_kfold(dataset, "selected_features") #XGBoost
        # print(f'Precision_XGBoost:{dataset}->{p_XGB}, Recall_XGBoost:{dataset}->{r_XGB}, F1_XGBoost:{f1_XGB}')

        # dependency_matrix = get_dependency_matrix(dataset,datasets)  #XCoDe
        # p_XCoDe, r_XCoDe, f1_XCoDe = XWCoDe(dataset, "selected_features", 1/3, 1/3, 1/3, dependency_matrix) #XCoDe
        # print(f'Precision_XCoDe:{dataset}->{p_XCoDe}, Recall_XCoDe:{dataset}->{r_XCoDe}, F1_XCoDe:{dataset}->{f1_XCoDe}')

        # ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets) #XWCoDe
        # p_XWCoDe, r_XWCoDe, f1_XWCoDe = XWCoDe(dataset, "selected_features", ci, mi, wi, dependency_matrix)
        # print(f'Precision_XWCoDe:{dataset}->{p_XWCoDe}, Recall_XWCoDe:{dataset}->{r_XWCoDe}, F1_XWCoDe:{dataset}->{f1_XWCoDe}')


        # ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        # f1_AC_XCoDe, precision_AC_XCoDe, recall_AC_XCoDE = xgboost_with_optim_parameter_best(dataset, "selected_features", 1/3, 1/3, 1/3, dependency_matrix)

        ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        best_individual_eTour = [0.356118737, 7.609326525, 0.888211172, 0.124011689, 0.224363367, 0.217800669]
        best_individual_eANCI = [0.337776485, 7.349362195, 0.961965044, 0.257936369, 0.305167627, 0.470519603]
        best_individual_iTrust = [0.086766218, 7.902444804, 0.63931166, 0.079436777, 0.081968587, 0.206560853]
        best_individual_smos = [0.27895015, 9.247654594, 0.961642787, 0.069235715, 0.339326463, 0.317987252]
        if dataset == 'eTour':
            f1_AC_XWCoDe, precision_AC_XWCoDe, recall_AC_XWCoDe = best_parameter_test_kfold(best_individual_eTour, dataset, "selected_features", ci, mi, wi,dependency_matrix)
        elif dataset == 'eANCI':
            f1_AC_XWCoDe, precision_AC_XWCoDe, recall_AC_XWCoDe = best_parameter_test_kfold(best_individual_eANCI, dataset, "selected_features", ci, mi, wi,dependency_matrix)
        elif dataset == 'iTrust':
            f1_AC_XWCoDe, precision_AC_XWCoDe, recall_AC_XWCoDe = best_parameter_test_kfold(best_individual_iTrust, dataset, "selected_features", ci, mi, wi, dependency_matrix)
        elif dataset == 'smos':
            f1_AC_XWCoDe, precision_AC_XWCoDe, recall_AC_XWCoDe = best_parameter_test_kfold(best_individual_smos, dataset, "selected_features", ci, mi, wi, dependency_matrix)
        # 将结果添加到列表
        results_list.append({'Dataset': dataset,
                             'XCoDe_precision': round(precision_AC_XWCoDe, 4),
                             'XCoDe_recall': round(recall_AC_XWCoDe, 4),
                             'XCoDe_f1': round(f1_AC_XWCoDe, 4)
                             })

    # 将列表转换为DataFrame
    results_df = pd.DataFrame(results_list)
    # 将DataFrame写入Excel文件
    results_df.to_excel('./AC_XWCoDe_albation_results_final10.xlsx', index=False)
