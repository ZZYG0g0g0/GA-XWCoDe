"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/1/25
Last Updated: 2024/1/25
Version: 1.0.0
"""
import pandas as pd
from machine_learning import xgboost_classification_test_threshold3


def write_results_to_excel(result, file_path):
    # 解包result元组
    test_size, results = result
    # 为当前test_size创建一个DataFrame
    test_size_df = pd.DataFrame(results, columns=["threshold1", "threshold2", "precision", "recall", "f1"])

    # 使用ExcelWriter，设置模式为追加模式，如果文件不存在则创建
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # 尝试加载现有的工作表，如果不存在则创建一个新的
        try:
            # 尝试读取现有的工作表
            df_existing = pd.read_excel(writer, sheet_name='Results_final')
        except Exception as e:
            # 如果读取失败，说明工作表不存在，创建一个空的DataFrame
            df_existing = pd.DataFrame()

        # 获取已存在数据的列数，用于确定新数据应该开始的列位置
        startcol = df_existing.shape[1] + 1  # 空出一列作为分隔

        # 在同一工作表中追加数据，使用startcol根据已存在数据确定起始列
        test_size_df.to_excel(writer, sheet_name='Results_final', startcol=startcol, index=False)

def write_results_to_excel_to_t3(result, file_path):
    # 解包result元组
    test_size, results = result
    # 为当前test_size创建一个DataFrame
    test_size_df = pd.DataFrame(results, columns=["threshold3", "precision", "recall", "f1"])

    # 使用ExcelWriter，设置模式为追加模式，如果文件不存在则创建
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # 尝试加载现有的工作表，如果不存在则创建一个新的
        try:
            # 尝试读取现有的工作表
            df_existing = pd.read_excel(writer, sheet_name='Results_p3_clf')
        except Exception as e:
            # 如果读取失败，说明工作表不存在，创建一个空的DataFrame
            df_existing = pd.DataFrame()

        # 获取已存在数据的列数，用于确定新数据应该开始的列位置
        startcol = df_existing.shape[1] + 1  # 空出一列作为分隔

        # 在同一工作表中追加数据，使用startcol根据已存在数据确定起始列
        test_size_df.to_excel(writer, sheet_name='Results_p3_clf', startcol=startcol, index=False)

if __name__ == '__main__':
    datasets = ['iTrust', 'smos', 'eTour', 'eANCI']
    test_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # classifiers = [xgboost_regression]
    # for dataset in datasets:
    #     ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
    #     for classifier in classifiers:
    #         res = classifier(test_size, dataset, "selected_features", ci, mi, wi, dependency_matrix)
    #         for r in res:
    #             if dataset == 'eANCI':
    #                 write_results_to_excel(r, './result_eANCI_final.xlsx')
    #             if dataset == 'eTour':
    #                 write_results_to_excel(r, './result_eTour_final.xlsx')
    #             if dataset == 'iTrust':
    #                 write_results_to_excel(r, './result_iTrust_final.xlsx')
    #             if dataset == 'smos':
    #                 write_results_to_excel(r, './result_smos_final.xlsx')
    for dataset in datasets:
        print(dataset)
        res = xgboost_classification_test_threshold3(test_size, dataset, "selected_features")
        # ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        # res = xgboost_regression(test_size, dataset, "selected_features", ci, mi, wi, dependency_matrix)
        # print(f'{dataset}->ci:{ci},mi:{mi},wi{wi}')
        for r in res:
            if dataset == 'eANCI':
                write_results_to_excel_to_t3(r, f'../res/result_eANCI.xlsx')
            if dataset == 'eTour':
                write_results_to_excel_to_t3(r, '../res/result_eTour.xlsx')
            if dataset == 'iTrust':
                write_results_to_excel_to_t3(r, '../res/result_iTrust.xlsx')
            if dataset == 'smos':
                write_results_to_excel_to_t3(r, '../res/result_smos.xlsx')