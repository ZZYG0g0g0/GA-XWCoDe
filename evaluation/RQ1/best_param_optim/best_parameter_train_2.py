"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/4/5
Last Updated: 2024/4/5
Version: 1.0.0
"""
from deap import base, creator, tools, algorithms
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ML.machine_learning import dependency_based_resort, calculate_dependency_importance
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def cxUniformBounded(ind1, ind2, low, up, indpb):
    """执行均匀交叉操作，并确保子代个体的参数在指定边界内。

    :param ind1: 第一个待交叉的个体。
    :param ind2: 第二个待交叉的个体。
    :param low: 参数的下界列表。
    :param up: 参数的上界列表。
    :param indpb: 每个属性被交换的概率。
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if np.random.random() < indpb:
            # 执行均匀交叉
            ind1[i], ind2[i] = ind1[i] * (1 - indpb) + ind2[i] * indpb, ind2[i] * (1 - indpb) + ind1[i] * indpb
            # 确保参数值不超过边界
            ind1[i] = np.clip(ind1[i], low[i], up[i])
            ind2[i] = np.clip(ind2[i], low[i], up[i])
    return ind1, ind2

def xgboost_with_optim_parameter(dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')

    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    # 假设calculate_dependency_importance等其他必要的函数已经在上下文中定义
    attention_weight = [ci, mi, wi]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_eta", np.random.uniform, 0, 1.0)
    toolbox.register("attr_max_depth", np.random.randint, 3, 10)
    toolbox.register("attr_subsample", np.random.uniform, 0, 1.0)
    toolbox.register("threshold_1", np.random.uniform, 0.0, 0.5)
    toolbox.register("threshold_2", np.random.uniform, 0.0, 0.5)
    toolbox.register("threshold_3", np.random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_eta, toolbox.attr_max_depth, toolbox.attr_subsample,
                                                                         toolbox.threshold_1, toolbox.threshold_2, toolbox.threshold_3), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    n_runs = 1
    for run in range(n_runs):
        indices = np.arange(len(X_normalized))
        X_train_val, X_test, y_train_val, y_test, train_val_index, test_index = train_test_split(X_normalized, labels,
                                                                                                 indices, test_size=0.1)
        print(f"run times: {run}")
        low = [0, 3, 0, 0, 0, 0]
        up = [1, 10, 1.0, 0.5, 0.5, 1.0]

        # 适应度函数，使用F1分数
        def evalXGBoost(individual):
            # 生成数据的索引
            indices = np.arange(len(X_normalized))
            X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_normalized, labels, indices, test_size=0.1)  # 9:1比例分割
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=individual[0],
                                    max_depth=int(individual[1]), subsample=individual[2])
            clf.fit(X_train, y_train.values.ravel())
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = labels.iloc[train_index].values.ravel().tolist()
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
            threshold_count = int(individual[4] * len(df))
            # 获取代码依赖关系较前的下标
            top_idx = df['Test_Index'][:threshold_count].tolist()
            # 获取代码依赖关系较后的下标
            if threshold_count > 0:
                last_idx = df['Test_Index'][-threshold_count:].tolist()
            else:
                last_idx = []
            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= individual[5] - individual[3] and y_pred_fb[i] <= individual[5] and test_index[i] in top_idx):
                    y_pred_fb[i] = 1  # 改为正
                elif (y_pred_fb[i] >= individual[5] and y_pred_fb[i] <= individual[5] + individual[3] and test_index[i] in last_idx):
                    y_pred_fb[i] = 0  # 改为负
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] < individual[5]):
                    y_pred_fb[i] = 0
                else:
                    y_pred_fb[i] = 1
            f1score = f1_score(y_test, y_pred_fb, average='macro')  # 计算宏平均F1分数
            return f1score  # 返回f1值

        toolbox.register("evaluate", evalXGBoost)
        # toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mate", cxUniformBounded, low=low, up=up, indpb=0.6)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=1.0, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 运行遗传算法
        population = toolbox.population(n=50)
        NGEN = 200
        # 初始化一个空的DataFrame来存储所有结果
        all_results_df = pd.DataFrame()
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = (fit,)
            population = toolbox.select(offspring, k=len(population))

            # 每迭代5次，打印当前所有个体的参数值
            if (gen + 1) % 5 == 0:  # gen从0开始计数，因此这里用gen + 1
                data = []
                print(f"Generation {gen + 1}:")
                for idx, ind in enumerate(population, 1):
                    print(f"Individual {idx}: {ind}")
                    # 这里调用cross_validation_score函数来执行交叉验证并获取F1分数
                    cv_f1_score = best_parameter_test(ind, dataset_name, feature_file_name, ci, mi, wi,
                                                             dependency_matrix, train_val_index, test_index)
                    print(f"Individual:{idx} with f1 in test:{cv_f1_score}")
                    data.append([gen + 1] + ind[:] + [cv_f1_score])

                # 将当前代的数据转换为DataFrame
                gen_df = pd.DataFrame(data, columns=['Generation', 'eta', 'max_depth', 'subsample', 'threshold_1',
                                                         'threshold_2', 'threshold_3', 'F1_Score'])
                # 将当前代的结果追加到总的DataFrame中
                all_results_df = pd.concat([all_results_df, gen_df], ignore_index=True)
            # 使用ExcelWriter来写入数据
            with pd.ExcelWriter(f'./(9)遗传概率0.9-变异概率0.1-{dataset_name}.xlsx', engine='xlsxwriter') as writer:
                all_results_df.to_excel(writer, sheet_name=dataset_name)


def best_parameter_test(individual, dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel(f'../../datasets/{dataset_name}/{feature_file_name}.xlsx')
    labels = pd.read_excel(f'../../datasets/{dataset_name}/labels.xlsx')

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    labels = labels.values.ravel()

    attention_weight = [ci, mi, wi]

    # Use the best individual's parameters for XGBoost
    eta, max_depth, subsample, threshold_1, threshold_2, threshold_3 = individual

    # skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf = KFold(n_splits=10, shuffle=True)
    all_f1_scores = []

    # Perform ten iterations of ten-fold cross-validation
    for iteration in range(1):
        iteration_scores = []
        for train_index, test_index in skf.split(X_normalized, labels):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=eta, max_depth=int(max_depth),
                                            subsample=subsample)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = labels[train_index].tolist()

            # Apply dependency-based resorting and threshold adjustments here
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                                 dependency_matrix,   train_label)
            threshold_count = int(threshold_2 * len(df))
            top_idx = df['Test_Index'][:threshold_count].tolist()
            last_idx = df['Test_Index'][-threshold_count:].tolist() if threshold_count > 0 else []

            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= threshold_3 - threshold_1 and y_pred_fb[i] <= threshold_3 and test_index[i] in top_idx):
                    y_pred_fb[i] = 1  # Adjust to positive
                elif (y_pred_fb[i] >= threshold_3 and y_pred_fb[i] <= threshold_3 + threshold_1 and test_index[i] in last_idx):
                    y_pred_fb[i] = 0  # Adjust to negative

            y_pred_final = [1 if pred >= threshold_3 else 0 for pred in y_pred_fb]
            f1 = f1_score(y_test, y_pred_final, average='macro')
            iteration_scores.append(f1)
        all_f1_scores.append(np.mean(iteration_scores))
    overall_average_f1_score = np.mean(all_f1_scores)
    return overall_average_f1_score

def xgboost_with_optim_parameter_best(dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')

    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    # 假设calculate_dependency_importance等其他必要的函数已经在上下文中定义
    attention_weight = [ci, mi, wi]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_eta", np.random.uniform, 0, 1.0)
    toolbox.register("attr_max_depth", np.random.randint, 3, 10)
    toolbox.register("attr_subsample", np.random.uniform, 0, 1.0)
    toolbox.register("threshold_1", np.random.uniform, 0.0, 0.5)
    toolbox.register("threshold_2", np.random.uniform, 0.0, 0.5)
    toolbox.register("threshold_3", np.random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_eta, toolbox.attr_max_depth, toolbox.attr_subsample,
                                                                         toolbox.threshold_1, toolbox.threshold_2, toolbox.threshold_3), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    n_runs = 1
    for run in range(n_runs):
        indices = np.arange(len(X_normalized))
        X_train_val, X_test, y_train_val, y_test, train_val_index, test_index = train_test_split(X_normalized, labels,
                                                                                                 indices, test_size=0.1)
        low = [0, 3, 0, 0, 0, 0]
        up = [1, 10, 1.0, 0.5, 0.5, 1.0]

        # 适应度函数，使用F1分数
        def evalXGBoost(individual):
            # 生成数据的索引
            indices = np.arange(len(X_normalized))
            X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_normalized, labels, indices, test_size=0.1)  # 9:1比例分割
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=individual[0],
                                    max_depth=int(individual[1]), subsample=individual[2])
            clf.fit(X_train, y_train.values.ravel())
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = labels.iloc[train_index].values.ravel().tolist()
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
            threshold_count = int(individual[4] * len(df))
            # 获取代码依赖关系较前的下标
            top_idx = df['Test_Index'][:threshold_count].tolist()
            # 获取代码依赖关系较后的下标
            if threshold_count > 0:
                last_idx = df['Test_Index'][-threshold_count:].tolist()
            else:
                last_idx = []
            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= individual[5] - individual[3] and y_pred_fb[i] <= individual[5] and test_index[i] in top_idx):
                    y_pred_fb[i] = 1  # 改为正
                elif (y_pred_fb[i] >= individual[5] and y_pred_fb[i] <= individual[5] + individual[3] and test_index[i] in last_idx):
                    y_pred_fb[i] = 0  # 改为负
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] < individual[5]):
                    y_pred_fb[i] = 0
                else:
                    y_pred_fb[i] = 1
            f1score = f1_score(y_test, y_pred_fb, average='macro')  # 计算宏平均F1分数
            return f1score  # 返回f1值

        toolbox.register("evaluate", evalXGBoost)
        toolbox.register("mate", cxUniformBounded, low=low, up=up, indpb=0.9)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=1.0, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 运行遗传算法
        population = toolbox.population(n=50)
        NGEN = 200

        for gen in range(NGEN):
            print(gen)
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = (fit,)
            population = toolbox.select(offspring, k=len(population))

        best_individual = tools.selBest(population, k=1)[0]
        best_f1 = best_parameter_test_kfold(best_individual, dataset_name, feature_file_name, ci,mi,wi, dependency_matrix)
        return best_f1

def best_parameter_test_kfold(individual, dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel(f'../../datasets/{dataset_name}/{feature_file_name}.xlsx')
    labels = pd.read_excel(f'../../datasets/{dataset_name}/labels.xlsx')

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    labels = labels.values.ravel()

    attention_weight = [ci, mi, wi]

    # Use the best individual's parameters for XGBoost
    eta, max_depth, subsample, threshold_1, threshold_2, threshold_3 = individual

    # skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf = KFold(n_splits=10, shuffle=True)
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []

    # Perform ten iterations of ten-fold cross-validation
    for iteration in range(50):
        iteration_f1_scores = []
        iteration_precision_scores = []
        iteration_recall_scores = []
        for train_index, test_index in skf.split(X_normalized, labels):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=eta, max_depth=int(max_depth),
                                            subsample=subsample)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = labels[train_index].tolist()

            # Apply dependency-based resorting and threshold adjustments here
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                                 dependency_matrix,   train_label)
            threshold_count = int(threshold_2 * len(df))
            top_idx = df['Test_Index'][:threshold_count].tolist()
            last_idx = df['Test_Index'][-threshold_count:].tolist() if threshold_count > 0 else []

            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= threshold_3 - threshold_1 and y_pred_fb[i] <= threshold_3 and test_index[i] in top_idx):
                    y_pred_fb[i] = 1  # Adjust to positive
                elif (y_pred_fb[i] >= threshold_3 and y_pred_fb[i] <= threshold_3 + threshold_1 and test_index[i] in last_idx):
                    y_pred_fb[i] = 0  # Adjust to negative

            y_pred_final = [1 if pred >= threshold_3 else 0 for pred in y_pred_fb]
            f1 = f1_score(y_test, y_pred_final, average='macro')
            precision = precision_score(y_test, y_pred_final, average='macro')
            recall = recall_score(y_test, y_pred_final, average='macro')
            iteration_f1_scores.append(f1)
            iteration_precision_scores.append(precision)
            iteration_recall_scores.append(recall)
        all_f1_scores.append(np.mean(iteration_f1_scores))
        all_precision_scores.append(np.mean(iteration_precision_scores))
        all_recall_scores.append(np.mean(iteration_recall_scores))
    overall_average_f1_score = np.mean(all_f1_scores)
    overall_average_precision_score = np.mean(all_precision_scores)
    overall_average_recall_score = np.mean(all_recall_scores)
    return overall_average_f1_score, overall_average_precision_score, overall_average_recall_score


def best_parameter_test_kfold_all_results(individual, dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel(f'../../datasets/{dataset_name}/{feature_file_name}.xlsx')
    labels = pd.read_excel(f'../../datasets/{dataset_name}/labels.xlsx')

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    labels = labels.values.ravel()

    attention_weight = [ci, mi, wi]

    # Use the best individual's parameters for XGBoost
    eta, max_depth, subsample, threshold_1, threshold_2, threshold_3 = individual

    skf = KFold(n_splits=10, shuffle=True)
    results = []

    # Perform fifty iterations of ten-fold cross-validation
    for iteration in range(50):
        for fold, (train_index, test_index) in enumerate(skf.split(X_normalized, labels), 1):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=eta, max_depth=int(max_depth),
                                    subsample=subsample)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = labels[train_index].tolist()

            # Apply dependency-based resorting and threshold adjustments here
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
            threshold_count = int(threshold_2 * len(df))
            top_idx = df['Test_Index'][:threshold_count].tolist()
            last_idx = df['Test_Index'][-threshold_count:].tolist() if threshold_count > 0 else []

            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= threshold_3 - threshold_1 and y_pred_fb[i] <= threshold_3 and test_index[
                    i] in top_idx):
                    y_pred_fb[i] = 1  # Adjust to positive
                elif (y_pred_fb[i] >= threshold_3 and y_pred_fb[i] <= threshold_3 + threshold_1 and test_index[
                    i] in last_idx):
                    y_pred_fb[i] = 0  # Adjust to negative

            y_pred_final = [1 if pred >= threshold_3 else 0 for pred in y_pred_fb]
            f1 = f1_score(y_test, y_pred_final, average='macro')
            precision = precision_score(y_test, y_pred_final, average='macro')
            recall = recall_score(y_test, y_pred_final, average='macro')

            # Save results
            results.append({
                'ID': f'{iteration}_{fold}',
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def AC_XWCoDe_all_test_size(individual, dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel(f'../../datasets/{dataset_name}/{feature_file_name}.xlsx')
    labels = pd.read_excel(f'../../datasets/{dataset_name}/labels.xlsx')

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    labels = labels.values.ravel()

    attention_weight = [ci, mi, wi]

    # Use the best individual's parameters for XGBoost
    eta, max_depth, subsample, threshold_1, threshold_2, threshold_3 = individual

    results = []

    # Perform fifty iterations for each test size
    for i in range(1, 10):  # Loop over test sizes from 10% to 90%
        current_test_size = i / 10
        for iteration in range(50):
            X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_normalized, labels, range(len(X_normalized)), test_size=current_test_size)

            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', eta=eta, max_depth=int(max_depth),
                                    subsample=subsample)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            train_label = y_train.tolist()

            # Apply dependency-based resorting and threshold adjustments here
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
            threshold_count = int(threshold_2 * len(df))
            top_idx = df['Test_Index'][:threshold_count].tolist()
            last_idx = df['Test_Index'][-threshold_count:].tolist() if threshold_count > 0 else []

            y_pred_fb = y_pred.copy()
            for i in range(len(y_pred_fb)):
                if (y_pred_fb[i] >= threshold_3 - threshold_1 and y_pred_fb[i] <= threshold_3 and i in top_idx):
                    y_pred_fb[i] = 1  # Adjust to positive
                elif (y_pred_fb[i] >= threshold_3 and y_pred_fb[i] <= threshold_3 + threshold_1 and i in last_idx):
                    y_pred_fb[i] = 0  # Adjust to negative

            y_pred_final = [1 if pred >= threshold_3 else 0 for pred in y_pred_fb]
            f1 = f1_score(y_test, y_pred_final, average='macro')

            precision = precision_score(y_test, y_pred_final, average='macro')
            recall = recall_score(y_test, y_pred_final, average='macro')

            # Save results
            results.append({
                'Test Size': current_test_size,
                'Iteration': iteration,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    return results_df




if __name__ == '__main__':
    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    # for dataset in datasets:
    #     ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        # individual = xgboost_with_optim_parameter(dataset, "selected_features", ci, mi, wi, dependency_matrix)
        # best_parameter_test(individual, dataset, "selected_features", ci, mi, wi, dependency_matrix)
    for d in datasets:
        ci, mi, wi, dependency_matrix = calculate_dependency_importance(d, datasets)
        # xgboost_with_optim_parameter(d, "selected_features", ci, mi, wi, dependency_matrix)
        xgboost_with_optim_parameter_best(d, "selected_features", ci, mi, wi, dependency_matrix)