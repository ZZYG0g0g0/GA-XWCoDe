import pandas as pd
from scipy.stats import mannwhitneyu

if __name__ == '__main__':
    # 数据集名称
    datasets = ['eTour', 'eANCI', 'iTrust', 'smos']
    metrics = ['Precision', 'Recall', 'F1']

    # 初始化结果存储
    results_summary = []

    # 对每个数据集进行检验
    for dataset in datasets:
        # 读取AC_XWCoDe和TRAIL的结果
        ac_xwcode_df = pd.read_excel(f'AC_XWCoDe_results_{dataset}.xlsx')
        trail_df = pd.read_excel(f'DF4RT_results_{dataset}.xlsx')

        # 处理每次十折交叉验证为一个实验
        ac_grouped = ac_xwcode_df.groupby('ID').mean()
        trail_grouped = trail_df.groupby('ID').mean()

        # 为precision, recall, f1分别执行Mann-Whitney U检验
        for metric in metrics:
            ac_data = ac_grouped[metric]
            trail_data = trail_grouped[metric]

            # 进行Mann-Whitney U检验
            stat, p_value = mannwhitneyu(ac_data, trail_data, alternative='greater')

            # 保存结果到列表，用于后续分析或展示
            results_summary.append({
                'Dataset': dataset,
                'Metric': metric,
                'U Statistic': stat,
                'P-value': p_value,
                'AC_XWCoDe better': p_value < 0.01  # 如果p-value < 0.05, 则AC_XWCoDe显著好于TRAIL
            })

    # 将汇总结果转换为DataFrame
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)
