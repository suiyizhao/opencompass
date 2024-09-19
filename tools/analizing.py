import pandas as pd
import glob
import os
from tabulate import tabulate
all_files=[
"/mnt/petrelfs/zhaosuiyi/projects/opencompass/outputs/CT20b-0912-ckpt-hf21907__internlm2-5_train_tempe0/20240919_143131/summary/20240919_143131/judged-by--deepseek-chat-dimension.csv",
]

all_transformed_data = []

for filename in all_files:
    df = pd.read_csv(filename)
    pivot_df = df.pivot(index='模型', columns='数据集', values='事实正确率@0.8')
    pivot_df['平均分'] = pivot_df.mean(axis=1)
#    pivot_df['来源文件'] = filename  # 添加文件名作为来源标识
    all_transformed_data.append(pivot_df)

# 将所有数据合并到一个 DataFrame
combined_df = pd.concat(all_transformed_data)

# 对所有数据计算平均分
final_averages = combined_df.groupby(combined_df.index).mean()
final_averages['最终平均'] = final_averages['平均分']

# 合并最终平均分到每个模型数据中
result_df = pd.merge(combined_df.reset_index(), final_averages['最终平均'], on='模型', how='left')

# 保存最终结果到一个新的 CSV 文件
result_df.sort_values(by='模型', inplace=True)
result_df.to_csv(os.path.dirname(all_files[0]) + '/final_transformed_output_with_details.csv', index=False)
print(tabulate(result_df, headers='keys', tablefmt='grid'))

print(f"所有文件处理完成，最终结果保存在 '{os.path.dirname(all_files[0])}/final_transformed_output_with_details.csv'")
