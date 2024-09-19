# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .alignmentbench import (CATEGORIES, post_process_alignbench,
                             post_process_alignbench_plus)
from .subjective_post_process import post_process_autoj, post_process_judgelm
from .utils import get_judgeanswer_and_reference, get_outdir


def get_dimension_results(judged_answers, references, fout, fout_flag, model,
                          dataset):
    dimension_ratings = defaultdict(int)
    dimension_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        for k, v in ans['rating'].items():
            if k != '综合得分' or k != 'Overall Score':
                dimension_ratings[k] += v
                dimension_counts[k] += 1
            else:
                if k == '综合得分':
                    dimension_ratings['综合得分'] += ans['score']
                    dimension_counts['综合得分'] += 1
                else:
                    dimension_ratings['Overall Score'] += ans['score']
                    dimension_counts['Overall Score'] += 1

    dimension_avg_ratings = defaultdict(float)
    for dimension, total_score in dimension_ratings.items():
        s = total_score / dimension_counts[dimension]
        s = round(s, 2)
        dimension_avg_ratings[dimension] = s

    # 计算事实正确率
    all = len(judged_answers)
    thrs = [6, 8, 10]
    for threshold in thrs:
        tp = 0
        for ans, ref in zip(judged_answers, references):
            pred_score = ans['rating']['事实正确性']
            if pred_score >= threshold:
                tp += 1
        dimension_avg_ratings['事实正确率' + f'@{threshold/10}'] = '{:.3f}'.format(
            tp / all)

    scores = {model: dimension_avg_ratings}
    rows = list(scores.keys())
    columns = list(scores[rows[0]].keys())
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['模型'] + ['数据集'] + columns)

        for row in rows:
            writer.writerow([row] + [dataset.abbr] +
                            [scores[row][column] for column in columns])


class ComacSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_models = self.cfg.get('judge_models', None)
        self.judge_type = judge_type
        assert self.judge_type in [
            'general', 'autoj', 'judgelm', 'general_plus'
        ]
        self.judge_map = {
            'general': post_process_alignbench,
            'general_plus': post_process_alignbench_plus,
            'autoj': post_process_autoj,
            'judgelm': post_process_judgelm
        }
        self.judge_function = self.judge_map[self.judge_type]
        self.category = CATEGORIES

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        for judge_model in self.judge_models:
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            fout_flag, fout_flag2 = 0, 0
            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                if os.path.isdir(subdir_path):
                    model = eval_model_abbr
                    if self.judge_type == 'general':
                        fout = osp.join(
                            output_dir,
                            'judged-by--' + judge_abbr + '-dimension.csv')
                    fout2 = osp.join(
                        output_dir,
                        'judged-by--' + judge_abbr + '-capability.csv')
                    for dataset in dataset_cfgs:
                        judged_answers, references = get_judgeanswer_and_reference(
                            dataset, subdir_path, self.judge_function)
                        if self.judge_type == 'general':
                            get_dimension_results(judged_answers, references,
                                                  fout, fout_flag, model,
                                                  dataset)
                            fout_flag += 1
                else:
                    print(subdir_path + ' is not exist! please check!')
        if self.judge_type == 'general':
            with open(fout, 'r') as f:
                x = from_csv(f, delimiter=',')
            print(x)
            print(fout)
        # with open(fout2, 'r') as f:
        #     x = from_csv(f, delimiter=',')
        print(x)
        # print(fout2)
