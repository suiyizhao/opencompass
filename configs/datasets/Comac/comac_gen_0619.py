from opencompass.datasets.comac import ComacDataset
from opencompass.openicl.icl_evaluator import BleuEvaluator

# 商飞 6月 12 日提供的SFT数据，直接进行问答评估。
SFData_datasets = [
    dict(
        type=ComacDataset,
        name='val',
        path='data/sf_data/0612',
        abbr='SFData-val0612',
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=512,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template='{input}\n\n 回答:\n',
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        reader_cfg=dict(
            input_columns='input',
            output_column='output'),
        eval_cfg = dict(
            evaluator=dict(type=BleuEvaluator),
            pred_role='BOT',
        )
    ),
]
