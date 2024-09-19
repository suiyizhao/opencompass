from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import ComacSubjectiveDataset

# 三套试卷
subjective_reader_cfg = dict(
    input_columns=['input', 'critiquellm_prefix'],
    output_column='output',
)

subjective_all_sets = [
    '机体翻译','机体简答','机体推理',
    '液压填空','液压翻译', '液压简答','液压推理',
    '通用翻译','通用填空','通用推理',
]

data_path = 'sft_data_0812_data/Comac/sft/humananotated/0708/All'

subjective_config_path = 'sft_data_0812_data/Comac/config'
subjective_config_name = 'multi-dimension'

subjective_datasets = []

# sys = '你应仅限于航空飞机设计领域回答问题，确保所有回答严格聚焦于相关专业领域。此外，你要遵循以下规则：1. 你的回答应以高置信度为目标，回答问题尽可能一步步进行，但要避免任何可能与航空飞机设计无关的解释；2. 如果你不明确问题所属的航空飞机设计子领域，请你首先考虑该问题是否和飞机的液压系统、飞行控制、动力系统和飞机结构等关键领域相关；3. 请确保在解释专业术语缩略词时，如果有多种含义，优先考虑其在飞机的液压系统、飞行控制、动力系统和飞机结构等关键领域中的含义，并提供应用实例。\n 问题是：'
sys = '你是一个专业的航空航天和飞机设计专家，你的任务是提供简明、准确、专业的回答。在回答问题时，请确保你的回答符合以下标准： 1、答案准确性：你需要确保彻底理解用户的提问，抓住问题中的关键点和意图，仔细分析问题所需的核心信息，并尽可能做出最精确的答复；2、简明性：你的回答要尽量简洁，只提供用户所需的信息，不要加入不必要的背景信息，除非问题本身要求这么做。\n 问题是：'
for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt='{input}'
                        # prompt=sys+'{input}'
                    ),
                ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=2048),
    )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=ComacSubjectiveDataset,
            path=data_path,
            name=_name,
            subjective_config_path=subjective_config_path,
            subjective_config_name=subjective_config_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
