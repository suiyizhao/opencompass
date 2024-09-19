from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
# from opencompass.datasets import AlignmentBenchDataset
from opencompass.datasets import ComacSubjectiveDataset

subjective_reader_cfg = dict(
    input_columns=['input', 'system', 'critiquellm_prefix'],
    output_column='output',
)

subjective_all_sets = [
    'AircraftDesignManual-12_val',
]
data_path = 'data/Comac/xtuner/sft/generated/AircraftDesignManual/12'

subjective_config_path = 'data/Comac/config'
subjective_config_name = 'multi-dimension'

subjective_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', fallback_role='HUMAN',
                         prompt='{system}'),
                ],
                round=[
                    dict(
                        role='HUMAN',
                        prompt='{input}'
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
