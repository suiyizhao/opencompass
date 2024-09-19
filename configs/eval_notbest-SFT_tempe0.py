from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, OpenAI, TurboMindModel, HuggingFaceCausalLM
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import ComacSummarizer
from opencompass.runners import LocalRunner

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
generation_kwargs = dict(
    temperature=0,
    top_k=50,
    top_p=1.0,
)

# _meta_template = dict(
#         begin=[
#             dict(role='SYSTEM', fallback_role='HUMAN', prompt=
#             '''
#                 你应仅限于航空飞机设计领域回答问题，确保所有回答严格聚焦于相关专业领域。此外，你要遵循以下规则：1. 你的回答应以高置信度为目标，回答问题尽可能一步步进行，但要避免任何可能与航空飞机设计无关的解释；2. 如果你不明确问题所属的航空飞机设计子领域，请你首先考虑该问题是否和飞机的液压系统、飞行控制、动力系统和飞机结构等关键领域相关；3. 请确保在解释专业术语缩略词时，如果有多种含义，优先考虑其在飞机的液压系统、飞行控制、动力系统和飞机结构等关键领域中的含义，并提供应用实例。\n 问题是：
#             '''
#             ),
#         ],
#         # begin=[
#         #     dict(role='SYSTEM', fallback_role='HUMAN', prompt=
#         #     '''
#         #         第一个问题是：一加一等于几？\n第二个问题是：
#         #     '''
#         #     ),
#         # ],
#         round=[
#             dict(role='HUMAN', api_role='HUMAN'),
#             dict(role='BOT', api_role='BOT', generate=True),
#         ],
#     )



# ------------- Dataset Configuration
with read_base():
    from .datasets.Comac.comac_subjective_0711_wo_system  import subjective_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ------------- Model Configuration
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='iter4060',
        path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/20240910-InternTrain/internlm2_7b_full_comac-sft-data_Interntrain_iter_4060',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
        # meta_template = _meta_template
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='iter8120',
        path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/20240910-InternTrain/internlm2_7b_full_comac-sft-data_Interntrain_iter_8120',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
        # meta_template = _meta_template
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='iter12162',
        path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/20240910-InternTrain/internlm2_7b_full_comac-sft-data_Interntrain_iter_12162',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
        # meta_template = _meta_template
    ),
]

# ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='deepseek-chat',
    type=OpenAI,
    path='deepseek-chat',
    openai_api_base='https://api.deepseek.com/v1/chat/completions',
    key='sk-8ebf5678b36e43b1a8953f96c8300e69',
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8,
    temperature=0,
)]

# ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=1000, mode='singlescore', models=models, judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=8,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=ComacSummarizer, judge_type='general')

work_dir = './outputs/notbest-SFT_tempe0'
