from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, OpenAI, TurboMindModel
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

# ------------- Dataset Configuration
with read_base():
    from .datasets.Comac.comac_subjective_0627_train import subjective_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ------------- Model Configuration
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='cp_3e_sft_1e',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0701_e3_e1_hf',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='cp_3e_sft_2e',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0701_e3_e2_hf',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='cp_3e_sft_3e',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0701_e3_e3_exp1_hf',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='cp_3e_sft_6e',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0627_e3_e6_exp1_hf',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
]

# ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='deepseek-chat',
    type=OpenAI,
    path='deepseek-chat',
    openai_api_base='https://api.deepseek.com/v1/chat/completions',
    key='',
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

work_dir = 'outputs/eval_internlm2_7b_chat_comac_subjective_0701_train'
