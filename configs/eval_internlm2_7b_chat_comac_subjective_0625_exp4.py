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
    from .datasets.Comac.comac_subjective_0625 import subjective_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ------------- Model Configuration
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2_7b_full_comac-sft-data-0625_e3_exp4_hf',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0625_e3_exp4_hf',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    )
]

# ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='deepseek-chat',
    type=OpenAI,
    path='deepseek-chat',
    openai_api_base='https://oneapi.liuyanxing.site:8443/v1/chat/completions',
    key='sk-XeAXyQ9Y0Exll4jD1b0e7351756d46408b6aD275081fB61f',
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
    runner=dict(type=LocalRunner, max_num_workers=2,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=ComacSummarizer, judge_type='general')

work_dir = 'outputs/eval_internlm2_7b_chat_comac_subjective_0625_exp4'
