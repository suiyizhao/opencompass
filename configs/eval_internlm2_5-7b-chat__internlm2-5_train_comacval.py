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
generation_kwargs = dict(
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)

# ------------- Dataset Configuration
with read_base():
    from .datasets.Comac.comac_subjective_0711_wo_system  import subjective_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ------------- Model Configuration
models = [
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='iter_4060',
    #     path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/internlm2_5-7b-chat__internlm2-5_train_comacval/iter_4060',
    #     max_out_len=1024,
    #     batch_size=4,
    #     generation_kwargs=generation_kwargs,
    #     run_cfg=dict(num_gpus=1),
    #     stop_words=['</s>', '<|im_end|>'],
    # ),
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='iter_8120',
    #     path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/internlm2_5-7b-chat__internlm2-5_train_comacval/iter_8120',
    #     max_out_len=1024,
    #     batch_size=4,
    #     generation_kwargs=generation_kwargs,
    #     run_cfg=dict(num_gpus=1),
    #     stop_words=['</s>', '<|im_end|>'],
    # ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='iter_12180',
        path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/internlm2_5-7b-chat__internlm2-5_train_comacval/iter_12180',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
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

work_dir = './outputs/internlm2_5-7b-chat__internlm2-5_train_comacval'
