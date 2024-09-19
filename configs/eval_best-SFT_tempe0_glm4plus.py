from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, OpenAI, ZhiPuV2AI, TurboMindModel, HuggingFaceCausalLM
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


# ------------- Dataset Configuration
with read_base():
    from .datasets.Comac.comac_subjective_0711_wo_system  import subjective_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# ------------- Model Configuration
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='best-SFT_tempe0_glm4',
        path='/mnt/petrelfs/zhaosuiyi/projects/xtuner/factories/hf-models/best-SFT',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
]

# ------------- JudgeLLM Configuration
# judge_models = [dict(
#     abbr='deepseek-chat',
#     type=OpenAI,
#     path='deepseek-chat',
#     openai_api_base='https://api.deepseek.com/v1/chat/completions',
#     key='sk-8ebf5678b36e43b1a8953f96c8300e69',
#     meta_template=api_meta_template,
#     query_per_second=16,
#     max_out_len=2048,
#     max_seq_len=2048,
#     batch_size=8,
#     temperature=0,
# )]

judge_models = [
     dict(
        abbr='glm4plus',
        type=ZhiPuV2AI,
        path='glm-4-plus',
        key='',
        generation_kwargs={
            'tools': [
                {
                    'type': 'web_search',
                    'web_search': {
                        'enable': False # turn off the search
                    }
                }
            ]
        },
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8)
]

# ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=1000, mode='singlescore', models=models, judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=8,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=ComacSummarizer, judge_type='general')

work_dir = './outputs/best-SFT_tempe0_glm4plus'