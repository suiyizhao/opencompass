## 简要介绍

目前对于问答题的评估主要采用主观评估，即利用大语言模型对模型的回答和参考回答进行评估打分，打分主要从事实正确性、满足用户需求、清晰度、完备性四个方向进行打分。在summary中，还额外增加了事实正确率的计算，计算方式为统计事实正确性高于一定阈值的问答对占所有验证集的比例。有事实正确率@0.6、事实正确率@0.8、事实正确率@1.0，分别代表在事实正确性阈值在6、8、10条件下的SFT问答率。一般采用事实正确率@0.8。

## 如何使用？

主要通过配置文件运行，执行：

```bash
python run.py ${CONFIG_PY}
```

其中CONFIG_PY是配置文件，例如[eval_internlm2_7b_chat_comac_subjective_0711](configs/eval_internlm2_7b_chat_comac_subjective_0711.py)
其中需要经常修改的几个字段如下：

1. 数据集配置: opencompass 会读取datasets字典列表，并将列表中的每个元素作为一个需要评测的dataset. with read_base()则代表将下列文件中的datasets引入到当前文件，具体使用方法可以参考[datasets](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/datasets.html#).在base datasets目录下为自行定义的商飞数据集测试数据，一般情况下不需要修改，如果仅需修改测试数据集修改datasets.Comac.comac_subjective_0711_wo_system下的路径即可，如果需要重新定义主观数据集则需要参考该文件自行定义新的主观测试集。

```python
with read_base():
    from .datasets.Comac.comac_subjective_0711_wo_system  import subjective_datasets

...

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), []) # 将现有变量中所有以datasets为后缀的数据集添加到datasets列表中。
```

2. 模型配置
   模型配置通过models目录配置：示例如下：该段配置则是对两个模型进行测试，模型采用标准的huggingface格式模型。实际测试的时候，不同的模型和数据集会被划分到不同的cpu和gpu上进行评测。<br>

```python
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2_7b_full_comac-sft-data-0711_e3_e1_hf',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0711_e3_e1_hf',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2_7b_full_comac-sft-data-0712_e3_e1_v2_hf',
        path='weights/lyx/internlm2_7b_full_comac-sft-data-0712_e3_e1_v2_hf',
        max_out_len=1024,
        batch_size=4,
        generation_kwargs=generation_kwargs,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    ),
]
```

3. 主观评测模型配置：主观评测模型采用标准的openai格式的api进行评测，需要将其中的openai_api_base、key、path修改为对应的openai服务器地址、密钥和模型名称即可（注意实验室集群在使用gpt进行评测的时候可能需要挂代理）

```
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
```

4. summarizer配置：该配置为自行定义的总结器，一般情况下不需要修改，如果要修改的话可以参考ComacSummarizer对基类进行继承，实现summarizer的相关函数导入即可使用。
5. work_dir配置： 评测结果保存路径。

## 常见脚本

[tools/scripts/opencompass](tools/scripts/opencompass)存放了一下评测的常见实用脚本，包含评测的配置，评测结果的合并等，仅供参考。

## 更多文档

该文档主要针对商飞数据进行适配，关于其他opencompass的通用配置可以参考[README_zh](README_zh-CN.md)。
