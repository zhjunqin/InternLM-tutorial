# XTuner 微调实践

参考了： https://github.com/InternLM/tutorial/blob/main/xtuner/self.md

## 一点记录

一开始我没有修改 max_length 参数，发现怎么训练都出不来效果，后来改了，发现就可以了。

```
# 训练中最大的文本长度
max_length = 512
```

## 训练日志

```
# xtuner train /data/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py
[2024-01-13 23:06:47,647] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-01-13 23:06:51,504] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
01/13 23:06:53 - mmengine - INFO - Config:
SYSTEM = ''
accumulative_counts = 16
batch_size = 2
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.DatasetInfoHook'),
    dict(
        evaluation_inputs=[
            '请介绍一下你自己',
            '请做一下自我介绍',
        ],
        every_n_iters=90,
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm_chat',
        system='',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.EvaluateChatHook'),
]
data_path = '/data/personal_assistant/data/personal_assistant.json'
dataloader_num_workers = 0
default_hooks = dict(
    checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
    logger=dict(interval=10, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 90
evaluation_inputs = [
    '请介绍一下你自己',
    '请做一下自我介绍',
]
launcher = 'none'
load_from = None
log_level = 'INFO'
lr = 0.0002
max_epochs = 3
max_length = 512
max_norm = 1
model = dict(
    llm=dict(
        pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
        quantization_config=dict(
            bnb_4bit_compute_dtype='torch.float16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            load_in_4bit=True,
            load_in_8bit=False,
            type='transformers.BitsAndBytesConfig'),
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    lora=dict(
        bias='none',
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    type='xtuner.model.SupervisedFinetune')
optim_type = 'bitsandbytes.optim.PagedAdamW32bit'
optim_wrapper = dict(
    accumulative_counts=16,
    clip_grad=dict(error_if_nonfinite=False, max_norm=1),
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0002,
        type='bitsandbytes.optim.PagedAdamW32bit',
        weight_decay=0),
    type='mmengine.optim.AmpOptimWrapper')
pack_to_max_length = True
param_scheduler = dict(
    T_max=3,
    by_epoch=True,
    convert_to_iter_based=True,
    eta_min=0.0,
    type='mmengine.optim.CosineAnnealingLR')
pretrained_model_name_or_path = '/data/models/internlm-chat-7b/'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.internlm_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        dataset=dict(
            data_files=dict(
                train='/data/personal_assistant/data/personal_assistant.json'),
            path='json',
            type='datasets.load_dataset'),
        dataset_map_fn=None,
        max_length=512,
        pack_to_max_length=True,
        remove_unused_columns=True,
        shuffle_before_pack=True,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.internlm_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.process_hf_dataset'),
    num_workers=0,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
train_dataset = dict(
    dataset=dict(
        data_files=dict(
            train='/data/personal_assistant/data/personal_assistant.json'),
        path='json',
        type='datasets.load_dataset'),
    dataset_map_fn=None,
    max_length=512,
    pack_to_max_length=True,
    remove_unused_columns=True,
    shuffle_before_pack=True,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.internlm_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path='/data/models/internlm-chat-7b/',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.process_hf_dataset')
visualizer = None
weight_decay = 0
work_dir = './work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy'

quantization_config convert to <class 'transformers.utils.quantization_config.BitsAndBytesConfig'>
01/13 23:06:53 - mmengine - WARNING - Failed to search registry with scope "mmengine" in the "builder" registry tree. As a workaround, the current "builder" registry in "xtuner" is used to build instance. This may cause unexpected failure when running the built modules. Please check whether "mmengine" is a correct scope, or whether the registry is initialized.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:14<00:00,  9.37s/it]
01/13 23:08:09 - mmengine - INFO - dispatch internlm attn forward
01/13 23:08:09 - mmengine - WARNING - Due to the implementation of the PyTorch version of flash attention, even when the `output_attentions` flag is set to True, it is not possible to return the `attn_weights`.
01/13 23:08:11 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
01/13 23:08:12 - mmengine - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) RuntimeInfoHook
(BELOW_NORMAL) LoggerHook
 --------------------
before_train:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(NORMAL      ) DatasetInfoHook
(NORMAL      ) EvaluateChatHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(NORMAL      ) DistSamplerSeedHook
 --------------------
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
 --------------------
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(NORMAL      ) EvaluateChatHook
(BELOW_NORMAL) LoggerHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
after_train_epoch:
(NORMAL      ) IterTimerHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_val:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) DatasetInfoHook
 --------------------
before_val_epoch:
(NORMAL      ) IterTimerHook
 --------------------
before_val_iter:
(NORMAL      ) IterTimerHook
 --------------------
after_val_iter:
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
after_val:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) EvaluateChatHook
 --------------------
after_train:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) EvaluateChatHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_test:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) DatasetInfoHook
 --------------------
before_test_epoch:
(NORMAL      ) IterTimerHook
 --------------------
before_test_iter:
(NORMAL      ) IterTimerHook
 --------------------
after_test_iter:
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_test:
(VERY_HIGH   ) RuntimeInfoHook
 --------------------
after_run:
(BELOW_NORMAL) LoggerHook
 --------------------
Downloading data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2416.07it/s]
Extracting data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 891.65it/s]
Generating train split: 10001 examples [00:00, 141973.59 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 10001/10001 [00:00<00:00, 19236.12 examples/s]
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10001/10001 [00:00<00:00, 103433.97 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10001/10001 [00:04<00:00, 2354.37 examples/s]
Flattening the indices: 100%|██████████████████████████████████████████████████████████████████████████████████| 10001/10001 [00:00<00:00, 35047.12 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 10001/10001 [00:00<00:00, 20028.98 examples/s]
01/13 23:08:29 - mmengine - WARNING - Dataset Dataset has no metainfo. ``dataset_meta`` in visualizer will be None.
01/13 23:08:29 - mmengine - INFO - Num train samples 762
01/13 23:08:29 - mmengine - INFO - train example:
01/13 23:08:29 - mmengine - INFO -  <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s><s><|User|
01/13 23:08:29 - mmengine - INFO - before_train in EvaluateChatHook.
01/13 23:08:36 - mmengine - INFO - Sample output:
 <s><|User|>:请介绍一下你自己<eoh>
<|Bot|>:你好，我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我致力于通过执行常见的基于语言的任务和提供建议来帮助人类。我能够回答问题、提供定义和解释、将文本从一种语言翻译成

01/13 23:08:42 - mmengine - INFO - Sample output:
 <s><|User|>:请做一下自我介绍<eoh>
<|Bot|>:你好，我是人工智能助手，我的名字是书生·浦语。我由上海人工智能实验室开发，致力于通过执行常见的基于语言的任务和提供建议来帮助人类。我可以使用汉语和英语进行交流，并且可以回答问题、提供定义

01/13 23:08:42 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
01/13 23:08:42 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
01/13 23:08:42 - mmengine - INFO - Checkpoints will be saved to /data/personal_assistant/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy.
/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
/opt/conda/lib/python3.10/site-packages/mmengine/optim/scheduler/param_scheduler.py:198: UserWarning: Detected call of `scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `scheduler.step()`. Failure to do this will result in PyTorch skipping the first value of the parameter value schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
01/13 23:08:56 - mmengine - INFO - Epoch(train) [1][ 10/381]  lr: 1.9997e-04  eta: 0:26:58  time: 1.4288  data_time: 0.0035  memory: 10428  loss: 1.2449
01/13 23:09:11 - mmengine - INFO - Epoch(train) [1][ 20/381]  lr: 1.9986e-04  eta: 0:27:01  time: 1.4585  data_time: 0.0026  memory: 10428  loss: 1.1690  grad_norm: 0.5485
01/13 23:09:25 - mmengine - INFO - Epoch(train) [1][ 30/381]  lr: 1.9968e-04  eta: 0:26:39  time: 1.4233  data_time: 0.0026  memory: 10427  loss: 1.0350  grad_norm: 0.5485
01/13 23:09:39 - mmengine - INFO - Epoch(train) [1][ 40/381]  lr: 1.9943e-04  eta: 0:26:22  time: 1.4282  data_time: 0.0027  memory: 10427  loss: 0.9046  grad_norm: 0.3827
01/13 23:09:53 - mmengine - INFO - Epoch(train) [1][ 50/381]  lr: 1.9909e-04  eta: 0:26:06  time: 1.4266  data_time: 0.0026  memory: 10427  loss: 0.8518  grad_norm: 0.3306
01/13 23:10:08 - mmengine - INFO - Epoch(train) [1][ 60/381]  lr: 1.9869e-04  eta: 0:25:50  time: 1.4233  data_time: 0.0027  memory: 10427  loss: 0.7461  grad_norm: 0.3306
01/13 23:10:22 - mmengine - INFO - Epoch(train) [1][ 70/381]  lr: 1.9821e-04  eta: 0:25:35  time: 1.4286  data_time: 0.0026  memory: 10427  loss: 0.5961  grad_norm: 0.3150
01/13 23:10:36 - mmengine - INFO - Epoch(train) [1][ 80/381]  lr: 1.9765e-04  eta: 0:25:21  time: 1.4295  data_time: 0.0027  memory: 10427  loss: 0.5235  grad_norm: 0.3167
01/13 23:10:50 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:11:07 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我能够执行常见的基于语言的任务，如回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。我致力于通过执行这些任务和提供建议来帮助人类。但是，我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<eoa>
</s>

01/13 23:11:21 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:你好，我是AI助手，我的名字是书生·浦语。我由上海人工智能实验室开发，致力于通过执行常见的基于语言的任务和提供建议来帮助人类。我可以使用汉语和英语进行交流，并且可以回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但是，我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<eoa>
</s>

01/13 23:11:21 - mmengine - INFO - Epoch(train) [1][ 90/381]  lr: 1.9702e-04  eta: 0:25:05  time: 1.4244  data_time: 0.0026  memory: 10427  loss: 0.3597  grad_norm: 0.3167
01/13 23:11:36 - mmengine - INFO - Epoch(train) [1][100/381]  lr: 1.9632e-04  eta: 0:30:14  time: 4.5296  data_time: 3.1030  memory: 10427  loss: 0.3128  grad_norm: 0.2992
01/13 23:11:50 - mmengine - INFO - Epoch(train) [1][110/381]  lr: 1.9555e-04  eta: 0:29:28  time: 1.4259  data_time: 0.0027  memory: 10427  loss: 0.2761  grad_norm: 0.2992
01/13 23:12:04 - mmengine - INFO - Epoch(train) [1][120/381]  lr: 1.9470e-04  eta: 0:28:46  time: 1.4309  data_time: 0.0028  memory: 10427  loss: 0.2077  grad_norm: 0.2720
01/13 23:12:19 - mmengine - INFO - Epoch(train) [1][130/381]  lr: 1.9378e-04  eta: 0:28:09  time: 1.4289  data_time: 0.0027  memory: 10427  loss: 0.2296  grad_norm: 0.2548
01/13 23:12:33 - mmengine - INFO - Epoch(train) [1][140/381]  lr: 1.9279e-04  eta: 0:27:35  time: 1.4264  data_time: 0.0025  memory: 10427  loss: 0.1525  grad_norm: 0.2548
01/13 23:12:47 - mmengine - INFO - Epoch(train) [1][150/381]  lr: 1.9173e-04  eta: 0:27:04  time: 1.4310  data_time: 0.0025  memory: 10427  loss: 0.1467  grad_norm: 0.2418
01/13 23:13:01 - mmengine - INFO - Epoch(train) [1][160/381]  lr: 1.9060e-04  eta: 0:26:35  time: 1.4327  data_time: 0.0030  memory: 10427  loss: 0.1189  grad_norm: 0.2284
01/13 23:13:16 - mmengine - INFO - Epoch(train) [1][170/381]  lr: 1.8940e-04  eta: 0:26:08  time: 1.4267  data_time: 0.0025  memory: 10427  loss: 0.0925  grad_norm: 0.2284
01/13 23:13:30 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:13:31 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是书生·浦语的7.0版本</s>

01/13 23:14:45 - mmengine - INFO - Sample output:
 <s><|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunqinzhjunq

01/13 23:14:45 - mmengine - INFO - Epoch(train) [1][180/381]  lr: 1.8814e-04  eta: 0:25:42  time: 1.4319  data_time: 0.0025  memory: 10427  loss: 0.0767  grad_norm: 0.1802
01/13 23:14:59 - mmengine - INFO - Epoch(train) [1][190/381]  lr: 1.8681e-04  eta: 0:31:33  time: 8.9177  data_time: 7.4959  memory: 10427  loss: 0.0691  grad_norm: 0.1802
01/13 23:15:13 - mmengine - INFO - Epoch(train) [1][200/381]  lr: 1.8541e-04  eta: 0:30:47  time: 1.4281  data_time: 0.0027  memory: 10427  loss: 0.0589  grad_norm: 0.1633
01/13 23:15:28 - mmengine - INFO - Epoch(train) [1][210/381]  lr: 1.8395e-04  eta: 0:30:04  time: 1.4287  data_time: 0.0027  memory: 10427  loss: 0.0464  grad_norm: 0.1439
01/13 23:15:42 - mmengine - INFO - Epoch(train) [1][220/381]  lr: 1.8242e-04  eta: 0:29:23  time: 1.4244  data_time: 0.0029  memory: 10427  loss: 0.0409  grad_norm: 0.1439
01/13 23:15:56 - mmengine - INFO - Epoch(train) [1][230/381]  lr: 1.8084e-04  eta: 0:28:45  time: 1.4316  data_time: 0.0031  memory: 10427  loss: 0.0371  grad_norm: 0.1192
01/13 23:16:11 - mmengine - INFO - Epoch(train) [1][240/381]  lr: 1.7919e-04  eta: 0:28:09  time: 1.4316  data_time: 0.0030  memory: 10427  loss: 0.0259  grad_norm: 0.0893
01/13 23:16:25 - mmengine - INFO - Epoch(train) [1][250/381]  lr: 1.7748e-04  eta: 0:27:34  time: 1.4256  data_time: 0.0030  memory: 10427  loss: 0.0396  grad_norm: 0.0893
01/13 23:16:39 - mmengine - INFO - Epoch(train) [1][260/381]  lr: 1.7571e-04  eta: 0:27:01  time: 1.4325  data_time: 0.0031  memory: 10427  loss: 0.0295  grad_norm: 0.0709
01/13 23:16:53 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:16:56 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，书生·浦语的7B大模型</s>

01/13 23:17:00 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:17:00 - mmengine - INFO - Epoch(train) [1][270/381]  lr: 1.7389e-04  eta: 0:26:30  time: 1.4262  data_time: 0.0032  memory: 10427  loss: 0.0326  grad_norm: 0.0709
01/13 23:17:14 - mmengine - INFO - Epoch(train) [1][280/381]  lr: 1.7201e-04  eta: 0:26:19  time: 2.0528  data_time: 0.6229  memory: 10427  loss: 0.0388  grad_norm: 0.0620
01/13 23:17:28 - mmengine - INFO - Epoch(train) [1][290/381]  lr: 1.7008e-04  eta: 0:25:49  time: 1.4310  data_time: 0.0030  memory: 10427  loss: 0.0297  grad_norm: 0.0506
01/13 23:17:43 - mmengine - INFO - Epoch(train) [1][300/381]  lr: 1.6809e-04  eta: 0:25:19  time: 1.4251  data_time: 0.0030  memory: 10427  loss: 0.0204  grad_norm: 0.0506
01/13 23:17:57 - mmengine - INFO - Epoch(train) [1][310/381]  lr: 1.6605e-04  eta: 0:24:51  time: 1.4306  data_time: 0.0031  memory: 10427  loss: 0.0278  grad_norm: 0.0384
01/13 23:18:11 - mmengine - INFO - Epoch(train) [1][320/381]  lr: 1.6396e-04  eta: 0:24:24  time: 1.4314  data_time: 0.0030  memory: 10427  loss: 0.0260  grad_norm: 0.0292
01/13 23:18:25 - mmengine - INFO - Epoch(train) [1][330/381]  lr: 1.6183e-04  eta: 0:23:58  time: 1.4254  data_time: 0.0028  memory: 10427  loss: 0.0236  grad_norm: 0.0292
01/13 23:18:40 - mmengine - INFO - Epoch(train) [1][340/381]  lr: 1.5964e-04  eta: 0:23:32  time: 1.4314  data_time: 0.0025  memory: 10427  loss: 0.0214  grad_norm: 0.0247
01/13 23:18:54 - mmengine - INFO - Epoch(train) [1][350/381]  lr: 1.5741e-04  eta: 0:23:07  time: 1.4279  data_time: 0.0027  memory: 10427  loss: 0.0212  grad_norm: 0.0247
01/13 23:19:08 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:19:11 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:19:15 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:19:15 - mmengine - INFO - Epoch(train) [1][360/381]  lr: 1.5514e-04  eta: 0:22:43  time: 1.4297  data_time: 0.0025  memory: 10427  loss: 0.0239  grad_norm: 0.0215
01/13 23:19:29 - mmengine - INFO - Epoch(train) [1][370/381]  lr: 1.5283e-04  eta: 0:22:32  time: 2.0558  data_time: 0.6287  memory: 10427  loss: 0.0204  grad_norm: 0.0205
01/13 23:19:43 - mmengine - INFO - Epoch(train) [1][380/381]  lr: 1.5048e-04  eta: 0:22:08  time: 1.4253  data_time: 0.0025  memory: 10427  loss: 0.0179  grad_norm: 0.0205
01/13 23:19:45 - mmengine - INFO - Exp name: internlm_chat_7b_qlora_oasst1_e3_copy_20240113_230652
01/13 23:19:45 - mmengine - INFO - Saving checkpoint at 1 epochs
/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
01/13 23:20:04 - mmengine - INFO - Epoch(train) [2][ 10/381]  lr: 1.4784e-04  eta: 0:21:42  time: 1.4490  data_time: 0.0037  memory: 10427  loss: 0.0221  grad_norm: 0.0202
01/13 23:20:18 - mmengine - INFO - Epoch(train) [2][ 20/381]  lr: 1.4541e-04  eta: 0:21:19  time: 1.4293  data_time: 0.0025  memory: 10427  loss: 0.0146  grad_norm: 0.0193
01/13 23:20:32 - mmengine - INFO - Epoch(train) [2][ 30/381]  lr: 1.4295e-04  eta: 0:20:57  time: 1.4250  data_time: 0.0025  memory: 10427  loss: 0.0132  grad_norm: 0.0193
01/13 23:20:47 - mmengine - INFO - Epoch(train) [2][ 40/381]  lr: 1.4045e-04  eta: 0:20:35  time: 1.4291  data_time: 0.0026  memory: 10427  loss: 0.0169  grad_norm: 0.0181
01/13 23:21:01 - mmengine - INFO - Epoch(train) [2][ 50/381]  lr: 1.3792e-04  eta: 0:20:13  time: 1.4249  data_time: 0.0026  memory: 10427  loss: 0.0217  grad_norm: 0.0181
01/13 23:21:15 - mmengine - INFO - Epoch(train) [2][ 60/381]  lr: 1.3536e-04  eta: 0:19:51  time: 1.4297  data_time: 0.0025  memory: 10427  loss: 0.0158  grad_norm: 0.0181
01/13 23:21:30 - mmengine - INFO - Epoch(train) [2][ 70/381]  lr: 1.3278e-04  eta: 0:19:30  time: 1.4296  data_time: 0.0025  memory: 10427  loss: 0.0179  grad_norm: 0.0181
01/13 23:21:44 - mmengine - INFO - Epoch(train) [2][ 80/381]  lr: 1.3017e-04  eta: 0:19:09  time: 1.4253  data_time: 0.0025  memory: 10427  loss: 0.0199  grad_norm: 0.0181
01/13 23:21:58 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:22:01 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:22:03 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:22:03 - mmengine - INFO - Epoch(train) [2][ 90/381]  lr: 1.2754e-04  eta: 0:18:49  time: 1.4299  data_time: 0.0025  memory: 10427  loss: 0.0143  grad_norm: 0.0181
01/13 23:22:18 - mmengine - INFO - Epoch(train) [2][100/381]  lr: 1.2488e-04  eta: 0:18:36  time: 1.9579  data_time: 0.5308  memory: 10427  loss: 0.0166  grad_norm: 0.0180
01/13 23:22:32 - mmengine - INFO - Epoch(train) [2][110/381]  lr: 1.2221e-04  eta: 0:18:16  time: 1.4230  data_time: 0.0025  memory: 10427  loss: 0.0182  grad_norm: 0.0180
01/13 23:22:46 - mmengine - INFO - Epoch(train) [2][120/381]  lr: 1.1953e-04  eta: 0:17:56  time: 1.4273  data_time: 0.0025  memory: 10427  loss: 0.0109  grad_norm: 0.0177
01/13 23:23:00 - mmengine - INFO - Epoch(train) [2][130/381]  lr: 1.1682e-04  eta: 0:17:36  time: 1.4235  data_time: 0.0025  memory: 10427  loss: 0.0093  grad_norm: 0.0177
01/13 23:23:15 - mmengine - INFO - Epoch(train) [2][140/381]  lr: 1.1411e-04  eta: 0:17:16  time: 1.4278  data_time: 0.0026  memory: 10427  loss: 0.0224  grad_norm: 0.0172
01/13 23:23:29 - mmengine - INFO - Epoch(train) [2][150/381]  lr: 1.1138e-04  eta: 0:16:57  time: 1.4278  data_time: 0.0024  memory: 10427  loss: 0.0130  grad_norm: 0.0168
01/13 23:23:43 - mmengine - INFO - Epoch(train) [2][160/381]  lr: 1.0865e-04  eta: 0:16:37  time: 1.4239  data_time: 0.0025  memory: 10427  loss: 0.0147  grad_norm: 0.0168
01/13 23:23:57 - mmengine - INFO - Epoch(train) [2][170/381]  lr: 1.0591e-04  eta: 0:16:18  time: 1.4283  data_time: 0.0025  memory: 10427  loss: 0.0103  grad_norm: 0.0163
01/13 23:24:12 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:24:14 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:24:17 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:24:17 - mmengine - INFO - Epoch(train) [2][180/381]  lr: 1.0316e-04  eta: 0:15:59  time: 1.4282  data_time: 0.0026  memory: 10427  loss: 0.0128  grad_norm: 0.0163
01/13 23:24:31 - mmengine - INFO - Epoch(train) [2][190/381]  lr: 1.0041e-04  eta: 0:15:46  time: 1.9503  data_time: 0.5270  memory: 10427  loss: 0.0157  grad_norm: 0.0163
01/13 23:24:46 - mmengine - INFO - Epoch(train) [2][200/381]  lr: 9.7664e-05  eta: 0:15:27  time: 1.4308  data_time: 0.0025  memory: 10427  loss: 0.0122  grad_norm: 0.0165
01/13 23:25:00 - mmengine - INFO - Epoch(train) [2][210/381]  lr: 9.4917e-05  eta: 0:15:09  time: 1.4268  data_time: 0.0025  memory: 10427  loss: 0.0133  grad_norm: 0.0165
01/13 23:25:14 - mmengine - INFO - Epoch(train) [2][220/381]  lr: 9.2175e-05  eta: 0:14:50  time: 1.4311  data_time: 0.0025  memory: 10427  loss: 0.0091  grad_norm: 0.0165
01/13 23:25:28 - mmengine - INFO - Epoch(train) [2][230/381]  lr: 8.9438e-05  eta: 0:14:32  time: 1.4302  data_time: 0.0025  memory: 10427  loss: 0.0089  grad_norm: 0.0158
01/13 23:25:43 - mmengine - INFO - Epoch(train) [2][240/381]  lr: 8.6709e-05  eta: 0:14:14  time: 1.4240  data_time: 0.0025  memory: 10427  loss: 0.0063  grad_norm: 0.0158
01/13 23:25:57 - mmengine - INFO - Epoch(train) [2][250/381]  lr: 8.3990e-05  eta: 0:13:56  time: 1.4289  data_time: 0.0025  memory: 10427  loss: 0.0101  grad_norm: 0.0152
01/13 23:26:11 - mmengine - INFO - Epoch(train) [2][260/381]  lr: 8.1283e-05  eta: 0:13:38  time: 1.4289  data_time: 0.0026  memory: 10427  loss: 0.0105  grad_norm: 0.0156
01/13 23:26:26 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:26:28 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:26:31 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:26:31 - mmengine - INFO - Epoch(train) [2][270/381]  lr: 7.8591e-05  eta: 0:13:20  time: 1.4250  data_time: 0.0025  memory: 10427  loss: 0.0106  grad_norm: 0.0156
01/13 23:26:45 - mmengine - INFO - Epoch(train) [2][280/381]  lr: 7.5915e-05  eta: 0:13:06  time: 1.9588  data_time: 0.5313  memory: 10427  loss: 0.0065  grad_norm: 0.0149
01/13 23:26:59 - mmengine - INFO - Epoch(train) [2][290/381]  lr: 7.3256e-05  eta: 0:12:48  time: 1.4260  data_time: 0.0027  memory: 10427  loss: 0.0111  grad_norm: 0.0149
01/13 23:27:14 - mmengine - INFO - Epoch(train) [2][300/381]  lr: 7.0618e-05  eta: 0:12:31  time: 1.4302  data_time: 0.0026  memory: 10427  loss: 0.0097  grad_norm: 0.0155
01/13 23:27:28 - mmengine - INFO - Epoch(train) [2][310/381]  lr: 6.8002e-05  eta: 0:12:13  time: 1.4294  data_time: 0.0026  memory: 10427  loss: 0.0101  grad_norm: 0.0155
01/13 23:27:42 - mmengine - INFO - Epoch(train) [2][320/381]  lr: 6.5411e-05  eta: 0:11:56  time: 1.4258  data_time: 0.0025  memory: 10427  loss: 0.0106  grad_norm: 0.0155
01/13 23:27:57 - mmengine - INFO - Epoch(train) [2][330/381]  lr: 6.2845e-05  eta: 0:11:38  time: 1.4300  data_time: 0.0025  memory: 10427  loss: 0.0083  grad_norm: 0.0154
01/13 23:28:11 - mmengine - INFO - Epoch(train) [2][340/381]  lr: 6.0308e-05  eta: 0:11:21  time: 1.4310  data_time: 0.0027  memory: 10427  loss: 0.0083  grad_norm: 0.0151
01/13 23:28:25 - mmengine - INFO - Epoch(train) [2][350/381]  lr: 5.7800e-05  eta: 0:11:04  time: 1.4269  data_time: 0.0028  memory: 10427  loss: 0.0073  grad_norm: 0.0151
01/13 23:28:39 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:28:42 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:28:45 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:28:45 - mmengine - INFO - Epoch(train) [2][360/381]  lr: 5.5325e-05  eta: 0:10:47  time: 1.4318  data_time: 0.0028  memory: 10427  loss: 0.0074  grad_norm: 0.0147
01/13 23:29:00 - mmengine - INFO - Epoch(train) [2][370/381]  lr: 5.2883e-05  eta: 0:10:33  time: 2.0098  data_time: 0.5855  memory: 10427  loss: 0.0067  grad_norm: 0.0147
01/13 23:29:14 - mmengine - INFO - Epoch(train) [2][380/381]  lr: 5.0477e-05  eta: 0:10:16  time: 1.4292  data_time: 0.0028  memory: 10427  loss: 0.0114  grad_norm: 0.0143
01/13 23:29:15 - mmengine - INFO - Exp name: internlm_chat_7b_qlora_oasst1_e3_copy_20240113_230652
01/13 23:29:15 - mmengine - INFO - Saving checkpoint at 2 epochs
/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
01/13 23:29:34 - mmengine - INFO - Epoch(train) [3][ 10/381]  lr: 4.7873e-05  eta: 0:09:57  time: 1.4488  data_time: 0.0037  memory: 10427  loss: 0.0039  grad_norm: 0.0144
01/13 23:29:48 - mmengine - INFO - Epoch(train) [3][ 20/381]  lr: 4.5548e-05  eta: 0:09:40  time: 1.4247  data_time: 0.0027  memory: 10427  loss: 0.0082  grad_norm: 0.0144
01/13 23:30:03 - mmengine - INFO - Epoch(train) [3][ 30/381]  lr: 4.3263e-05  eta: 0:09:23  time: 1.4277  data_time: 0.0027  memory: 10427  loss: 0.0059  grad_norm: 0.0144
01/13 23:30:17 - mmengine - INFO - Epoch(train) [3][ 40/381]  lr: 4.1022e-05  eta: 0:09:06  time: 1.4285  data_time: 0.0027  memory: 10427  loss: 0.0037  grad_norm: 0.0137
01/13 23:30:31 - mmengine - INFO - Epoch(train) [3][ 50/381]  lr: 3.8824e-05  eta: 0:08:50  time: 1.4246  data_time: 0.0029  memory: 10427  loss: 0.0068  grad_norm: 0.0137
01/13 23:30:45 - mmengine - INFO - Epoch(train) [3][ 60/381]  lr: 3.6674e-05  eta: 0:08:33  time: 1.4302  data_time: 0.0032  memory: 10427  loss: 0.0079  grad_norm: 0.0142
01/13 23:31:00 - mmengine - INFO - Epoch(train) [3][ 70/381]  lr: 3.4571e-05  eta: 0:08:16  time: 1.4453  data_time: 0.0171  memory: 10427  loss: 0.0063  grad_norm: 0.0137
01/13 23:31:14 - mmengine - INFO - Epoch(train) [3][ 80/381]  lr: 3.2517e-05  eta: 0:08:00  time: 1.4254  data_time: 0.0031  memory: 10427  loss: 0.0091  grad_norm: 0.0137
01/13 23:31:28 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:31:32 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:31:35 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:31:35 - mmengine - INFO - Epoch(train) [3][ 90/381]  lr: 3.0514e-05  eta: 0:07:43  time: 1.4315  data_time: 0.0031  memory: 10427  loss: 0.0057  grad_norm: 0.0133
01/13 23:31:49 - mmengine - INFO - Epoch(train) [3][100/381]  lr: 2.8564e-05  eta: 0:07:29  time: 2.0972  data_time: 0.6744  memory: 10427  loss: 0.0064  grad_norm: 0.0133
01/13 23:32:04 - mmengine - INFO - Epoch(train) [3][110/381]  lr: 2.6668e-05  eta: 0:07:12  time: 1.4305  data_time: 0.0032  memory: 10427  loss: 0.0068  grad_norm: 0.0135
01/13 23:32:18 - mmengine - INFO - Epoch(train) [3][120/381]  lr: 2.4827e-05  eta: 0:06:56  time: 1.4298  data_time: 0.0032  memory: 10427  loss: 0.0052  grad_norm: 0.0137
01/13 23:32:32 - mmengine - INFO - Epoch(train) [3][130/381]  lr: 2.3043e-05  eta: 0:06:39  time: 1.4246  data_time: 0.0032  memory: 10427  loss: 0.0086  grad_norm: 0.0137
01/13 23:32:47 - mmengine - INFO - Epoch(train) [3][140/381]  lr: 2.1318e-05  eta: 0:06:23  time: 1.4314  data_time: 0.0032  memory: 10427  loss: 0.0081  grad_norm: 0.0137
01/13 23:33:01 - mmengine - INFO - Epoch(train) [3][150/381]  lr: 1.9651e-05  eta: 0:06:07  time: 1.4282  data_time: 0.0025  memory: 10427  loss: 0.0081  grad_norm: 0.0139
01/13 23:33:15 - mmengine - INFO - Epoch(train) [3][160/381]  lr: 1.8045e-05  eta: 0:05:50  time: 1.4275  data_time: 0.0027  memory: 10427  loss: 0.0078  grad_norm: 0.0139
01/13 23:33:29 - mmengine - INFO - Epoch(train) [3][170/381]  lr: 1.6502e-05  eta: 0:05:34  time: 1.4292  data_time: 0.0026  memory: 10427  loss: 0.0077  grad_norm: 0.0141
01/13 23:33:44 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:33:46 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:33:49 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:33:49 - mmengine - INFO - Epoch(train) [3][180/381]  lr: 1.5021e-05  eta: 0:05:18  time: 1.4248  data_time: 0.0025  memory: 10427  loss: 0.0033  grad_norm: 0.0141
01/13 23:34:03 - mmengine - INFO - Epoch(train) [3][190/381]  lr: 1.3604e-05  eta: 0:05:03  time: 1.9485  data_time: 0.5233  memory: 10427  loss: 0.0063  grad_norm: 0.0141
01/13 23:34:17 - mmengine - INFO - Epoch(train) [3][200/381]  lr: 1.2253e-05  eta: 0:04:47  time: 1.4285  data_time: 0.0025  memory: 10427  loss: 0.0073  grad_norm: 0.0139
01/13 23:34:32 - mmengine - INFO - Epoch(train) [3][210/381]  lr: 1.0968e-05  eta: 0:04:31  time: 1.4242  data_time: 0.0025  memory: 10427  loss: 0.0082  grad_norm: 0.0139
01/13 23:34:46 - mmengine - INFO - Epoch(train) [3][220/381]  lr: 9.7503e-06  eta: 0:04:14  time: 1.4280  data_time: 0.0025  memory: 10427  loss: 0.0090  grad_norm: 0.0139
01/13 23:35:00 - mmengine - INFO - Epoch(train) [3][230/381]  lr: 8.6008e-06  eta: 0:03:58  time: 1.4287  data_time: 0.0026  memory: 10427  loss: 0.0058  grad_norm: 0.0140
01/13 23:35:12 - mmengine - INFO - Exp name: internlm_chat_7b_qlora_oasst1_e3_copy_20240113_230652
01/13 23:35:15 - mmengine - INFO - Epoch(train) [3][240/381]  lr: 7.5203e-06  eta: 0:03:42  time: 1.4245  data_time: 0.0026  memory: 10427  loss: 0.0089  grad_norm: 0.0140
01/13 23:35:29 - mmengine - INFO - Epoch(train) [3][250/381]  lr: 6.5096e-06  eta: 0:03:26  time: 1.4287  data_time: 0.0025  memory: 10427  loss: 0.0071  grad_norm: 0.0143
01/13 23:35:43 - mmengine - INFO - Epoch(train) [3][260/381]  lr: 5.5696e-06  eta: 0:03:10  time: 1.4256  data_time: 0.0026  memory: 10427  loss: 0.0051  grad_norm: 0.0143
01/13 23:35:57 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:36:00 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:36:02 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:36:03 - mmengine - INFO - Epoch(train) [3][270/381]  lr: 4.7009e-06  eta: 0:02:54  time: 1.4294  data_time: 0.0024  memory: 10427  loss: 0.0075  grad_norm: 0.0137
01/13 23:36:17 - mmengine - INFO - Epoch(train) [3][280/381]  lr: 3.9042e-06  eta: 0:02:39  time: 1.9441  data_time: 0.5179  memory: 10427  loss: 0.0038  grad_norm: 0.0136
01/13 23:36:31 - mmengine - INFO - Epoch(train) [3][290/381]  lr: 3.1801e-06  eta: 0:02:23  time: 1.4245  data_time: 0.0025  memory: 10427  loss: 0.0105  grad_norm: 0.0136
01/13 23:36:45 - mmengine - INFO - Epoch(train) [3][300/381]  lr: 2.5291e-06  eta: 0:02:07  time: 1.4285  data_time: 0.0025  memory: 10427  loss: 0.0047  grad_norm: 0.0138
01/13 23:37:00 - mmengine - INFO - Epoch(train) [3][310/381]  lr: 1.9518e-06  eta: 0:01:51  time: 1.4284  data_time: 0.0025  memory: 10427  loss: 0.0072  grad_norm: 0.0135
01/13 23:37:14 - mmengine - INFO - Epoch(train) [3][320/381]  lr: 1.4485e-06  eta: 0:01:36  time: 1.4244  data_time: 0.0025  memory: 10427  loss: 0.0042  grad_norm: 0.0135
01/13 23:37:28 - mmengine - INFO - Epoch(train) [3][330/381]  lr: 1.0196e-06  eta: 0:01:20  time: 1.4292  data_time: 0.0024  memory: 10427  loss: 0.0065  grad_norm: 0.0129
01/13 23:37:42 - mmengine - INFO - Epoch(train) [3][340/381]  lr: 6.6557e-07  eta: 0:01:04  time: 1.4248  data_time: 0.0025  memory: 10427  loss: 0.0071  grad_norm: 0.0129
01/13 23:37:57 - mmengine - INFO - Epoch(train) [3][350/381]  lr: 3.8654e-07  eta: 0:00:48  time: 1.4292  data_time: 0.0024  memory: 10427  loss: 0.0056  grad_norm: 0.0132
01/13 23:38:11 - mmengine - INFO - after_train_iter in EvaluateChatHook.
01/13 23:38:14 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:38:16 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:38:16 - mmengine - INFO - Epoch(train) [3][360/381]  lr: 1.8276e-07  eta: 0:00:32  time: 1.4290  data_time: 0.0024  memory: 10427  loss: 0.0036  grad_norm: 0.0135
01/13 23:38:30 - mmengine - INFO - Epoch(train) [3][370/381]  lr: 5.4388e-08  eta: 0:00:17  time: 1.9375  data_time: 0.5158  memory: 10427  loss: 0.0063  grad_norm: 0.0135
01/13 23:38:45 - mmengine - INFO - Epoch(train) [3][380/381]  lr: 1.5109e-09  eta: 0:00:01  time: 1.4286  data_time: 0.0025  memory: 10427  loss: 0.0045  grad_norm: 0.0135
01/13 23:38:46 - mmengine - INFO - Exp name: internlm_chat_7b_qlora_oasst1_e3_copy_20240113_230652
01/13 23:38:46 - mmengine - INFO - Saving checkpoint at 3 epochs
01/13 23:38:50 - mmengine - INFO - after_train in EvaluateChatHook.
01/13 23:38:53 - mmengine - INFO - Sample output:
 <s> <|User|>:请介绍一下你自己<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>

01/13 23:38:56 - mmengine - INFO - Sample output:
 <s> <|User|>:请做一下自我介绍<eoh>
<|Bot|>:我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型</s>
```

## 验证

```
# xtuner chat ./merged --prompt-template internlm_chat
[2024-01-14 11:15:26,199] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
[2024-01-14 11:15:30,809] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:39<00:00, 12.45s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 请介绍一下你自己

我是zhjunqin的小助手，基于上海AI实验室书生·浦语的7B大模型

```

## 进阶作业

后面再研究一下。