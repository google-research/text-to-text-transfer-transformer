# T5: 统一文本到文本迁移 Transformer (Text-To-Text Transfer Transformer)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

### 截至 2022 年 7 月，我们推荐使用 T5X：

[T5X](https://github.com/google-research/t5x) 是 T5（以及更多模型）在 JAX 和 Flax 中的全新改进实现。
基于 TensorFlow 与 MeshTF 的 T5 已不再积极开发。如果您是 T5 的新用户，我们建议直接从 [T5X](https://github.com/google-research/t5x) 开始。

[![Build Status](https://github.com/google-research/text-to-text-transfer-transformer/workflows/build/badge.svg)](https://github.com/google-research/text-to-text-transfer-transformer/actions?query=workflow%3Abuild)

`t5` 库的主要作用是提供用于复现论文《[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer][paper]》中实验的代码。在论文中，我们展示了如何使用在大规模文本语料库上预训练的文本到文本 Transformer 模型在多个自然语言处理（NLP）任务上取得最前沿（State-of-the-Art）的结果。

本仓库的大部分代码用于数据集的加载、预处理、混合与评估。
它还提供了一种对随论文一同发布的[预训练模型检查点](#released-model-checkpoints)进行微调的方法。

`t5` 库可用于未来的模型研发，提供了在文本到文本混合任务上训练和微调（潜在的*超大规模*）模型的实用模块。

## 目录 (Table of Contents)

* [代码库架构](#library)
* [使用指南](#usage)
  * [数据集准备](#dataset-preparation)
    * [C4 数据集](#c4)
  * [安装指南](#installation)
  * [在 GCP 上配置 TPU](#setting-up-tpus-on-gcp)
  * [模型训练](#training)
  * [模型微调](#fine-tuning)
  * [模型评估](#eval)
  * [文本解码与预测](#decode)
  * [模型导出](#export)
  * [GPU 使用指南](#gpu-usage)
  * [复现我们的实验](#reproducing-our-experiments)
  * [实用配置选项](#useful-options)
* [开源模型检查点](#released-model-checkpoints)
* [引用方式](#how-to-cite)

<a id="library"></a>
## 代码库架构 (Library)

#### t5.data

`t5.data` 是一个用于定义提供 `tf.data.Dataset` 的 `Task` 对象的包。

每个 `Task` 由以下部分组成：

  * 数据源（data source）
  * 文本预处理器函数（text preprocessor function(s)）
  * SentencePiece 分词模型（SentencePiece model）
  * 评估指标函数（metric function(s)）

此外，还可以选择性提供：

  * 词元预处理器函数（token preprocessor function(s)）
  * 后处理函数（postprocess function(s)）

**数据源**可以是一个提供 `tf.data.Dataset` 的任意函数，但我们也为 [TensorFlow Datasets (TFDS)][tfds] 中提供的数据集（`TfdsTask`）或每行存储一条样本的文本文件（`TextLineTask`）提供了更简便的包装器。

**文本预处理器**将源数据集中的样本转换为带有 `inputs` 和 `targets` 字段的、适用于文本到文本模型的标准格式。例如，预定义的 `t5.data.preprocessors.translate` 预处理器将以下格式的输入：

```py
{'de': 'Das ist gut.', 'en': 'That is good.'}
```

转换为如下格式：

```py
{'inputs': 'translate German to English: Das ist gut.', 'targets': 'That is good.'}
```

除文本预处理外，您还可以使用一个或多个**词元预处理器**对分词后的输入进行修改。我们便是使用这些词元预处理器实现了论文中的无监督预训练目标。

我们在 `t5.data.preprocessors` 中提供了许多预定义的预处理器，但您也可以根据需要自定义预处理器。

**SentencePiece 模型**用于对输入字符串进行分词，并对输出词元进行解码。您可以使用 [google/sentencepiece](https://github.com/google/sentencepiece) 库创建自己的模型，也可以直接使用位于 `t5.data.DEFAULT_SPM_PATH` 的默认模型。如果您训练自己的分词模型，在运行 `spm_train` 时必须指定 `--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1` 标志，以便与我们的模型代码保持兼容。

**评估指标函数**根据模型给出的预测与真实目标计算并返回得分。您还可以定义**后处理函数**，在调用评估指标之前将目标文本和预测文本转换为另一种格式。我们在 `t5.evaluation.metrics` 中提供了一些预定义的评估指标。

最后，`t5.data` 包含一个 `Mixture` 类，可用于组合多个 `Task` 数据集以进行多任务训练，支持通过多种策略函数指定各任务的混合比例。

#### t5.evaluation

`t5.evaluation` 包含两个核心组件：

1. 评估过程中使用的各项指标（metrics）
2. 在评估阶段计算并应用这些指标的实用工具

#### t5.models

`t5.models` 包含了将 T5 `Task` 和 `Mixture` 与具体的模型实现相连接的适配层（shims），用于训练、评估和推理。

目前提供了两个适配层：一个用于我们在论文中使用的 [Mesh TensorFlow Transformer][mtft]，另一个用于 [Hugging Face Transformers 库](https://huggingface.co/transformers/)。
Hugging Face API 目前属于实验性质且可能会发生变动，但它提供了一种在单张 GPU 上使用 PyTorch 加载、微调和评估我们预训练模型的简便方法。
如果您希望在 TPU 上使用我们最大规格的模型，或者复现论文中的结果，建议使用 [MtfModel](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/models/mtf_model.py) API 及 `t5_mesh_transformer` 二进制程序。
如果您希望在单张 GPU 上使用 PyTorch 微调模型，可以尝试 [HfPyTorchModel](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/models/hf_model.py) API。
由于 HfPyTorchModel 属于实验性支持，本文档后续内容均默认基于 MtfModel 及其对应的二进制程序进行说明。
HfPyTorchModel 的使用示例参见[此处](https://github.com/google-research/text-to-text-transfer-transformer/blob/a08f0d1c4a7caa6495aec90ce769a29787c3c87c/t5/models/hf_model.py#L38)。

<a id="usage"></a>
## 使用指南 (Usage)

体验 T5 最简单的方式是在我们的 [Colab 交互式教程](https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-trivia.ipynb)中使用免费的 TPU 算力。

以下我们提供了如何使用本代码库通过命令行进行预训练、微调、评估与文本生成的示例。您可以遵循这些说明复现我们的实验结果、使用自己的数据或超参数微调开源检查点，或者从零开始预训练模型。

<a id="dataset-preparation"></a>
### 数据集准备 (Dataset Preparation)

您可以使用新定义或现有的 `Task`，也可以直接从预处理好的 TSV 文件中加载样本。

#### 使用 `Task`

根据您的数据源（参见[上文](#t5data)），您需要适当地准备数据：

##### `Task`

如果使用基础 Task，只需确保 `dataset_fn` 加载的所有文件均可供 TPU 访问（即位于 GCS 存储桶中）即可！

##### `TfdsTask`

我们预定义的大多数 `Task` 均使用 [TensorFlow Datasets (TFDS)][tfds] 作为其数据源。当您使用 `TfdsTask` 运行我们的训练程序（参见[下文](#training)说明）时，数据集将在首次使用时自动下载并进行准备。准备完成后，数据集将缓存到您的本地存储中，以避免在后续运行中产生重复开销。如果在云端环境运行，建议设置 `--t5_tfds_data_dir` 标志指向持久化存储路径，例如 [GCS 存储桶][gcs]。在 TPU 上进行训练时，这是强制要求。

<a id="c4"></a>
#### C4 数据集

我们在无监督预训练中创建的 [C4][c4] 数据集已集成在 TensorFlow Datasets 中，但下载原始 [Common Crawl][cc] 抓取数据（约 7 TB）需要消耗可观的网络带宽，且数据准备需要大量的计算资源（约 335 个 CPU-天）。建议利用 TFDS 中的 [Apache Beam][beam] 分布式支持，在 [Google Cloud Dataflow][gcd] 上进行分布式预处理。使用 500 个工作节点（workers）时，整个处理任务约在 16 小时内完成。

在正确设置 `MY_PROJECT` 和 `MY_BUCKET` 环境变量后，您可以在 GCP 上通过以下命令在 DataFlow 中构建该数据集：

```sh
pip install tfds-nightly[c4]
echo 'tfds-nightly[c4]' > /tmp/beam_requirements.txt
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4/en \
  --data_dir=gs://$MY_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"
```

更多细节请参阅 [TFDS Beam 说明文档][tfds_beam]。

##### `TextLineTask`

当您的数据源为按行存储样本的文本文件时，`TextLineTask` 非常有用。您可以使用文本预处理器将每一行转换为包含 inputs 和 targets 的字典。

同样，请确保您的文件可供 TPU 访问（即存放在 GCS 存储桶中）。

<a id="using-a-tsv-file-directly"></a>
#### 直接使用 TSV 文件

除了定义新的 `Task` 之外，您还可以直接使用 TSV 文件作为数据集，其中每行的格式均为 `<input>\t<target>`。

但需要注意以下几点限制：

  * 无法额外定义文本处理器，因此 TSV 文件本身必须已经是预处理好的文本格式。
  * 目前直接使用 TSV 文件时，无法设置词元预处理器、后处理函数或评估指标函数。

如果您需要上述任何特性，则必须定义新的 `Task`、`TfdsTask` 或 `TextLineTask`。

与上述情况类似，您的 TSV 文件必须能够被 TPU 访问（即存放在 GCS 存储桶中）。

<a id="installation"></a>
### 安装指南 (Installation)

要安装 T5 软件包，只需运行：

```sh
pip install t5[gcp]
```

<a id="setting-up-tpus-on-gcp"></a>
### 在 GCP 上配置 TPU (Setting up TPUs on GCP)

您首先需要在 Google Cloud 上启动一台虚拟机（VM）。有关启动 VM 的详细说明请参阅 [Google Cloud 文档](https://cloud.google.com/compute/docs/instances/create-start-instance)。

要在 Cloud TPU 上运行训练或评估，您必须根据自己的项目、可用区及 GCS 存储桶正确配置以下环境变量。详情请参考 [Cloud TPU 快速入门](https://cloud.google.com/tpu/docs/quickstart)指南。

```sh
export PROJECT=your_project_name
export ZONE=your_project_zone
export BUCKET=gs://yourbucket/
export TPU_NAME=t5-tpu
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"
```

请在 Cloud VM 中运行以下命令创建 TPU 设备：

```sh
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```

<a id="training"></a>
### 模型训练 (Training)

在以下命令中，我们从零开始在 [GLUE 基准测试](https://gluebenchmark.com/) 的 MRPC 任务上训练一个模型。您可以修改 `MIXTURE_NAME` gin 参数，以使用我们代码库中提供的任意任务或混合任务集：

```sh
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'"
```

可以通过运行以下命令获取完整的任务与混合集列表：

```sh
python -c "import t5; print(t5.data.MixtureRegistry.names())"
```

您还可以在新文件中定义其他任务和混合集，并通过 `--module_import` 标志进行导入。

此外，您也可以使用格式为 `<input>\t<target>` 的 TSV 文件进行训练（参见[上文](#using-a-tsv-file-directly)）。

<a id="fine-tuning"></a>
### 模型微调 (Fine-tuning)

为了微调我们的[预训练模型检查点](#released-model-checkpoints)，您需要将预训练模型的有效配置文件（operative config）传递给训练脚本。有效配置应通过 `gin_file` 标志传入，其中指定了模型架构及其他超参数。此外，您还需要指定微调所用的任务混合集。例如，要在 `glue_mrpc_v002` 混合集上微调 T5-small 模型，请运行：

```sh
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin"
```

正确的预训练检查点路径已包含在 operative config 之中。

您还可以在新文件中定义自定义任务和混合集，并通过 `--module_import` 标志进行导入。

此外，您也可以使用格式为 `<input>\t<target>` 的 TSV 文件进行微调（参见[上文](#using-a-tsv-file-directly)）。例如，可以使用 WMT '19 [News Commentary 14](http://data.statmt.org/news-commentary/v14/training/) 训练集中的双语翻译数据集（例如 [英语-法语](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-fr.tsv.gz)）。使用 TSV 文件时，需要将 `MIXTURE_NAME` 替换为：

```sh
--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn"
--gin_param="tsv_dataset_fn.filename = 'gs:/path/to/tsv'"
```

若要采用我们在[论文][paper]中使用的相同超参数进行微调（使用 0.001 的恒定学习率），可以传入 T5 包中包含的如下 gin 配置文件：

```
--gin_file="learning_rate_schedules/constant_0_001.gin"
```

预训练模型的 operative config 默认设置为实际上不对训练步数进行限制。如果您希望微调特定的步数，需要显式指定该参数。由于预训练模型已经训练了 1,000,000 步，您应当指定预训练与微调的总累计步数。例如，如果您希望额外微调 10,000 步，应传入：

```
--gin_param="run.train_steps = 1010000"
```

您还可以在微调时指定不同的批次大小（Batch Size）。我们根据批次中的总词元数（tokens）来设置批次大小。默认情况下，一个批次使用的序列长度为 512。若要设置批次中的词元数量，可以配置：

```
--gin_param = "tokens_per_batch=1048576"
```

<a id="eval"></a>
### 模型评估 (Eval)

若要在 T5 框架中评估模型，需要使用 `eval.gin` 文件，指定模型目录、解码方法以及待评估的检查点步数。例如，若要在 GLUE MRPC 任务上使用束搜索（Beam Search）评估*所有*检查点，请运行以下命令：

```sh
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --t5_tfds_data_dir=${DATA_DIR} \
  --gin_file="eval.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_param="eval_checkpoint_step = 'all'"
```

若要评估特定的检查点，只需将 `eval_checkpoint_step` 参数设置为对应的步数：

```
--gin_param="eval_checkpoint_step = 100000"
```

您还可以在上述命令中使用 `greedy_decode.gin` 或 `sample_decode.gin` 替代 `beam_search.gin`。

<a id="decode"></a>
### 文本解码与预测 (Decode)

若要在 T5 框架中让模型生成预测文本，您需要指定模型目录、解码方法以及用于解码的检查点步数。假设您的输入序列文本文件存放在 `/path/to/inputs.txt`，示例命令如下：

```sh
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="input_filename = '/path/to/inputs.txt'"\
  --gin_param="output_filename = '/tmp/outputs.txt'"\
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"\
  --gin_param="infer_checkpoint_step = 'all'"
```

若要使用特定的检查点进行推理预测，只需设置：

```
--gin_param="infer_checkpoint_step = 100000"
```

您同样可以在上述命令中使用 `beam_search.gin` 或 `greedy_decode.gin` 替代 `sample_decode.gin`。

<a id="export"></a>
### 模型导出 (Export)

您可能还希望导出 [`SavedModel`](https://www.tensorflow.org/guide/saved_model) 格式，以便在线部署模型（例如使用 [ML Engine](https://cloud.google.com/ml-engine/docs/deploying-models) 或打包至 [Docker](https://docs.docker.com) 镜像中）：

```sh
t5_mesh_transformer \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --use_model_api \
  --mode="export_predict" \
  --export_dir="/path/to/export/dir"
```

上述命令会导出模型目录中的最新检查点。若要导出特定的检查点，请添加以下标志：

```sh
  --checkpoint_mode="specific" \
  --checkpoint_steps=1000000
```

[t5-deploy 笔记本](https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-deploy.ipynb) 演示了如何导出 `SavedModel` 并将其打包到 Docker 镜像中以供线上部署。

<a id="gpu-usage"></a>
### GPU 使用指南 (GPU Usage)

如果您希望使用 GPU 代替 TPU，可以修改上述命令：移除 TPU 特定的标志（`--tpu`、`--tpu_zone`、`--gcp_project`），并根据您的硬件拓扑设置 `mesh_shape` 和 `mesh_devices` 的 gin 参数。

例如，如果您的机器拥有 6 块 GPU，且您希望进行 3 路模型并行和 2 路数据并行，则上述微调命令修改为：

```sh
t5_mesh_transformer  \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.run.mesh_shape = 'model:3,batch:2'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1','gpu:2','gpu:3','gpu:4','gpu:5']" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin"
```

若使用单张 GPU，命令为：

```sh
t5_mesh_transformer  \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin"
```

<a id="reproducing-our-experiments"></a>
### 复现我们的实验 (Reproducing our experiments)

我们在 [gs://t5-data/experiments](https://console.cloud.google.com/storage/browser/t5-data/experiments) 中提供了[论文][paper]中所有实验的 operative config 配置文件。
`experiments` 文件夹下包含对应于论文不同章节的子目录。
例如，[gs://t5-data/experiments/objectives](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives) 包含了第 3.3 节（“无监督目标”）的所有实验。
`objectives` 文件夹的每个子目录包含特定实验的配置（宽泛而言，“实验”对应于论文表格中的一行）。

假设您想复现“前缀语言建模（Prefix language modeling）”目标（论文表 4 的第一行）。
该实验的配置文件位于 [gs://t5-data/experiments/objectives/obj-prefix_lm](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm)。
在根目录下，有一个用于预训练模型的配置文件（[gs://t5-data/experiments/objectives/obj-prefix_lm/operative_config.gin](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm/operative_config.gin)）。
接下来，每个下游微调混合任务都有各自的子目录，包含独立的微调配置（例如 [gs://t5-data/experiments/objectives/obj-prefix_lm/cnn_dailymail_v002/operative_config.gin](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm/cnn_dailymail_v002/operative_config.gin)）。
要运行此实验，首先使用预训练配置预训练模型：

```sh
export PRETRAIN_MODEL_DIR="${BUCKET}/obj-prefix_lm"
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${PRETRAIN_MODEL_DIR}" \
  --gin_file="gs://t5-data/experiments/objectives/obj-prefix_lm/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"
```

随后，即可在 CNN/Daily Mail 上对预训练模型进行微调：

```sh
export FINETUNE_MODEL_DIR="${BUCKET}/obj-prefix_lm/cnn_dailymail_v002"
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${FINETUNE_MODEL_DIR}" \
  --gin_file="gs://t5-data/experiments/objectives/obj-prefix_lm/cnn_dailymail_v002/operative_config.gin" \
  --gin_param="init_checkpoint = '${PRETRAIN_MODEL_DIR}/model.ckpt-524288'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"
```

<a id="useful-options"></a>
### 实用配置选项 (Useful Options)

某些训练变体需要同时设置多个参数标志。对于以下各变体，可将参数组添加到 `./third_party/py/t5/google/scripts/run_finetune.sh`：

*确定性训练 (Deterministic training)*

```sh
  --train_gin_param="mesh_train_dataset_fn.seed=${SEED}" \
  --train_gin_param="utils.run.skip_seen_data = True" \
```

*语言模型 (Language model)*

```sh
  --objective="lm" \
  --train_gin_param="utils.run.model_type = \"lm\"" \
```

<a id="released-model-checkpoints"></a>
## 开源模型检查点 (Released Model Checkpoints)

我们开源了[论文][paper]中介绍的以下预训练模型检查点：

* **T5-Small**（6000 万参数）：[gs://t5-data/pretrained_models/small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/small/)
* **T5-Base**（2.2 亿参数）：[gs://t5-data/pretrained_models/base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/base/)
* **T5-Large**（7.7 亿参数）：[gs://t5-data/pretrained_models/large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/large/)
* **T5-3B**（30 亿参数）：[gs://t5-data/pretrained_models/3B](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/3B/)
* **T5-11B**（110 亿参数）：[gs://t5-data/pretrained_models/11B](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/11B/)

更多额外的实验性预训练模型检查点列表请参阅[此处文档][released_checkpoints]。

<a id="how-to-cite"></a>
## 引用方式 (How to Cite)

如果您在工作中基于本研究进行了扩展或使用了本代码库，请引用介绍 T5 的原始[论文][paper]：

```bibtex
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
```

[paper]: https://arxiv.org/abs/1910.10683
[released_checkpoints]: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md
[beam]: https://beam.apache.org
[c4]: https://www.tensorflow.org/datasets/catalog/c4
[cc]: https://commoncrawl.org
[dataflow]: https://cloud.google.com/dataflow/
[gcs]: https://www.tensorflow.org/datasets/gcs
[gcd]: https://cloud.google.com/dataflow/
[gin]: https://github.com/google/gin-config
[mtft]: https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer
[tfds]: https://www.tensorflow.org/datasets
[tfds_beam]: https://www.tensorflow.org/datasets/beam_datasets

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月19日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
