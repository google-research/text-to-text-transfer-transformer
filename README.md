# T5: Text-To-Text Transfer Transformer


### As of July 2022, we recommend using T5X:

[T5X](https://github.com/google-research/t5x) is the new and improved implementation of T5 (and more) in JAX and Flax.
T5 on Tensorflow with MeshTF is no longer actively developed. If you are new
to T5, we recommend starting with [T5X](https://github.com/google-research/t5x).

[![Build Status](https://github.com/google-research/text-to-text-transfer-transformer/workflows/build/badge.svg)](https://github.com/google-research/text-to-text-transfer-transformer/actions?query=workflow%3Abuild)

The `t5` library serves primarily as code for reproducing the experiments in [_Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_][paper]. In the paper, we demonstrate how to achieve state-of-the-art results on multiple NLP tasks using a text-to-text transformer pre-trained on a large text corpus.

The bulk of the code in this repository is used for loading, preprocessing, mixing, and evaluating datasets.
It also provides a way to fine-tune the [pre-trained models](#released-model-checkpoints) released alongside the publication.

The `t5` library can be used for future model development by providing useful modules for training and fine-tuning (potentially *huge*) models on mixtures of text-to-text tasks.

## Table of Contents

* [Library](#library)
* [Usage](#usage)
  * [Dataset Preparation](#dataset-preparation)
    * [C4](#c4)
  * [Installation](#installation)
  * [Setting up TPUs on GCP](#setting-up-tpus-on-gcp)
  * [Training](#training)
  * [Fine-Tuning](#fine-tuning)
  * [Eval](#eval)
  * [Decode](#decode)
  * [Export](#export)
  * [GPU Usage](#gpu-usage)
  * [Reproducing our experiments](#reproducing-our-experiments)
  * [Useful Options](#useful-options)
* [Released Model Checkpoints](#released-model-checkpoints)
* [How to Cite](#how-to-cite)

## Library

#### t5.data

`t5.data` is a package for defining `Task` objects that provide `tf.data.Dataset`s.

Each `Task` is made up of:

  * a data source
  * text preprocessor function(s)
  * a SentencePiece model
  * metric function(s)

Additionally, you may optionally provide:

  * token preprocessor function(s)
  * postprocess function(s)

The **data source** can be an arbitrary function that provides a `tf.data.Dataset`, but we also provide simpler wrappers for datasets available in [TensorFlow Datasets (TFDS)][tfds] (a `TfdsTask`) or stored as text files with one example per line (a `TextLineTask`).

The **text preprocessor** converts the examples in the source dataset into the appropriate format for a text-to-text model with fields for `inputs` and `targets`.  For example, the predefined `t5.data.preprocessors.translate` preprocessor converts inputs in the form

```py
{'de': 'Das ist gut.', 'en': 'That is good.'}
```

to the form

```py
{'inputs': 'translate German to English: Das ist gut.', 'targets': 'That is good.'}
```

In addition to text preprocessing, you can also use one or more **token preprocessors** to modify the inputs post-tokenization. We implemented our unsupervised pre-training objectives using these token preprocessors.

We provide many predefined preprocessors in `t5.data.preprocessors`, but you may also define your own.

The **SentencePiece model** is used to tokenize the input strings and decode the output tokens. You can create your own model with the [google/sentencepiece](https://github.com/google/sentencepiece) library, or use our default one at `t5.data.DEFAULT_SPM_PATH`. If you create your own, you must use the flags `--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1` with `spm_train` to be compatible with our model code.

The **metric function** returns a score given the target and prediction from the model. You may also define a **postprocess function** to convert the target and prediction text to another format before calling the metric. We provide some predefined metrics in `t5.evaluation.metrics`.

Finally, `t5.data` contains a `Mixture` class that can be instantiated to combine multiple `Task` datasets for multi-task training using various functions for specifying the mixture rates.

#### t5.evaluation

`t5.evaluation` contains two core components:

1. metrics to be used during evaluation
2. utilities for applying these metrics at evaluation time

#### t5.models

`t5.models` contains shims for connecting T5 `Tasks` and `Mixtures` to a model implementation for training, evaluation, and inference.

Currently there are two shims available: One for the [Mesh TensorFlow Transformer][mtft] that we used in our paper and another for the [Hugging Face Transformers library](https://huggingface.co/transformers/).
The Hugging Face API is currently experimental and subject to change, but provides a simple and easy way to load, fine-tune, and evaluate our pre-trained models using PyTorch on a single GPU.
If you want to use our largest models on TPUs and/or reproduce the results in our paper, you should use the [MtfModel](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/models/mtf_model.py) API and the `t5_mesh_transformer` binary.
If you are interested fine-tuning our models on a GPU in PyTorch, you should try the [HfPyTorchModel](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/models/hf_model.py) API.
Since the HfPyTorchModel is experimental, the remainder of this README assumes usage of the MtfModel and its associated binary.
A usage example of HfPyTorchModel is available [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/a08f0d1c4a7caa6495aec90ce769a29787c3c87c/t5/models/hf_model.py#L38).

## Usage

The easiest way to try out T5 is with a free TPU in our [Colab Tutorial](https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-trivia.ipynb).

Below we provide examples for how to pre-train, fine-tune, evaluate, and decode from a model from the command-line with our codebase. You can use these instructions to reproduce our results, fine-tune one of our released checkpoints with your own data and/or hyperparameters, or pre-train a model from scratch.

### Dataset Preparation

You may either use a new or pre-existing `Task`, or you may load examples from a preprocessed TSV file.

#### Using a `Task`

Depending on your data source (see [above](#t5data)), you will need to prepare your data appropriately.

##### `Task`

If using a vanilla task, just make sure any file(s) loaded by your `dataset_fn` are accessible to the TPU (i.e., are in a GCS bucket), and you should be good to go!

##### `TfdsTask`

Most of our predefined `Task`s use [TensorFlow Datasets (TFDS)][tfds] as their data source. When you run our training binary (see instructions [below](#training)) with a `TfdsTask`, the dataset will automatically be downloaded and prepared on its first use. After preparation is complete, the dataset is cached to your local storage to avoid this overhead in future runs.  If working in the cloud, we recommend you set the `--t5_tfds_data_dir` flag to point to a persistent storage location, such as a [GCS bucket][gcs]. This is a requirement when training on TPU.

#### C4

The [C4][c4] dataset we created for unsupervised pre-training is available in TensorFlow Datasets, but it requires a significant amount of bandwidth for downloading the raw [Common Crawl][cc] scrapes (~7 TB) and compute for its preparation (~335 CPU-days). We suggest you take advantage of the [Apache Beam][beam] support in TFDS, which enables distributed preprocessing of the dataset and can be run on [Google Cloud Dataflow][gcd]. With 500 workers, the job should complete in ~16 hours.

After defining `MY_PROJECT` and `MY_BUCKET` appropriately, you can build the dataset in DataFlow from GCP using the following commands:

```sh
pip install tfds-nightly[c4]
echo 'tfds-nightly[c4]' > /tmp/beam_requirements.txt
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4/en \
  --data_dir=gs://$MY_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"
```

Read more in the [TFDS Beam instructions][tfds_beam].

##### `TextLineTask`

A `TextLineTask` is useful when your data source is a text file (or files) with one example per line. You can then use a text preprocessor to convert each line into a dictionary of inputs and targets.

Make sure your files are accessible to the TPU (i.e., are in a GCS bucket), and you should be good to go!

#### Using a TSV File Directly

Instead of defining a new `Task`, you may use a TSV file (or files) directly as your dataset where each line is formatted as `<input>\t<target>`.

However, there are a couple of caveats:

  * There is no way to define a text processor, so the TSV will need to contain your data in a preprocessed format.
  * There is also currently no way to set a token preprocessor, postprocess function, or metric function for evaluation when using a TSV file directly.

If you need any of these features, you must define a new `Task`, `TfdsTask`, or `TextLineTask`.

Similar to the above cases, your TSV file(s) must be accessible to the TPU (i.e., are in a GCS bucket).

### Installation

To install the T5 package, simply run:

```sh
pip install t5[gcp]
```

### Setting up TPUs on GCP

You will first need to launch a Virtual Machine (VM) on Google Cloud. Details about launching the VM can be found at the [Google Cloud Documentation](https://cloud.google.com/compute/docs/instances/create-start-instance).

In order to run training or eval on Cloud TPUs, you must set up the following variables based on your project, zone and GCS bucket appropriately. Please refer to the [Cloud TPU Quickstart](https://cloud.google.com/tpu/docs/quickstart) guide for more details.

```sh
export PROJECT=your_project_name
export ZONE=your_project_zone
export BUCKET=gs://yourbucket/
export TPU_NAME=t5-tpu
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"
```

Please use the following command to create a TPU device in the Cloud VM.

```sh
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```


### Training

In the command below, we train a model on the [GLUE Benchmark](https://gluebenchmark.com/) MRPC task from scratch. You can change the `MIXTURE_NAME` gin parameter to use any of the tasks or mixtures provided in our package.

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

The full list of tasks and mixtures can be obtained by running:

```sh
python -c "import t5; print(t5.data.MixtureRegistry.names())"
```

You may also define additional tasks and mixtures in a new file and import it using the `--module_import` flag.

Alternatively, you could train with a TSV file where each line is formatted as `<input>\t<target>` (see [above](#using-a-tsv-file-directly)).

### Fine-tuning

In order to fine-tune one of our [pre-trained models](#released-model-checkpoints), you need to pass the operative config of the pre-trained model to the training script. The operative config should be passed in as a `gin_file` flag. It specifies the model architecture and other hyperparameters. In addition, you need to specify the mixture to fine-tune on. For example, to fine-tune the T5-small model on the `glue_mrpc_v002` mixture, please run:

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

The correct pre-trained checkpoint path is included in the operative config.

You may also define additional tasks and mixtures in a new file and import it using the `--module_import` flag.

Alternatively, you could fine-tune with a TSV file where each line is formatted as `<input>\t<target>` (see [above](#using-a-tsv-file-directly)). For example, you could try one of the paired translation datasets from WMT '19 [News Commentary 14](http://data.statmt.org/news-commentary/v14/training/) training set
(e.g., [English-French](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-fr.tsv.gz)). When using a TSV file, you would replace the `MIXTURE_NAME` flag with:

```sh
--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn"
--gin_param="tsv_dataset_fn.filename = 'gs:/path/to/tsv'"
```

To fine-tune with the same hyperparameters we used in the [paper][paper] (using a constant learning rate of 0.001), you can pass in this gin file which is included in the T5 package:

```
--gin_file="learning_rate_schedules/constant_0_001.gin"
```

The operative config for the pre-trained models are set so that there is effectively no limit on the number of train steps. If you'd like to train for a specific number of steps, you'll need to pass that in. Since the pre-trained model has already been trained for 1,000,000 steps, you should specify the total number of steps after pre-training and fine-tuning. For example, if you want to fine-tune for an additional 10,000 steps, you should pass

```
--gin_param="run.train_steps = 1010000"
```

You can also use a different batch size for fine-tuning. We set the batch size according to the total number of tokens in a batch. By default, a batch uses a sequence length of 512. To set the number of tokens in a batch, you should set

```
--gin_param = "tokens_per_batch=1048576"
```

### Eval

In order to evaluate a model in the T5 framework, you need to use the `eval.gin` file, specify the model directory, decoding method, and which checkpoint step(s) to evaluate. So, to evaluate on the GLUE MRPC task using beam search on *all* checkpoints, use the following command:

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

To evaluate a specific checkpoint, simply set the `eval_checkpoint_step` parameter to appropriate checkpoint.

```
--gin_param="eval_checkpoint_step = 100000"
```

You can also use `greedy_decode.gin` or `sample_decode.gin` instead of `beam_search.gin` in the command above.


### Decode

In order to produce predictions from a model in the T5 framework, you need to specify the model directory, decoding method, and which checkpoint step(s) to use for decoding. Assuming you have a text file of input sequences stored at `/path/to/intputs.txt`, an example command would be:

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

To predict with a specific checkpoint, simply set the `infer_checkpoint_step` parameter to appropriate checkpoint.

```
--gin_param="infer_checkpoint_step = 100000"
```

You can also use `beam_search.gin` or `greedy_decode.gin` instead of `sample_decode.gin` in the command above.

### Export

You may also want to export a [`SavedModel`](https://www.tensorflow.org/guide/saved_model), which is useful for serving your trained model, (e.g., when deploying with [ML Engine](https://cloud.google.com/ml-engine/docs/deploying-models) or in a [Docker](https://docs.docker.com) image).

```sh
t5_mesh_transformer \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --use_model_api \
  --mode="export_predict" \
  --export_dir="/path/to/export/dir"
```

The command above exports the latest checkpoint in the model directory. To export a particular checkpoint, add the following flags:

```sh
  --checkpoint_mode="specific" \
  --checkpoint_steps=1000000
```

The [t5-deploy notebook](https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-deploy.ipynb) demonstrates exporting a `SavedModel` and packaging it in a [Docker](https://docs.docker.com) image for serving.

### GPU Usage

If you would like to use GPU instead of TPUs, you can modify the above commands by removing TPU-specific flags (`--tpu`, `--tpu_zone`, `--gcp_project`) and setting the gin params for `mesh_shape` and `mesh_devices` based on your desired setup.

For example, if your machine has access to 6 GPUs and you'd like to do 3-way model parallelism and 2-way data parallelism, the fine-tuning command above would become:

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

With a single GPU, the command is:

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



### Reproducing our experiments

We provide operative configs for all of the experiments in the [paper][paper] in [gs://t5-data/experiments](https://console.cloud.google.com/storage/browser/t5-data/experiments).
The `experiments` folder has different subdirectories corresponding to the different sections in our paper.
For example, [gs://t5-data/experiments/objectives](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives) contains the experiments from Section 3.3 ("Unsupervised objectives").
Each subdirectory of the `objectives` folder contains operative configs for some particular experiment (where loosely speaking an "experiment" is one of the rows in one of the tables in our paper).

Let's say you want to reproduce the results for the "Prefix language modeling" objective (the first row in Table 4).
The operative configs for that experiment live in [gs://t5-data/experiments/objectives/obj-prefix_lm](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm).
In the base directory, there is an operative config for pre-training the model ([gs://t5-data/experiments/objectives/obj-prefix_lm/operative_config.gin](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm/operative_config.gin)).
Then, there are subdirectories for each of the downstream fine-tuning mixtures we consider, each of which has its own operative config (for example, [gs://t5-data/experiments/objectives/obj-prefix_lm/cnn_dailymail_v002/operative_config.gin](https://console.cloud.google.com/storage/browser/t5-data/experiments/objectives/obj-prefix_lm/cnn_dailymail_v002/operative_config.gin)).
To run this experiment, first pre-train a model with the pre-training operative config:

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

Then, you can fine-tune the pre-trained model on CNN/Daily Mail like so:

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

### Useful Options

Some training variants need multiple flags to be set at the same time. For each
of the below variants, add the group of flags to
`./third_party/py/t5/google/scripts/run_finetune.sh`.

*Deterministic training*

```sh
  --train_gin_param="mesh_train_dataset_fn.seed=${SEED}" \
  --train_gin_param="utils.run.skip_seen_data = True" \
```

*Language model*

```sh
  --objective="lm" \
  --train_gin_param="utils.run.model_type = \"lm\"" \
```

## Released Model Checkpoints

We have released the following checkpoints for pre-trained models described in our [paper][paper]:

* **T5-Small** (60 million parameters): [gs://t5-data/pretrained_models/small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/small/)
* **T5-Base** (220 million parameters): [gs://t5-data/pretrained_models/base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/base/)
* **T5-Large** (770 million parameters): [gs://t5-data/pretrained_models/large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/large/)
* **T5-3B** (3 billion parameters): [gs://t5-data/pretrained_models/3B](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/3B/)
* **T5-11B** (11 billion parameters): [gs://t5-data/pretrained_models/11B](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/11B/)

See [here][released_checkpoints] for a list of additional experimental pre-trained model checkpoints.

# How to Cite
If you extend or use this work, please cite the [paper][paper] where it was introduced:

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
