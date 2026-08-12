# SeqIO: Task-based datasets, preprocessing, and evaluation for sequence models.

**SeqIO** is a library for processing sequential data to be fed into downstream
sequence models. It uses [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
to create data pipelines but requires minimal use of TensorFlow.

SeqIO is a refactor of the [`t5.data`](https://github.com/google-research/text-to-text-transfer-transformer/)
library used (in conjunction with the [Mesh Tensorflow](https://github.com/tensorflow/mesh)
Transformer implementation) to train the T5 models introduced in
[_Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_](https://arxiv.org/abs/1910.10683).

If you have used `t5.data` in the past and want to know how SeqIO differs, please read [this section](#differences-from-t5data).

<!--Uncomment once forked.
## Installation

### From Pypi

```sh
pip install seqio
```

### From Source

```sh
git clone https://github.com/google/seqio.git
cd seqio
pip install -e .
-->

## Usage Tutorial

### Defining a `Task`

The most important class in SeqIO is the `Task`. It is an abstraction that combines:

  * a raw *data source*
  * one or more *preprocessing* steps
  * a *vocabulary* to tokenize/detokenize each preprocessed feature for the model
  * a *postprocessor* to convert detokenized model outputs into a format for evaluation
  * one or more *metrics* to evaluate with

Oftentimes a `Task` lines up with a common benchmark. In this tutorial we will
create a task definition for the closed-book, open-domain version of [TriviaQA](https://nlp.cs.washington.edu/triviaqa/),
defining various parts as we go. In the end, our `Task` will look like this:

```py
seqio.TaskRegistry.add(
    "trivia_qa_open",
    source=seqio.TfdsDataSource(
      tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
      splits={
          "train": "train[:90%]",
          "validation": "train[90%:]",
          "test": "validation"
      }),
    preprocessors=[
        tqa_open_preprocessor,
        seqio.tokenize,
        seqio.append_eos,
    ],
    output_features={
        "inputs": seqio.Feature(
           seqio.SentencePieceVocabulary("/path/to/inputs/vocab"),
           add_eos=False, dtype=tf.int32
        ),
        "targets": seqio.Feature(
           seqio.SentencePieceVocabulary("/path/to/targets/vocab"),
           add_eos=True, dtype=tf.int32
        ),
    },
    postprocess_fn=tqa_open_postprocessor,
    metric_fns=[tqa_metric])
```

We typically add the `Task` to the global registry when we define it (as shown
above) to make it easier to use with model configs and flags. Thus, it  must
have a unique string name (`"trivia_qa_open"` in this case). Note, however, that
you may also instantiate a `seqio.Task` directly without adding it to the
registry, if desired.

We'll now break down each part of the task definition.

#### Data Source

Data sources are the first step in your pipeline, providing a way to load raw
data in many format as a `tf.data.Dataset`.
All data sources are subclasses of the `DataSource` base class and are defined in
[dataset_providers](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/dataset_providers.py),

Existing implementations include:

  * `TfdsDataSource` for loading examples from [TensorFlow Datasets](https://www.tensorflow.org/datasets).
  * `TextLineDataset` for loading examples from text files (e.g., tsv).
  * `TFExampleDataSource` for loading [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord) protos from a file (e.g. a `TFRecord` file.)
  * `FunctionDataSource` for providing an custom function that returns a `tf.data.Dataset`.

In our example, we are using the `TfdsDataSource`. We specify the name of the TriviaQA dataset in TFDS ([`"trivia_qa"`](https://www.tensorflow.org/datasets/catalog/trivia_qa)), the specific config that excludes the context for the open domain setting (`"unfiltered.nocontext"`), and the version number (`"1.1.0"`). We also override the default splits to match what is commonly used for the open domain setting. Specifically, we set our "test" split to be the TFDS "validation" split, and create a small pseudo-"validation" set by taking examples out of the TFDS "train" split.

#### Output Features

The `output_features` field expects a dictionary that maps string feature names
to `seqio.Feature` objects. This defines what the `Task` is expected to produce
in its output examples. The output examples *may* contain additional fields, but
they *must* contain these fields in the specified format or exceptions will be
raised.


Each `Feature` includes:

  * A `vocabulary`, which must subclass [`seqio.Vocabulary`](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/vocabularies.py), to specify how the feature can be tokenized and detokenized. You may use `seqio.PassThroughVocabulary` if tokenization is not necessary.
  * `add_eos`, which specifies whether the feature should end with the vocabulary's EOS token.
  * The output `dtype` which must be a `tf.dtypes.DType`.

**Note:** specifying these options on `Feature` does not by itself ensure the proper transformations are applied -- you must also include then necessary preprocessors.

The [tasks used in T5](TODO) all produce "inputs" and "targets" features to be consumed by the text-to-text model. For a decoder-only language model, only a single feature (e.g., "targets") would be necessary.
Nevertheless, SeqIO is flexible enough to generate arbitrary output features what will be converted into model features by the [`FeatureConverter`](#featureconverter) later in the pipeline.

#### Preprocessors

Preprocessors are functions that transform one `tf.data.Dataset` into a new `tf.data.Dataset`. Typically this involves executing a `map` over the given dataset. The preprocessors provided to the `Task` will be executed sequentially.

As an example, let's look at the previously undefined `tqa_open_preprocessor` from the "trivia_qa_open" example above.

```py
def trivia_qa_open(
    dataset: tf.data.Dataset,
    prefix:str = "trivia_qa question: "
  ) -> tf.data.Dataset:
  """Convert TriviaQA dataset to open domain qa examples.

  The function takes the trivia_qa TFDS dataset and emits examples of the
  form:
  {
    "inputs": "trivia_qa question: What are the names of the Olsen Twins?"
    "targets": "Mary-Kate and Ashley",
    "answers": ["Mary-Kate and Ashley", "Ashley and Mary-Kate"]
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.

  Returns:
    a tf.data.Dataset
  """
  def tqa_map(ex):
    """Map TriviaQA example to text-to-text example."""
    return {
        "inputs": prefix + ex["question"],
        "targets": ex["answer"]["value"],
        "answers": ex["answer"]["aliases"],
    }

  return dataset.map(tqa_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

A few **important** notes:

  1. The only arguments each preprocessor function may have are `dataset`, `output_features`, and `sequence_length`. The first argument must be provided, although it doesn't have to be named `dataset`. If an argument named `output_features` is provided, the [output feature mapping](#output-features) will be passed to the preprocessor. If `sequence_length` is provided, a mapping from feature name to its *maximum* final sequence length ([provided by the caller](#getting-a-preprocessed-dataset) will be passed -- any sequences that are too long after preprocessing will be automatically truncated. In the above example, `prefix` must be partialed out if it needs to be changed from the default.

  1. Mapping functions operate on and return `tf.Tensor`s using TensorFlow operations, although it is possible to take advantage of automatic [AutoGraph](https://blog.tensorflow.org/2018/07/autograph-converts-python-into-tensorflow-graphs.html) conversion for `numpy` or use [`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) to wrap arbitrary Python code. See `tf.data.Dataset` [documentation](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for more details.

  1. When calling `map`, it is important to **always** set `num_parallel_calls=tf.data.experimental.AUTOTUNE` to avoid creating a bottleneck. The `seqio.map_over_dataset` decorator helps enforce this as follows:

  ```py
  def trivia_qa_open(
    dataset: tf.data.Dataset,
    prefix: str = "trivia_qa question: "
  ) -> tf.data.Dataset:

    @seqio.map_over_dataset
    def tqa_map(ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      """Map TriviaQA example to text-to-text example."""
      return {
          "inputs": prefix + ex["question"],
          "targets": ex["answer"]["value"],
          "answers": ex["answer"]["aliases"],
      }

  return tqa_map(dataset)
  ```

  1. Stochastic operations must be [stateless](https://www.tensorflow.org/guide/random_numbers#stateless_rngs) if deterministic pipelines are needed. To get (optionally deterministic) seeds for these operations, use the `seqio.map_over_dataset(num_seeds=n)` decorator. For example:

  ```py
  def random_chunk(
    dataset: tf.data.Dataset,
    sequence_length: Mapping[str, int]
  ) -> tf.data.Dataset:
  """Takes a random chunk out of each feature the size of `sequence_length`."""

    @seqio.map_over_dataset(num_seeds=1)
    def take_chunk(
        ex: Mapping[str, tf.Tensor],
        seed
    ) -> Mapping[str, tf.Tensor]:
      new_ex = {}
      for k, v in ex.items():
        if k in sequence_length:
          length = sequence_length[k]
          start_idx = tf.random.stateless_uniform(
             (), seed, 0, tf.size(v) - (length + 1))
          new_ex[k] = v[start_idx:start_idx+length]
        else:
          new_ex[k] = v
      return new_ex

  return take_chunk(dataset)
  ```

  If `num_seeds > 1`, the arg will instead be called `seeds` and will contain a sequence of seeds.

In our "trivia_qa_open" task, we also use the predefined preprocessors `seqio.tokenize` and `seqio.append_eos`. The former uses each `Feature.vocabulary` to tokenize it, and the the latter appends `Feature.vocabulary.eos_id` to the feature if the `Feaure.add_eos` is True. See [preprocessors.py](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/preprocessors.py) for their implementations and other useful preprocessors.

#### Postprocessor

During evaluation, the model outputs are first detokenized using the output feature vocabulary. Before passing these predictions to the metric functions, they can be run through a Python postprocess function, joined with the full input example. Similarly, the raw targets are run through this function before being passed to the metrics.
Since the postprocess function is used on both the model output and the targets, it is passed an `is_target` boolean in case the behavior should be different. It is also passed the fully preprocessed example, including fields that were excluded from `output_features`.

As an example, lets look at the previously undefined `tqa_open_postprocessor`.

```py
def tqa_open_postprocessor(output_or_target, example=None, is_target=False):
  """Returns output as answer, or all answers if the full example is provided."""
  if is_target:
    return [a.decode("utf-8") for a in example["answers"]]
  else:
    return output_or_target.decode("utf-8")
```

When processing the target, we ignore `output_or_target` (equivalent to `example["targets"]`) since it is just selecting a single answer in `trivia_qa_open`. Instead, we extract the full list of answers from the example and convert them from bytes to text. When handling the model output, we simply convert it to text from detokenized bytes.

#### Metrics

Metrics are functions that are passed (by the [Evaluator](#evaluator)) the fully-materialized list of postprocessed model outputs (or scores) and targets and return a mapping from string names to `Metric` objects containing their values. These are most commonly floating-point scalars, but may also be text, images, audio, histograms, etc (see [evaluation.py](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/evaluation.py) for the full list).

The first argument of a metric function must always be called `targets`. If the second argument of a metric function is called `predictions`, it will be passed the decoded and detokenized model prediction. If it is called `scores`, it will be passed a list of log-likelihood scores for each example.

If multiple metric functions are provided, they will all be used and their returned mappings merged.

##### Prediction Metrics

Prediction metrics are computed using the postprocessed targets and model outputs (predictions).
The args must be named `targets` and `predictions`.

Let's look at the previously undefined `tqa_metric` prediction metric:

```
def tqa_metric(
  targets: Sequence[Sequence[str]],
  predictions: Sequence[str]
) -> Mapping[str, seqio.Metric]:
  """Computes official TriviaQA metrics.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """

  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")

  def _normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
      return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
      to_replace = set(string.punctuation)
      return "".join("" if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
      return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text

  # Normalize answers before comparing.
  targets = [[_normalize_answer(t) for t in u] for u in targets]
  predictions = [_normalize_answer(p) for p in predictions]

  def _f1_score(target, prediction):
    """Computes token f1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
      return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

  em = np.mean([
      max(pred == gt for gt in ground_truths)
      for pred, ground_truths in zip(predictions, targets)
  ])
  f1 = np.mean([
      max(_f1_score(gt, pred) for gt in ground_truths)
      for pred, ground_truths in zip(predictions, targets)
  ])
  em *= 100
  f1 *= 100
  return {
      "em": seqio.evaluation.Scalar(em),
      "f1": seqio.evaluation.Scalar(f1),
  }
```

##### Score Metrics

Score metrics are computed using the postprocessed targets and their log-likelihood scores according to the model.
The args must be named `targets` and `scores`.

```py
def perplexity(targets: Sequence[str], scores: Sequence[int]):
  return {
    "perplexity": seqio.evaluation.Scalar(np.exp(np.mean(scores)))
  }
```

### Defining a `Mixture`

Once you have multiple `Task`s added to the `TaskRegistry`, you can define `Mixture`s that will combine the examples from them according to some specified rate.
Examples will then be sampled from each task in proportion to its rate.

As an example, [Multilingual T5](goo.gle/mt5) uses a `Mixture` of per-language
`Task`s with tail languages up-weighted in the mixture.

There are 3 ways to specify the tasks and their rates:

  1. Provide a rate along with each task's name (rates are normalized before sampling):

  ```py
  seqio.MixtureRegistry.add(
    "mix1",
    [("task1", 1), ("task2", 7)]
  )
  ```

  1. Provide a constant default rate for some or all tasks, which will be used when only the name is provided.
  The example below will produce identical mixing rates as the previous one.

  ```py
  seqio.MixtureRegistry.add(
    "mix1",
    [("task1", 0.5), "task2"],
    default_rate=3.5
  )
  ```

  1. Provide a function that generates the rate for each task at runtime.
  The example below uses the provided [`seqio.mixing_rate_num_examples`](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/utils.py), which uses the number of examples (computed during [offline caching](#optional-offline-caching)) as the rate for each task.

  ```py
  seqio.MixtureRegistry.add(
    "mix2",
    ["task1", "task2"],
    default_rate=seqio.mixing_rate_num_examples
  )
  ```

You can also include `Mixture`s in your `Mixture`! For example, the following task would contain 1/24 (from "mix1") + 1/3 "task1", 7/24 (from "mix1") of "task2", and 1/3 "task3".

```py
seqio.MixtureRegistry.add(
  "mix3",
  ["mix1", task1", "task3"],
  default_rate=1
)
```

### Getting a Preprocessed Dataset

Now that your `Task` (and/or `Mixture`) is defined, its primary functionality is to use it to generate a dataset.

You may first need to use `seqio.get_mixture_or_task(mixture_or_task_name)` to access your dataset provider from the registry.

After that, you can call `get_dataset` to build the `tf.data.Dataset`. For example:

```py
dataset = seqio.get_mixture_or_task("mix1").get_dataset(
    sequence_length={"inputs": 256, targets": 128},
    dataset_split="train",
    shuffle=True,
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42
)

# Print the first 5 examples.
for _, ex in zip(range(5), dataset.as_numpy_iterator()):
  print(ex)
```

Some notes on a few the arguments:

  * `sequence_length`: An *optional* mapping from feature name to *maximum* length. Will be passed to the preprocessors with a `sequence_length` argument. If not `None`, the final example features will be truncated if they exceed the specified length. Note that this value may be required to be set if any of the preprocessors use the `sequence_length` argument and do not handle the `None` case.
  * `num_epochs`: The number of times to repeat the source dataset. Preprocessing will be re-applied with new seeds to enable new samples from stochastic steps. Note that if the `CacheDatasetPlaceholder` is included (see below) preprocessing is only re-applied after that step.
  * `shard_info`: An optional sharding specification for loading a deterministic subset of the dataset. Loading will be most efficient if the number of shards evenly divides the number of shards in the raw data source.
  * `use_cached`: Specifies whether to load from a pre-cached task for increased performance or to do the preprocessing on-the-fly. See the [following section](#optional-offline-caching) for details on how to cache your task, which must be done before this can be set to `True`.
  * `seed`: An optional seed to use for deterministic shuffling and (stateless) stochastic ops. These operations will still be pseudorandom but will be reproducible with the same seed. Set to `None` if determinism is not desired.

### (Optional) Offline Caching

For improved performance at load time and avoid redundant computations for commonly used tasks, you can pre-cache your `Task` with all or part of the preprocessing done in advance of training.

The first step to doing so is to add a `seqio.CacheDatasetPlaceholder(required=False)` as one of the steps in your preprocessing pipeline. All steps before the placeholder will be cached offline and all steps after will be executed on the fly at load time. You may set `required=True` if you want `get_dataset` to fail unless `use_cached=True`.

Caveats:

* Any stochastic operations that you wish to be re-run when `num_epochs > 1` or with a different `seed` *should* go after the placeholder since only a single sample will be cached.
* Any preprocessing steps that use the `sequence_length` argument *must* come after the `seqio.CacheDatasetPlaceholder` preproessor since this is only known at runtime, or an exception will be raised. If you wish to cache for a specific sequence length, you can use [`seqio.experimental.add_fully_cached_task`](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/seqio/experimental.py).

Once your `Task` is registered, you can run [`cache_tasks_main`](scripts/cache_tasks_main.py) to execute the offline preprocessing, providing it with the module containing your task definitions via the `--module_import` flag. For very large datasets, it's recommended you run this [Apache Beam](https://beam.apache.org/) script on a distributed framework like [Google Cloud DataFlow](https://beam.apache.org/documentation/runners/dataflow/).

Finally, you are ready to load the cached version of your `Task` (or `Mixture`) containing it. You will need to add the path to the directory you passed to `--output_cache_dir` via `seqio.add_global_cache_dirs(["/my/cache/dir"])`. Now when you call `task_or_mixture.get_dataset(..., use_cached=True)`, the data will be loaded from the cache directory instead of the raw data source.

### Feature Converters

The role of `Task` is to provide the dataset object with as little
model-specific features (e.g., generic "inputs" and "targets") while the Feature
Converters transform the model-agnostic features to model-specific features
(e.g., "encoder_input_tokens"). We refer to the former as "task features" and
the latter as "model features".


Let's use machine translation (English to German) as a running example.

The raw data consists of sentence pairs such as

```
"That is good\tDas ist gut."
```

A task registered to `Task` (e.g.,
[wmt_t2t_ende_v003](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/data/tasks.py?l=156&rcl=337594707))
reads these sentence pairs from the data source and applies a series of
[preprocessors](https://github.com/google-research/text-to-text-transfer-transformer/tree/master/t5/data/preprocessors.py?rcl=343354647).
One of the internal representations looks like

```python
{"inputs": "translate English to German: That is good.",
 "targets": "Das ist gut."}
```

The final output from the `Task` is a tokenized version of the parallel
sentences.  In the following toy example (the token ids do not correspond to the
above string example), the dataset consists of 2 examples.

```python
dataset = [{"inputs": [7, 8, 5], "targets": [3, 9]},
           {"inputs": [8, 4, 9, 3], "targets": [4]}]
```

The format is in the `tf.data.Dataset` (i.e., each example is a dictionary with
"inputs" and "targets" fields.

The `FeatureConverter` then takes this as an input and converts to the
model-specific features. For example, let's assume that we are using the
standard Transformer architecture with an encoder and a decoder. In addition,
the feature converter performs packing and padding for efficient use with TPUs.
The output of the feature converter is

```python
converted_dataset = [{
    "encoder_input_token": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
     "encoder_segment_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
       "encoder_position": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
   "decoder_target_token": [3, 9, 1, 4, 1, 0, 0],
    "decoder_input_token": [0, 3, 9, 0, 4, 0, 0],
    "decoder_loss_weight": [1, 1, 1, 1, 1, 0, 0],
       "decoder_position": [0, 1, 2, 0, 1, 0, 0],
     "decoder_segment_id": [1, 1, 1, 2, 2, 0, 0],
}]
```

In this particular example, two examples are packed into one. `*_segment_id` and
`*_position` are the fields used to denote the membership and position of packed
token in the original sequence. The EOS ids (i.e., 1) are appended. In addition,
each fields is padded to the specified length.


#### Feature converters provided out of the box

We provide feature converters for three common architectures: encoder-decoder,
decoder-only and encoder-only. Here we describe how users can use the feature
converters for each of these architectures out of the box as a part of the
`SeqIO` library.

In the `SeqIO` library, each architecture has a class defining how the features
are converted. Since these feature converters are already implemented, it is
straightforward to use them. Providing the class as a `feature_converter_cls`
argument of the `seqio.get_dataset` function is all you need.


##### Encoder-decoder architecture: `seqio.EncDecFeatureConverter`
This is the architecture of the original Transformer paper. For the
English-to-German translation task, the following function call retrieves the
`tf.data.Dataset` object with the

```python
dataset: tf.data.Dataset = seqio.get_dataset(
    mixture_or_task_name="wmt_t2t_ende_v003",
    task_feature_lengths={"inputs": 32, "targets": 32},
    dataset_split="train",
    pack=True,
    shuffle=True,
    feature_converter=seqio.EncDecFeatureConverter(pack=True)
)
```

The resulting dataset object has the following 7 fields

|Feature name          | Explanation                |
|----------------------|---------------------------|
|`encoder_input_token` | Input tokens to the encoder |
|`encoder_position`    | Position index in the sequence before packing|
|`encoder_segment_id`  | Sequence membership before packing. Two positions with the ` same positive integer mean that they belong to the same sequence before packing. |
|`decoder_input_token` | Input tokens to the decoder |
|`decoder_target_token`| Output tokens from the decoder |
|`decoder_loss_weight` | Binary mask to indicate where the loss should be taken |
|`decoder_position`    | Position index in the sequence before packing|
|`decoder_segment_id`  | Same as `encoder_segment_id` but for decoder.|

The last column "Pack" means that the feature is only present for the packed
dataset. For example, if `pack=False` in the `seqio.get_dataset` function call,
`encoder_segment_id` will not be included in the returned dataset because it is
generated during packing operation.


##### Decoder-only architecture

This architecture consists of a single autoregressive stack, which we denote as
a "decoder".

A decoder autoregressively produces an output sequence.
Therefore, it can be used as a standard language model if an input data has
"targets" field only, i.e., unsupervised. If the input data also has "inputs"
field, e.g., machine translation, the decoder can still be used by concatenating
the inputs and targets fields. See [Raffel et al.
(2020)](https://arxiv.org/abs/1910.10683), Section 3.2.1 for more detailed take
on this topic.

We support both uses cases and refer to the former as Standard Language Model
and the latter as Prefix Language Model. Each of these models is described
separately.

Note that we do not provide special tensors to denote how the dataset should be
consumed. For example, Transformer-based fully autoregressive decoder has a
fully-causal self-attention layer. Since there are many ways of implementing the
masking pattern for such attention layer and, more importantly, `SeqIO` is not
limited to attention-based models, we leave it up to the model implementations
to implement the masking pattern. There is one exception, and we cover this in
the Prefix LM section.

A common use pattern is to pre-train a decoder model with the left-to-right
language modeling objective (unsupervised) using `seqio.LMFeatureConverter` and
then fine-tune (supervised) using `seqio.PrefixLMFeatureConverter`.


###### Standard LM

For the standard language model, the input dataset only has "targets" field.
Therefore, the sequence length specification only needs to specify targets.

```python
dataset: tf.data.Dataset = seqio.get_dataset(
    mixture_or_task_name="standard_lm",
    task_feature_lengths={"targets": 32},
    dataset_split="train",
    pack=True,
    shuffle=True,
    feature_converter=seqio.LMFeatureConverter(pack=True)
)
```

Note that "standard_lm" is not a registered task in the T5 codebase.

The output dataset has the following model features.

|Feature name          | Explanation                |
|----------------------|---------------------------|
|`decoder_target_token`| Output tokens from the decoder |
|`decoder_input_token` | Input tokens to the decoder |
|`decoder_loss_weight` | Binary mask to indicate where the loss should be taken |
|`decoder_position`    | Position index in the sequence before packing|
|`decoder_segment_id`  | Sequence membership before packing. Two positions with the ` same positive integer mean that they belong to the same sequence before packing. |



###### Prefix LM: `seqio.PrefixLMFeatureConverter`

If the input dataset has a notion of "inputs" and "targets", we can concatenate
them so that we can still use a single stack decoder. Therefore, the output only
contains "targets" just like standard LM case.

We use the same toy example for English-to-German translation task as a running
example:

```
{"inputs": "translate English to German: That is good.",
 "targets": "Das ist gut."}
```

To be consumed by the decoder-only stack, `seqio.PrefixLMFeatureConverter`
concatenates them form the new "targets". Consider 2-layer decoder architecture
whose activations are shown below

```

That  is  good <EOS> Das ist gut <EOS>
 |    |    |    |    |   |    |   |
 u1   u2   u3   u4   u5  u6   u7  u8
 |    |    |    |    |   |    |   |
 v1   v2   v3   v4   v5  v6   v7  v8
 |    |    |    |    |   |    |   |
<BOS> That is  good <EOS> Das ist gut

```

Let's us denote the first layer's activation in the `i`th position as `vi`.
Similarly, let `ui` denote the activation of the second layer in the `i`th
position.


For attention-based sequence models such as Transformer decoders, the self
attention layer is used to encode contextualized representation of the sequence.
At a given layer, each position's representation is computed as a function of
the representations of the tokens *before* its position in the previous layer.

Referring to the toy example, when computing `u2` with fully-causing masking, we
do not use `v3`. This results in a representation `u2` of the word "is" that
does not take into account the word "good", which is unnecessarily limiting.

For Prefix LM, this issue is resolved by having the fully visible masking
pattern for the inputs portion only. For example, when computing `u2`, `v1`,
`v2`, `v3`, `v4` and `v5` are all visible and taken into account. For the tokens in
the "targets" of the `Task` dataset, we use the causal masking. For example,
when computing `u6`, all `vi` for `i <= 6` are taken into account but not `v7`.

<details>
  <summary>Why `v5` is included in the inputs attention pattern</summary>
  In the same translation example, we note that when computing `u2`, the
  activation corresponding to the position where \<EOS\> token was input (i.e.,
  `v5`) was visible. This doesn't count as "cheating" because the model doesn't
  see the next word "Das". This can provide additional context in building the
  representation for "good". In this case, `u4` has the context that "good" is
  the last word in the sentence.
</details>

`seqio.PrefixLMFeatureConverter` provides a feature `decoder_causal_attention`
to encode this information. For the above example, we have


```
decoder_causal_attention = [1, 1, 1, 1, 1, 0, 0, 0]
```

indicating that the non-causal attention can be applied to the first five
positions. Note that this feature seems trivial but for a packed dataset,
the inputs and targets boundary are more nuanced.


A final consideration for the prefix LM is that because we concatenate "inputs"
and "targets", which tokens are used for the loss computation is a modeling
decision. For example, we can penalize the models only for the "targets" tokens
or we may choose to penalize building the representation for "inputs" tokens.
This is controlled by `loss_on_targets_only` argument (defaults to True) to
`seqio.PrefixLMFeatureConverter` constructor. In the above example, we would get

```
decoder_loss_weights = [0, 0, 0, 0, 1, 1, 1, 1]
```

This indicates that the last 4 positions are used for the loss computation.

To get the dataset with prefix LM features, we can use

```python
dataset: tf.data.Dataset = seqio.get_dataset(
    mixture_or_task_name="wmt_t2t_ende_v003",
    task_feature_lengths={"inputs": 32, "targets": 32},
    dataset_split="train",
    pack=True,
    shuffle=True,
    feature_converter=seqio.PrefixLMFeatureConverter(
        pack=True,
        loss_on_targets_only=True)
)
```

The resulting features have length 64 because it concatenates inputs and targets
each with length 32.

The output dataset has the following model features. Note that the only
additional feature is `decoder_causal_attention`.

|Feature name          | Explanation                |
|----------------------|---------------------------|
|`decoder_target_token`| Output tokens from the decoder |
|`decoder_input_token` | Input tokens to the decoder |
|`decoder_loss_weight` | Binary mask to indicate where the loss should be taken |
|`decoder_position`    | Position index in the sequence before packing|
|`decoder_segment_id`  | Sequence membership before packing. Two positions with the ` same positive integer mean that they belong to the same sequence before packing. |
|`decoder_causal_attention`| Binary mask denoting which tokens are in the non-causal masking region.|




###### Encoder-only architecture
Like decoder-only architecture, this one is a single stack, but not
autoregressive.

One notable assumption is that the inputs and targets are *aligned*, i.e., they
have the same sequence length and `i`th position in the targets correspond to
the output representation of the `i`th token in the inputs.

A common model using encoder-only architecture is
[BERT](https://arxiv.org/abs/1810.04805). We provide `Encoder` feature converter
class to support the Masked Language Modeling (MLM) objective from BERT.

We assume that a unique sentinel such as `[MASK]` token is used to mask some
fraction of the input text and the task is to recover the original text.
Therefore, the "targets" is naturally defined as the original text whereas
"inputs" are the masked text.

Encoder-only models are often used for classification tasks. In BERT, a special
token `[CLS]` is prepended to the input sequence. The last layer's activation
corresponding to this sentinel token is the contextualized representation of the
sequence. We assume that such "classification" sentinel is prepended.

Consider the following example for the MLM task. The input dataset has two
examples, which is packed to one example. We assume that `mask_id = 9` and the
`[CLS]` token has id of 8.

```
dataset = [{"inputs": [8, 9, 9, 3, 4], "targets": [8, 7, 4, 3, 4]},
           {"inputs": [8, 3, 9], "targets": [8, 3, 6]}]

converted_dataset = {
     "encoder_input_token": [8, 9, 9, 3, 4, 1, 8, 3, 9, 1, 0],
    "encoder_target_token": [8, 7, 4, 3, 4, 1, 8, 3, 6, 1, 0],
      "encoder_segment_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0],
        "encoder_position": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0],
     "encoder_loss_weight": [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
}
```

Note that the packed sequence has `[CLS]` token at the beginning of each
sequences. Also note that the loss is taken only on the masked position.

To use the pre-defined `EncoderFeatureConverter`, provide `mask_id` as an
argument.

```python
dataset: tf.data.Dataset = seqio.get_dataset(
    mixture_or_task_name="some mlm task",
    task_feature_lengths={"inputs": 32, "targets": 32},
    dataset_split="train",
    pack=True,
    shuffle=True,
    feature_converter=seqio.EncoderFeatureConverter(
        pack=True,
        mask_id=9)
)
```

The resulting dataset object has the following 5 fields

|Feature name          | Explanation                |
|----------------------|---------------------------|
|`encoder_input_token` | Input tokens to the encoder |
|`encoder_position`    | Position index in the sequence before packing|
|`encoder_segment_id`  | Sequence membership before packing. Two positions with the ` same positive integer mean that they belong to the same sequence before packing. |
|`encoder_target_token`| Output tokens from the encoder |
|`encoder_loss_weight` | Binary mask to indicate where the loss should be taken |



### `Evaluator`

TODO(hwchung)

## Differences from `t5.data`

The original `t5` library introduced and implemented the `t5.data.Task` abstraction for specifying preprocessing and evaluation metrics for text-to-text tasks. When creating a task, users specify a source dataset of raw text, some preprocessing steps, a vocabulary for tokenization, and evaluation metrics. The fully-specified Task can then be used to pre-train or fine-tune a encoder-decoder transformer model. However, the design included many baked-in assumptions about the types of tasks users could specify.

SeqIO removes some of the constraints of this abstraction:

* Inputs and outputs are no longer required to be strings (e.g., it may be images or audio).
* Architectures other than the original encoder-decoder are supported (e.g., decoder-only languaged models like GPT or encoder-only models like BERT).
* Users can control at which stage of the pipeline offline caching occurs.
* Users can control when and where EOS tokens are added.

Furthermore, SeqIO has been made more modular with respect to the Mesh TensorFlow Transformer. This allows it to be used with other model implementations with more consistency and much less code duplication.
