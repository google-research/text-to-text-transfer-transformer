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

  1. When instantiating a `Task`, the preprocessor functions can have the following arguments: `dataset`, `output_features`, and `sequence_length`. The first (positional) dataset argument is always required. If an argument named `output_features` is provided, the [output feature mapping](#output-features) will be passed to the preprocessor. If `sequence_length` is provided, a mapping from feature name to its *maximum* final sequence length ([provided by the caller](#getting-a-preprocessed-dataset) will be passed -- any sequences that are too long after preprocessing will be automatically truncated. If a preprocessor function does have other arguments, they must have default values or be bound (e.g., with `functools.partial`) before instantiating the `Task`.

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

During evaluation, the model outputs are first detokenized using the output feature vocabulary. Before passing these predictions to the metric functions, they can be run through a Python postprocessing function, alongside the full input example. Similarly, the raw targets are run through this function before being passed to the metrics.
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
    # Remove articles.
    text = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation.
    for punc in string.punctuation:
      text = text.replace(punc, '')
    # Normalize white space
    text = " ".join(s.split())
    return text

  # Normalize answers before comparing.
  targets = [[_normalize_answer(t) for t in u] for u in targets]
  predictions = [_normalize_answer(p) for p in predictions]

  em = np.mean([
      max(pred == gt for gt in ground_truths)
      for pred, ground_truths in zip(predictions, targets)
  ])
  return {
      "exact_match": seqio.evaluation.Scalar(em),
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

### `FeatureConverter`

Given the `Task`'s preprocessed output features, the `FeatureConverter` determines how they will be seen by the model...

TODO(hwchung)

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
