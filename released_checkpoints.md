# Experimental T5 Pre-Trained Model Checkpoints

Below are some pointers to checkpoints for experimental models we have trained after writing our [paper][paper].
We have found that these models can produce better performance in some cases.
These checkpoints are not officially supported - use at your own risk!

### t5.1.1.*

Similar to the models described in our [paper][paper], with the following improvements:

*  GEGLU activation in feed-forward hidden layer, rather than ReLU - see https://arxiv.org/abs/2002.05202 .

* Dropout was turned off in pre-training (quality win).  Dropout should be re-enabled during fine-tuning.

* Pre-trained on C4 only without mixing in the downstream tasks.

* no parameter sharing between embedding and classifier layer

* "xl" and "xxl" replace "3B" and "11B".  The model shapes are a bit different - larger d_model and smaller num_heads and d_ff.

The checkpoints are located here:

* **t5.1.1.small** (~77 million parameters): [gs://t5-data/pretrained_models/t5.1.1.small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.small/)
* **t5.1.1.base** (~250 million parameters): [gs://t5-data/pretrained_models/t5.1.1.base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.base/)
* **t5.1.1.large** (~800 million parameters): [gs://t5-data/pretrained_models/t5.1.1.large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.large/)
* **t5.1.1.xl** (~3 billion parameters): [gs://t5-data/pretrained_models/t5.1.1.xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.xl/)
* **t5.1.1.xxl** (~11 billion parameters): [gs://t5-data/pretrained_models/t5.1.1.xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.xxl/)

### LM-Adapted: t5.1.1.lm100k

These "LM adapted" models are initialized from t5.1.1 (above) and train for an
additional 100K steps on the LM objective discussed in the [T5 paper][paper].
This adaptation improves the ability of the model to be used for [prompt
tuning](https://arxiv.org/abs/2104.08691).

* **t5.1.1.lm100k.small** (~77 million parameters): [gs://t5-data/pretrained_models/t5.1.1.lm100k.small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.small/)
* **t5.1.1.lm100k.base** (~250 million parameters): [gs://t5-data/pretrained_models/t5.1.1.lm100k.base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.base/)
* **t5.1.1.lm100k.large** (~800 million parameters): [gs://t5-data/pretrained_models/t5.1.1.lm100k.large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.large/)
* **t5.1.1.lm100k.xl** (~3 billion parameters): [gs://t5-data/pretrained_models/t5.1.1.lm100k.xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.xl/)
* **t5.1.1.lm100k.xxl** (~11 billion parameters): [gs://t5-data/pretrained_models/t5.1.1.lm100k.xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.xxl/)

### Talking Heads: t5.1.th.*

Variation on the t5.1.1 models using talking-heads attention (https://arxiv.org/abs/2003.02436).

* **t5.1.th.base** (~250 million parameters): [gs://t5-data/pretrained_models/t5.1.th.base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.th.base/)
* **t5.1.th.large** (~800 million parameters): [gs://t5-data/pretrained_models/t5.1.th.large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.th.large/)

[paper]: https://arxiv.org/abs/1910.10683

### First Layers Narrow: t5.1.n4w10.*

Variation on the t5.1.1 models.  Each of the encoder and decoder consists of 14
layer groups, with the last ten twice as "wide" as the first four.  (double d_ff
and num_heads). Parameter count and computation are kept similar to the
corresponding t5.1.1 models.  For the base model, this increases the number of
layers, resulting in better quality, and for the large and xl models, this
decreases the number of layers from 24 to 14, decreasing quality, but also
decreasing the amount of communication necessary for model parallelism.

* **t5.1.n4w10.base** (~250 million parameters): [gs://t5-data/pretrained_models/t5.1.n4w10.base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.n4w10.base/)
* **t5.1.n4w10.large** (~800 million parameters): [gs://t5-data/pretrained_models/t5.1.n4w10.large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.n4w10.large/)
* **t5.1.n4w10.xl** (~3 billion parameters): [gs://t5-data/pretrained_models/t5.1.n4w10.xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.n4w10.xl/)



[paper]: https://arxiv.org/abs/1910.10683
