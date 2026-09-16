**Constraining-Data-Distribution-For-Language-Modeling**

It has been observed that LLMs store around 2 bits of factual information per parameter, given sufficient repetition in the training data (see [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws by Zeyuan Allen-Zhu and Yuanzhi Li](https://physics.allen-zhu.com/part-3-knowledge/part-3-3)). A 7B param model would store ~<2GB of factual information (14B bits -> 1.75B bytes).

A large scale rewrite of training data (or just UNK token replacements for words not in a whitelist) may be used to remove encyclopedia style factual knowledge from a pretrain dataset. Then, models pretrained would have significant more capacity available (for reasoning, in-context learning, etc.) that previously was used for factual knowledge storage. Factual information could be brought into the context anyways with RAG. It could be a new scaling avenue. At frontier sizes, trillions of parameter models trained on as constrained as possible datasets could get an OOM improvement in overparameterization against the data **today** without need for further hardware.

The repository has the code for word level tokenization and UNK token replacements and a manually curated whitelist.

A public email you can reach out to me through: akemeklis [at] protonmail [dot] com
