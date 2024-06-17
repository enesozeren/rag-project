---
language: en
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
pipeline_tag: sentence-similarity
---

# sentence-transformers/gtr-t5-xl

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space. The model was specifically trained for the task of sematic search.

This model was converted from the Tensorflow model [gtr-xl-1](https://tfhub.dev/google/gtr/gtr-xl/1) to PyTorch. When using this model, have a look at the publication: [Large Dual Encoders Are Generalizable Retrievers](https://arxiv.org/abs/2112.07899). The tfhub model and this PyTorch model can produce slightly different embeddings, however, when run on the same benchmarks, they produce identical results.

The model uses only the encoder from a T5-3B model. The weights are stored in FP16.  


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/gtr-t5-xl')
embeddings = model.encode(sentences)
print(embeddings)
```

The model requires sentence-transformers version 2.2.0 or newer.

## Evaluation Results

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/gtr-t5-xl)



## Citing & Authors

If you find this model helpful, please cite the respective publication:
[Large Dual Encoders Are Generalizable Retrievers](https://arxiv.org/abs/2112.07899)
