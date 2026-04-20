# Lexical and Position Bias in Post-hoc Explanations
Good quality explanations strengthen the understanding of language models and data. Feature attribution methods, such as Integrated Gradient, are a type of post-hoc explainer that can provide token-level insights. However, explanations on the same input may vary greatly due to underlying biases of different methods. Users may be aware of this issue and mistrust their utility, while unaware users may trust them inadequately. In this work, we delve beyond the superficial inconsistencies between attribution methods, structuring their biases through a model- and method-agnostic framework of three evaluation metrics. We systematically assess both lexical and position bias (_what_ and _where_ in the input) for two transformers; first, in a controlled, pseudo-random classification task on artificial data; then, in a semi-controlled causal relation detection task on natural data. We find a trade-off between lexical and position biases in our model comparison, with models that score high on one type score low on the other. We also find signs that anomalous explanations are more likely to be biased.

## Code guidelines:
* `prepare_artificial_data.py` : preprocesses synthetic datasets --> toy_data/
* `prepare_causal_data.py` : preprocess the causal relations dataset --> causal_data/exp1/
* `bert.py` : finetune the models (BERT, ModernBERT)
* `explain.py` : creates explanations for the dev/test instances.
* `analyse.py` : analysis script. 
* `run_models.sh` : loops and runs models.
* `compute_sufficiency.py` : main script to compute sufficiency for a model
* `compute_sufficiency.sh` : computes sufficiency in a loopt for different models


## Citation
If you use this work, please cite:

```bibtex
@article{kamp2025explanation,
  title={Explanation Bias is a Product: Revealing the Hidden Lexical and Position Preferences in Post-Hoc Feature Attribution},
  author={Kamp, Jonathan and Bakker, Roos and Blok, Dominique},
  journal={arXiv preprint arXiv:2512.11108},
  year={2026}
}
