# Lexical and Position Bias in Post-hoc Explanations
Good quality explanations strengthen the understanding of language models and data. Feature attribution methods, such as Integrated Gradient, are a type of post-hoc explainer that can provide token-level insights. However, explanations on the same input may vary greatly due to underlying biases of different methods. Users may be aware of this issue and mistrust their utility, while unaware users may trust them inadequately. In this work, we delve beyond superficial disagreement between attribution methods, structuring their biases through a model- and method-agnostic framework. We systematically assess both the lexical and position bias (_what_ and _where_ in the input) for two transformers; first, in a controlled, pseudo-random classification task where the model is not learning; then, in a semi-controlled causal relation detection task where the model properly learns. We find that lexical and position biases are structurally unbalanced in our model comparison, with models scoring high on the one type scoring low on the other. We also find signs that methods producing anomalous explanations are more likely to be biased themselves.

## Code guidelines:
* `prepare_artificial_data.py` : preprocesses synthetic datasets --> toy_data/
* `prepare_causal_data.py` : preprocess the causal relations dataset --> causal_data/exp1/
* `bert.py` : finetune the models (BERT, ModernBERT)
* `explain.py` : creates explanations for the dev/test instances.
* `analyse.py` : analysis script. 
* `run_models.sh` : loops and runs models.
* `compute_sufficiency.py` : main script to compute sufficiency for a model
* `compute_sufficiency.sh` : computes sufficiency in a loopt for different models
