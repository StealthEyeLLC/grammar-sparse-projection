# Replication wanted

We want independent replications of GSP across GPUs, models, serving stacks, batch shapes, and grammar workloads.

## Highest-value replications

1. Datacenter GPUs: H100/H200, B100/B200, A100.
2. Consumer GPUs: RTX 4090/5090 and other Ada/Blackwell cards.
3. Other large-vocabulary models and output-head widths.
4. Integration experiments with XGrammar, vLLM, SGLang, or another production serving stack.
5. Heterogeneous batching with realistic mixtures of constrained and free-string states.

## Minimum report

Please include:

- GPU model and driver
- OS
- Python, PyTorch, CUDA, Triton, Transformers, and XGrammar versions
- model identifier and dtype
- batch size
- grammar/workload
- baseline definition
- GSP route selected
- raw timing samples, not only averages
- whether legal-token support and greedy top-1 matched
- commit SHA used

Open a Replication report issue even if the result is negative.

Negative results are valuable. A configuration where dense execution wins is part of the crossover map we want to measure.

## Reproduction entry points

See REPRODUCIBILITY.md and the raw evidence under results/.
