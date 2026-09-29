# Contributing

Contributions are welcome, especially independent reproductions, portability work, new GPU/model measurements, kernel improvements, and serving-stack integrations.

Before proposing a performance change:

1. preserve exact constrained-token support;
2. report greedy top-1 agreement against the dense masked baseline;
3. include raw timing evidence and environment metadata;
4. test a dense or near-unconstrained control;
5. do not remove negative results.

By intentionally submitting a contribution for inclusion in this repository, you agree that the contribution is provided under the repository's applicable license terms, including Apache-2.0 for software contributions.

For benchmark-only results, open a Replication report issue; a code change is not required.
