# DPHeteSem

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cose.2026.105248-blue)](https://doi.org/10.1016/j.cose.2026.105248)

DPHeteSem releases multiple node-aligned semantic graph views under
semantic-edge-level differential privacy. The package publishes a shared
partition, noisy block counts, and noisy node degrees, then synthesizes sparse
graphs using only those released quantities.

## Paper

**DPHeteSem: Differentially private semantic graph release for privacy-preserving
data sharing in recommender systems**

Chenyue Luo, Xianjin Fang, Pengfei Zhang, and Zhi Liu

*Computers & Security*, 2026, article 105248.

- [Article / DOI](https://doi.org/10.1016/j.cose.2026.105248)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167404826004244)
- [BibTeX](CITATION.bib)

Accepted on 6 November 2026; available online as a **Journal Pre-proof**.
Volume and issue information will be added when assigned by the publisher.

## Scope

This repository contains the core data-release mechanism. It intentionally does
not include raw datasets, benchmark-specific loaders, or downstream recommender
implementations. Inputs and outputs use SciPy sparse `.npz` files so that the
released views can be consumed by different graph-learning pipelines.

The guarantee is scoped to one semantic-edge change under the adjacency defined
in the manuscript. It is not a user-level, attribute-level, membership-level,
or original-interaction guarantee.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Quick start

Create two small node-aligned semantic views:

```bash
python examples/create_demo_views.py --output demo-input
```

Release them with a total privacy budget of 1.0:

```bash
dphetesem release \
  --views demo-input/view_1.npz demo-input/view_2.npz \
  --output demo-output \
  --epsilon 1.0 \
  --blocks 4 \
  --seed 2024
```

Windows PowerShell users can place the command on one line or replace `\` with
the PowerShell continuation character.

The output directory contains:

- `released_view_1.npz`, `released_view_2.npz`: synthetic sparse views;
- `partition.npy`: the shared node partition;
- `release_metadata.json`: configuration, output sizes, and privacy accounting.

Validate the generated package:

```bash
python scripts/validate_release.py demo-output
```

## Input contract

Each input must be a square, binary-compatible, symmetric SciPy sparse matrix.
All views in one release must have the same shape and node ordering. Self-loops
are removed before statistics are computed. The release command never writes
the original views to its output directory.

## Privacy accounting

The command accepts one total budget `epsilon`. A configurable fraction is used
for the shared partition sketch. The remaining budget is divided evenly across
views and then split between block-count and degree releases. The accounting
record in `release_metadata.json` verifies that

```text
partition budget + sum(block-count budgets) + sum(degree budgets) = epsilon.
```

Cross-view coupling and graph synthesis use only noisy released statistics and
therefore add no privacy cost by post-processing.

## Tests

```bash
python -m ruff check .
python -m pytest
```

The test suite checks deterministic seeded execution, privacy-budget
composition, sparse output invariants, and rejection of invalid inputs.

## Reproducibility notes

- Set `--seed` to reproduce the randomized release for fixed inputs and package
  versions.
- Record the complete `release_metadata.json` file with every experiment.
- Do not tune a release by repeatedly inspecting private data. Repeated releases
  require additional privacy accounting.
- Do not publish raw semantic views or benchmark data unless their licenses and
  privacy conditions explicitly permit redistribution.

## Repository layout

```text
src/dphetesem/       Core package and command-line interface
examples/            Synthetic input generator
tests/               Functional and accounting tests
.github/workflows/   Continuous integration configuration
```

## Citation

If you use DPHeteSem in your research, please cite the article:

> Chenyue Luo, Xianjin Fang, Pengfei Zhang, and Zhi Liu. DPHeteSem:
> Differentially private semantic graph release for privacy-preserving data
> sharing in recommender systems. *Computers & Security*, 2026, 105248.
> https://doi.org/10.1016/j.cose.2026.105248

```bibtex
@article{luo2026dphetesem,
  title   = {{DPHeteSem}: Differentially private semantic graph release for privacy-preserving data sharing in recommender systems},
  author  = {Luo, Chenyue and Fang, Xianjin and Zhang, Pengfei and Liu, Zhi},
  journal = {Computers \& Security},
  year    = {2026},
  pages   = {105248},
  doi     = {10.1016/j.cose.2026.105248},
  url     = {https://doi.org/10.1016/j.cose.2026.105248}
}
```

The same entry is available in [`CITATION.bib`](CITATION.bib). GitHub's
**Cite this repository** button uses the preferred article citation in
[`CITATION.cff`](CITATION.cff).

## License

The standalone implementation in this repository is released under the MIT
License. See `PROVENANCE.md` for the source boundary.
