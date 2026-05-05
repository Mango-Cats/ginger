# ginger

A Transformer-based grapheme-to-phoneme for Filipino.

The primary references of this project are listed below:

- <https://aclanthology.org/P16-1038/>
- <https://arxiv.org/abs/2004.06338>

## Moving Around

The main training pipeline is orchestrated on [`ginger.ipynb`](/ginger.ipynb).

There are four main modules in the source folder:

1. [`architecture`](/src/architecture/) contains the learning
   architecture. The architecture is an implementation of what was
   discussed in [2004.06338](https://arxiv.org/abs/2004.06338)
1. [`dataman`](/src/dataman/) contains the data management function
   definitions (i.e., preprocessing, cleaning, etc.)
1. [`trainer.py`](/src/trainer.py) contains the training pipeline.
1. [`evaluation.py`](/src/evaluation/) contains the evaluation pipeline
   and metrics.

The dataset is extracted from Wiktionary using [wikipron](https://github.com/CUNY-CL/wikipron) [(Lee et al, 2020)](https://aclanthology.org/2020.lrec-1.521/).
A copy of this dataset is stored in [`data/`](/data/).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the Python
project manager. Simply run

```bash
uv sync
```

Then set your Jupyter Kernel as the Python Vitrual Environment created
by uv.

## Working on Google Colab

The `ginger` notebook is organized in such a way that allows one to run
it in their local machine or via Google Colab. For Google Colab, be
sure to have a GitHub token ready.

Then simply run the notebook in Google Colab (no additional setup
needed). To see how this works, look at the "Setup" section in the
`ginger` notebook.
