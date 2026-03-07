Markdown# MomentumSort

**Distribution-Aware Sorting via Entropy Projection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A two-stage sorting algorithm that removes **~50% of comparisons** on real-world skewed data by projecting onto a generalized-gamma manifold using only three empirical moments.

**Paper**: *MomentumSort: Distribution-Aware Sorting via Entropy Projection*  
**Author**: Kwanhee Lee

---

## ✨ Features

- **Correctness** — Always produces a correctly sorted array (provably correct for arbitrary inputs)
- **Adaptive** — Automatically detects skewness and projects onto linear or gamma rank space
- **Efficient on real data** — Removes ~50% of residual comparisons on skewed numerical columns
- **Graceful degradation** — Falls back to standard behavior when no structure exists
- **Recursive extension** — Further reduces comparisons on heavy-tailed data
- **Clean reference + optimized implementation** included

## 🚀 Quick Start

```bash
pip install numpy scipy pandas
Pythonfrom core.momentumsort import MomentumSort
import numpy as np

ms = MomentumSort()
data = np.random.gamma(0.5, 1, 100_000)        # highly skewed example
sorted_data = ms.sort(data)
```

## 📊 Reproduce the Paper Results

```Bash
# 1. Download all real datasets (one-time)
python -m data.download
```

# 2. Run the exact benchmarks from the paper
```
python -m benchmarks.synthetic
python -m benchmarks.real
python -m benchmarks.scaling      # n=200k + recursion demo
```
## 📁 Repository Structure
```
textmomentumsort/
├── core/momentumsort.py
├── benchmarks/
│   ├── synthetic.py
│   ├── real.py
│   └── scaling.py
├── data/
│   ├── download.py
│   └── *.csv (auto-downloaded)
├── tests/test_correctness.py
├── README.md
└── requirements.txt
```
## 📖 Citation
```
bibtex@misc{lee2026momentumsort,
  author       = {Kwanhee Lee},
  title        = {MomentumSort: Distribution-Aware Sorting via Entropy Projection},
  year         = {2026},
  howpublished = {\url{https://github.com/gigantul/momentumsort}},
  note         = {SSRN preprint}
}
```
## 📄 License
MIT License — free to use in research, production, or database engines.