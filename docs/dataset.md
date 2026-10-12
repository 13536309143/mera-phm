# Dataset, attribution and availability

The research evaluation uses the synthetic steam-generator dataset associated with:

> Zhe Yang, Piero Baraldi, Enrico Zio. *A multi-branch deep neural network model for failure prognostics based on multimodal data*. Journal of Manufacturing Systems, 59 (2021), 42–50. https://doi.org/10.1016/j.jmsy.2021.01.007

Use the authors' dataset distribution and cite the paper above. Place the extracted files locally; the repository provides its own sample generator and source code. Dataset identity for the published evaluations is recorded in `experiments/numerical/tree-groupcv/protocol.json`.

Unpack the original `Multimodal data for prognostics` directory into `data/raw/`, or pass its root explicitly with `--data`. Its root must contain directories `1` through `50`:

```text
Multimodal data for prognostics/
├── 1/
│   ├── time.txt
│   ├── workorder.txt
│   ├── nextinslevel.txt
│   ├── Numerical/
│   │   ├── nextinstime.txt
│   │   ├── timefromlastrepair.txt
│   │   ├── Numberofchemicalcleaning.txt
│   │   └── Numberofmechanicalcleaning.txt
│   └── Image/1.tiff, 2.tiff, ...
└── 2/, ..., 50/
```

Each row is a current observation. The target is the degradation level at the **next scheduled inspection**, in arbitrary degradation units (`adu`). Time and the known inspection horizon are measured in arbitrary time units (`atu`). These are not percentages, days, current health classes, or remaining useful life.

The numerical inputs are the inspection horizon, elapsed time since repair, chemical-cleaning count, and mechanical cleanings since the preceding chemical cleaning. Images show simulated transient curves, not photographs. Missing images are blank curve plots with axes. An uncollected inspection text repeats the preceding record; the implemented text-change age is a proxy and can also miss newly collected but identically worded observations.

The supplied dataset contains 50 equipment trajectories and 7,626 observations. Numerical experiments use causal observation prefixes; target loading is restricted to offline training and evaluation.

The project MIT license covers its code and sample generator. Original paper data is obtained separately; its redistribution license was not established during release preparation.
