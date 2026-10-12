# Dataset, attribution and availability

The research evaluation uses the synthetic steam-generator dataset associated with:

> Zhe Yang, Piero Baraldi, Enrico Zio. *A multi-branch deep neural network model for failure prognostics based on multimodal data*. Journal of Manufacturing Systems, 59 (2021), 42–50. https://doi.org/10.1016/j.jmsy.2021.01.007

The local dataset README requests citation of that paper. Its redistribution license was not established during release preparation, so the repository does not redistribute raw records, images, target vectors or trained weights. Obtain the original dataset from its authors' distribution. An independently verified direct download URL is not currently provided; the paper link is attribution, not an automatic download endpoint. The package supplied for the reported experiments is identified by its file fingerprint in `results/validation/protocol.json`.

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

The supplied dataset contains 50 equipment trajectories and 7,626 observations. Its existing exploration and the initial stress-test-driven design change are disclosed: the reported five-fold results are internal exploratory validation, not pristine external confirmation.

The source-code MIT license applies to MERA-PHM code and its original toy generator. It does not grant rights to the original paper dataset or external material. The toy generator is not the authors' steam-generator simulator and cannot confirm domain accuracy.
