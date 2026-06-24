# PDF Provenance Classifier — Robustness Under Image Distortion

Different software leaves distinct fingerprints in the binary structure of a PDF. This project
trains classifiers to detect **which tool produced a PDF** (Microsoft Word, Google Docs, or
Python/ReportLab) from its rendered binary image — and, more interestingly, measures **how robust
those classifiers stay when the images are degraded**.

The pipeline takes 894 binary images, augments them with 5 types of distortion, trains 4 classifiers,
and quantifies how much performance degrades under each distortion type.

## Dataset

- **Word images:** 398 PNGs (from Microsoft Word PDFs)
- **Google images:** 396 PNGs (from Google Docs PDFs)
- **Python images:** 100 PNGs (from ReportLab PDFs)
- **Total:** 894 originals, 5,364 after augmentation (6x)

## Repo structure

```
pdf-provenance-classifier/
├── README.md
├── SETUP.md
├── requirements.txt
├── data/
│   ├── original_pdfs/          (894 source images)
│   └── augmented_images/       (4,470 augmented images)
├── src/
│   ├── augmentation.py         - applies the 5 augmentations
│   ├── classification.py       - trains SVM, SGD, RF, XGBoost
│   ├── analysis.py             - runs evaluation + stats tests
│   └── utils.py                - shared data loading functions
├── results/
│   ├── confusion_matrices/     - 24 heatmap PNGs
│   ├── robustness_plots/       - accuracy degradation chart
│   ├── performance_metrics.csv
│   └── significance_tests.csv
└── reports/
    └── final_research_report.pdf
```

## Augmentations

1. Gaussian noise (σ between 5-20)
2. JPEG compression (quality 20-80)
3. DPI downsampling (300 → 150 or 72)
4. Random border cropping (1-3%)
5. Bit-depth reduction (8-bit → 4-bit)

## Classifiers

- SVM (RBF kernel) — baseline
- SGD (logistic loss) — baseline
- Random Forest (100 trees) — additional
- XGBoost (gradient boosting) — additional

## Results summary

Basically all 4 classifiers get ~99.9% accuracy on clean images. Most augmentations (gaussian noise, jpeg, bit-depth) dont really affect performance at all. The interesting stuff is that DPI downsampling tanks SVM to 72% while SGD stays fine, and cropping completely breaks all classifiers (drops to 60-67%). Full writeup is in the report under `reports/`.

## How to run

```bash
python src/augmentation.py       # generate augmented images
python src/classification.py     # train classifiers
python src/analysis.py           # evaluate + generate plots
```

See `SETUP.md` for environment setup instructions.
