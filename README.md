# IHC_TIME
Tumour regions semantic segmentation and cell nucleus instance segmentation

the models are available in here: 

[<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6806312.svg">](https://zenodo.org/badge/DOI/10.5281/zenodo.6806312)

```
parenchyma-stroma/
├── scripts/
│   ├── 1_slidePreparation/
│   │   ├── Tissue_segmentation.groovy
│   │   ├── Border_generation.groovy
│   │   └── mask_control.groovy
│   ├── 2_slidePreparation/
│   │   ├── deleteEmptyTiles.ijm
│   │   └── exporttilesandlabels.groovy
│   ├── 3_Training/
│   │   ├── Dataset_splitter.py
│   │   ├── LR_finder.py
│   │   ├── trainer_step1.py
│   │   ├── trainer_step2.py
│   │   └── Inference.py
│   └── 4_Prediction/
│       ├── Region_export.groovy
│       └── MaskReImport.groovy
└── models/
    └── parenchyma-stroma/
        └── best_unet-resnet34_after-unfreeze-WD-1_BS32_20220317_1041.pth

cell_detection/
├── scripts/
│   ├── 1_dataPreparation/
│   │   ├── cell_annotation_export.groovy
│   │   └── data.ipynb
│   ├── 2_training/
│   │   ├── training.py
│   │   └── Inference.ipynb
│   ├── 3_prediction/
│   │   └── stardistDetection.groovy
│   └── 4_classification/
│       ├── run_cell_classifier.groovy
│       ├── data_export_for_confMatrix.groovy
│       └── cellClassification_confusionMatrix.Rmd
└── models/
    ├── detection/
    │   ├── modelstardist_final_bs2_epoch800_20220527_1434.zip
    │   └── stardist_final_bs2_epoch800_20220527_1434/
    └── classification/
        └── cell_classifier.json
```
