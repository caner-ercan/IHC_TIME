# IHC_TIME
Tumour regions semantic segmentation and cell nucleus instance segmentation


.<br/>
├── parenchyma-stroma/<br/>
│&emsp;├── scripts/<br/>
│&emsp;│&emsp;├── 1_slidePreperation/<br/>
│&emsp;│&emsp;│&emsp;├── Tissue_segmentation.groovy<br/>
│&emsp;│&emsp;│&emsp;├── Border_generation.groovy<br/>
│&emsp;│&emsp;│&emsp;└── mask_control.groovy<br/>
│&emsp;│&emsp;├── 2_slidePreperation/<br/>
│&emsp;│&emsp;│&emsp;├── deleteEmptyTiles.ijm<br/>
│&emsp;│&emsp;│&emsp;└── exporttilesandlabels.groovy<br/>
│&emsp;│&emsp;├── 3_Training/<br/>
│&emsp;│&emsp;│&emsp;├── Dataset_splitter.py<br/>
│&emsp;│&emsp;│&emsp;├── LR_finder.py<br/>
│&emsp;│&emsp;│&emsp;├── trainer_step1.py<br/>
│&emsp;│&emsp;│&emsp;├── trainer_step2.py<br/>
│&emsp;│&emsp;│&emsp;└── Inference.py<br/>
│&emsp;│&emsp;└── 4_Prediction/<br/>
│&emsp;│&emsp;&emsp; ├── Region_export.groovy<br/>
│&emsp;│&emsp;&emsp; └── MaskReImport.groovy<br/>
│&emsp;└── models/<br/>
│&emsp;&emsp; └── parenchyma-stroma/<br/>
│&emsp;&emsp;&emsp;  └── best_unet-resnet34_after-unfreeze-WD-1_BS32_20220317_1041.pth<br/>
└──&nbsp;cell_detection/
&emsp;&nbsp;├──&nbsp;scripts/<br/>
&emsp;&nbsp;│&emsp;├──&nbsp;1_dataPreperation/<br/>
&emsp;&nbsp;│&emsp;│&emsp;├──&nbsp;cell_annotation_export.groovy<br/>
&emsp;&nbsp;│&emsp;│&emsp;└──&nbsp;data.ipynb<br/>
&emsp;&nbsp;│&emsp;├──&nbsp;2_training/&ensp;<br/>
&emsp;&nbsp;│&emsp;│&emsp;├──&nbsp;training.py&ensp;<br/>
&emsp;&nbsp;│&emsp;│&emsp;└──&nbsp;Inference.ipynb&nbsp;<br/>
&emsp;&nbsp;│&emsp;├──&nbsp;3_prediction/<br/>
&emsp;&nbsp;│&emsp;│&emsp;└──&nbsp;stardistDetection.groovy<br/>
&emsp;&nbsp;│&emsp;└──&nbsp;4_classification/<br/>
&emsp;&nbsp;│&emsp;&emsp;&nbsp;├──&nbsp;run_cell_classifier.groovy<br/>
&emsp;&nbsp;│&emsp;&emsp;&nbsp;├──&nbsp;data_export_for_confMatrix.groovy<br/>
&emsp;&nbsp;│&emsp;&emsp;&nbsp;└──&nbsp;cellClassification_confusionMatrix.Rmd&emsp;&emsp;<br/>
&emsp;&nbsp;└──&nbsp;models/<br/>
&emsp;&emsp;&ensp;├── detection/<br/>
&emsp;&emsp;&ensp;│&emsp;├── modelstardist_final_bs2_epoch800_20220527_1434.zip<br/>
&emsp;&emsp;&ensp;│&emsp;└── stardist_final_bs2_epoch800_20220527_1434/<br/>
&emsp;&emsp;&ensp;└── classification/<br/>
&emsp;&emsp;&emsp;&emsp;└── cell_classifier.json<br/>