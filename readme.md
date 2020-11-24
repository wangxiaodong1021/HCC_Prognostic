
## Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning
[Link to paper](https://gut.bmj.com/content/early/2020/09/29/gutjnl-2020-320930)

PyTorch implementation of "Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning".
### Abstract
#### Objective
Tumour pathology contains rich information, including tissue structure and cell morphology, that reflects disease progression and patient survival. However, phenotypic information is subtle and complex, making the discovery of prognostic indicators from pathological images challenging.
#### Design 
An interpretable, weakly supervised deep learning framework incorporating prior knowledge was proposed to analyse hepatocellular carcinoma (HCC) and explore new prognostic phenotypes on pathological whole-slide images (WSIs) from the Zhongshan cohort of 1125 HCC patients (2451 WSIs) and TCGA cohort of 320 HCC patients (320 WSIs). A tumour risk score (TRS) was established to evaluate patient outcomes, and then risk activation mapping (RAM) was applied to visualise the pathological phenotypes of TRS. The multi-omics data of The Cancer Genome Atlas(TCGA) HCC were used to assess the potential pathogenesis underlying TRS.
#### Results 
Survival analysis revealed that TRS was an independent prognosticator in both the Zhongshan cohort (p<0.0001) and TCGA cohort (p=0.0003). The predictive ability of TRS was superior to and independent of clinical staging systems, and TRS could evenly stratify patients into up to five groups with significantly different prognoses. Notably, sinusoidal capillarisation, prominent nucleoli and karyotheca, the nucleus/cytoplasm ratio and infiltrating inflammatory cells were identified as the main underlying features of TRS. The multi-omics data of TCGA HCC hint at the relevance of TRS to tumour immune infiltration and genetic alterations such as the FAT3 and RYR2 mutations.
#### Conclusion 
Our deep learning framework is an effective and labour-saving method for decoding pathological images, providing a valuable means for HCC risk stratification and precise patient treatment.
  
### Prerequisites:
```
Python 3.7
Numpy 1.17.2
Scipy 1.3.0 
Pytorch 1.3.0/CUDA 10.1
torchvision 0.4.1
Pillow 6.0.0
opencv-python 4.1.0.25
openslide-python 1.1.1
Tensorflow 2.0.0
Tensorboard 2.0.0
tensorboardX 1.7
pandas 0.25.3
lifelines 0.23.4
```

## Data Preparation:
Use gdc_manifest.2020-01-03.txt to download WSI from TCGA dataset.
Use the download tool 'gdc-client' to download data. Run the command:
gdc-client download -m gdc_manifest.2020-01-03.txt -d ./data/svs/

## Classification network:
### Data Preprocess:
Run python ./data_preprocess/cls/make_slide_cuting.py to generate the thumbnail.
Run python ./data_preprocess/cls/make_tissue_mask.py to generate the tissue mask.
Run python ./data_preprocess/cls/make_lab_mask.py and python ./data_preprocess/cls/make_other_mask.py to generate Tumor,
Liver, Stroma and Hemorrhage & Necrosis mask.
Run python ./data_preprocess/cls/create_patch.py to obtain patches.

Training data should be a dictionary stored to disk containing the following keys set in ./configs/resnet18_tcga.json:
"data_path_40": the path to store tiled slides in non-overlapping 768x768 pixel windows at a magnification of 40x
"json_path_train":the path to store the training json containing the dictionary of a list of tuple (x,y) coordinates. 
"json_path_valid":the path to store the valid json containing the dictionary of a list of tuple (x,y) coordinates. 
Valid data should contain the slides and masks of the foreground area for each slide.

### Classification network training:
To train a model, use script ./Classification/bin/train_tcga.py. Run python ./Classification/bin/train_tcga.py -h to get help regarding input parameters.
The data path and label path are set in ./configs/resnet18_tcga.json (data_path_40,  json_path_train, json_path_valid).

### Classification network valid: 
To run a model on test set use script ./Classification/bin/probs_map.py. Run python ./Classification/bin/probs_map.py -h to get help regarding input parameters.
Script outputs: Segmentation mask maps of different tissues for each slide.

## Prognostic network:
### Data Sample:
Sample patches from slides according to the segmentation mask maps generated by Classification network.
Use ./data_preprocess/prognostic/sample_patch_x4.py, ./data_preprocess/prognostic/sample_patch_x10.py,
./data_preprocess/prognostic/sample_patch_x40.py to generate 40x, 10x, and 4x patches.

### Prognostic network training:
To train the prognostic network, use script ./Prognositic/bin/train.py. Run python  ./Prognositic/bin/train.py -h for help regarding input parameters.

### Prognostic network valid:
To run a model on a test set, use script ./Prognositic/vis/vis_save.py. Run python  ./Prognositic/vis/vis_save.py -h for help regarding input parameters.
### Script outputs:
.csv file with TRS(tumor risk score), patients' OS, outcome, patient ID.
To generate heatmap , use script ./Prognositic/vis/vis_save_1.py. Run python  ./Prognositic/vis/vis_save_1.py -h for help regarding input parameters.

## Citation

```bibtex
@article {Shigutjnl-2020-320930,
  title = {Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning},
	author = {Shi, Jie-Yi and Wang, Xiaodong and Ding, Guang-Yu and Dong, Zhou and Han, Jing and Guan, Zehui and Ma, Li-Jie and Zheng, Yuxuan and Zhang, Lei and Yu, Guan-Zhen and Wang, Xiao-Ying and Ding, Zhen-Bin and Ke, Ai-Wu and Yang, Haoqing and Wang, Liming and Ai, Lirong and Cao, Ya and Zhou, Jian and Fan, Jia and Liu, Xiyang and Gao, Qiang},
	elocation-id = {gutjnl-2020-320930},
	year = {2020},
	doi = {10.1136/gutjnl-2020-320930},
	publisher = {BMJ Publishing Group},
	URL = {https://gut.bmj.com/content/early/2020/09/29/gutjnl-2020-320930},
	eprint = {https://gut.bmj.com/content/early/2020/09/29/gutjnl-2020-320930.full.pdf},
	journal = {Gut}
}
```
