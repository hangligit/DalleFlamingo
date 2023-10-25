# Do DALL-E and Flamingo Understand Each Other?
###  [Project Website](https://dalleflamingo.github.io/) | [ICCV 2023 Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Do_DALL-E_and_Flamingo_Understand_Each_Other_ICCV_2023_paper.pdf)

This is the official code repository for the ICCV 2023 paper titled "Do DALL-E and Flamingo Understand Each Other?" Should you have any inquiries or require further assistance, please do not hesitate to reach us out.

## Installation Guide

The code base has been verified with python 3.9, CUDA Version 11.7, CUDA Driver Version 515.65.01. To get started, follow the steps below:

1. Create a new conda environment with Python 3.9:

```
conda create -n pytorch python=3.9
conda activate pytorch
```

2. Install the required dependencies:
```
git clone git@github.com:hangligit/DalleFlamingo.git
cd DalleFlamingo
pip install -r requirements.txt
pip install -e transformers
```
## Finetuned Weights
You can find our finetuned weights of BLIP and SD available for download:

<a href="https://drive.google.com/file/d/1_LEKreXqfO5RDIQliOm7SXNSiuWNBV_Z/view?usp=sharing">BLIP-Base</a> | 
<a href="https://drive.google.com/file/d/1VacTwvXdhiJ21OPr79MdBiiPhx5NabgB/view?usp=sharing">BLIP-Large</a> | 
<a href="https://drive.google.com/file/d/1ohoHzNjIPhArv5ftwwt0gmBSIJFixSTy/view?usp=sharing">BLIP2</a> | 
<a href="https://drive.google.com/file/d/1UQZRQm_W1Qgt4Ewn7rAr6WoKxrp0ur05/view?usp=sharing">SD-w/-Base</a> |
<a href="https://drive.google.com/file/d/1izBxVGlRGeH7Mfy1WWsWsdTd2Jz-7H9u/view?usp=sharing">SD-w/-Large</a>


## Data Download
To access the training data, please download the dataset from <a href="https://drive.google.com/file/d/1OL5RVf3d5-SXfKrcLCQ_eLRf3Uwzu3Aa/view?usp=sharing">here</a> and organize the data in the following structure. In the "captions" directory, you will find symbolic links to the actual image files, which should be stored in the "images" folder. The actual images can be downloaded from the <a href="https://cocodataset.org/#captions-2015">COCO website</a>.
```
/root_dir/datasets/coco/
    --captions
        --train
        --val
        --test
    --images
        --train2014
        --val2014
        --test2014
```

## Training Guide

After installation, use the following script to finetune the BLIP and SD model:

```
python train.py
```

## Evaluation Guide

After completing the training, you can evaluate the performance using the following two scripts:
### Evaluate Image-to-Text
```
python evaluate_blip.py
```
### Evaluate Text-to-Image
```
python evaluate_sd.py
```


## Citing our work
If you find our work valuable, please consider citing our research as follows:
```
@InProceedings{Li_2023_ICCV,
        author    = {Li, Hang and Gu, Jindong and Koner, Rajat and Sharifzadeh, Sahand and Tresp, Volker},
        title     = {Do DALL-E and Flamingo Understand Each Other?},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2023},
        pages     = {1999-2010}
    }
```