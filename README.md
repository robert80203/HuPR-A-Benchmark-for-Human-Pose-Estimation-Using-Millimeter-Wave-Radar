#  HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar

This is the official implementation of [HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar](https://arxiv.org/abs/2210.12564)

Please cite our WACV 2023 paper if our paper/implementation is helpful for your research:
```
@InProceedings{Lee_2023_WACV,
    author    = {Lee, Shih-Po and Kini, Niraj Prakash and Peng, Wen-Hsiao and Ma, Ching-Wen and Hwang, Jenq-Neng},
    title     = {HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5715-5724}
}

```

## Preparation

Setup the conda environment.

```
conda env create -f environment.yml
```

Run setup.py to generate the directories needed

Dowload the dataset and annotations from the following link
[HuPR dataset](https://drive.google.com/drive/folders/1-8f1eyjhaqly3RrmzAyKu99mObYsIkYG)

Please request access with your institutional email id and please provide following information
- Your Full Name
- Institution
- Advisor/Supervisor Name
- Current Position/Title
- Emaill Address (with institutional domain name)
- Purpose


Extract the dataset in the 'preprocessing/raw_data/iwr1843'


## Preprocessing

Preprocess the raw radar data collected by two radar sensors (IWR1843Boost)

```
cd preprocessing
python process_iwr1843
```

Each dataset structure should be aligned in this way:
```
data/HuPR
    - hrnet_annot_test.json
    - hrnet_annot_val.json
    - hrnet_annot_train.json
    - single_1
        - hori
            - 000000000.npy
            .
            .
            .
        - vert
            - 000000000.npy
            .
            .
            .
        - visualization
    - single_2
    .
    .
    .
```

Specify the root of dataset HuPR in YOUR_CONFIG.yaml -> DATASET -> dataDir

Download the PythonAPI of COCO [here](https://github.com/cocodataset/cocoapi)

Replace coco.py and cocoeval.py with misc/coco.py and misc/cocoeval.py respectively

## Training
```
python main.py --config <YOUR_CONFIG>.yaml --dir <YOUR_DIR>
```
YOUR_DIR specifies the diretory that will be saved in logs/

e.g.
```
python main.py --config mscsa_prgcn.yaml --dir mscsa_prgcn
```

## Evaluation
```
python main.py --dir <YOUR_DIR> --config <YOUR_CONFIG>.yaml --eval
```
YOUR_DIR specifies the diretory that will be loaded from logs/

The loaded weighted should be named as 'model_best.pth'

Download the trained weight from the [drive](https://drive.google.com/file/d/1Hmi2mw_KuSBCS4PVKI7dGWrRmtpKHkJI/view?usp=sharing)

Evaluate model performance and visualize the results
```
python main.py --dir <YOUR_DIR> --config <YOUR_CONFIG>.yaml --eval --vis <YOUR_VIS_DIR>
```
YOUR_VIS_DIR specifies the directory where the results will be saved in visualization/

e.g.
```
python main.py --config mscsa_prgcn.yaml --dir mscsa_prgcn --eval
```
