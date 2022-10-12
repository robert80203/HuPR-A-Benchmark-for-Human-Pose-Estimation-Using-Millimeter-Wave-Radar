#  2D Human Pose Estimation using mmWave Radar

## Preparation

Setup the conda environment.

```
conda env create -f environment.yml
```

Run setup.py to generate the directories needed

Dowload the dataset and annotations from the following link
[HuPR dataset](https://drive.google.com/drive/folders/1-8f1eyjhaqly3RrmzAyKu99mObYsIkYG)

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

Evaluate model performance and visualize the results
```
python main.py --dir <YOUR_DIR> --config <YOUR_CONFIG>.yaml --eval --vis <YOUR_VIS_DIR>
```
YOUR_VIS_DIR specifies the directory where the results will be saved in visualization/

e.g.
```
python main.py --config mscsa_prgcn.yaml --dir mscsa_prgcn --eval
```