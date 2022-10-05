#  2D Human Pose Estimation using mmWave Radar

## Preparation
Create visualization/, logs/, data/
```
mkdir visualization logs data
```

Specify the root of dataset (YOUR_DATASET_NAME) in YOUR_CONFIG.yaml -> DATASET -> dataDir

Each dataset structure will be like:
```
YOUR_DATASET_NAME
    - single_1
        - annot
            - hrnet_annot.json
        - hori
            - 000000000.npy
            .
            .
            .
        - verti
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