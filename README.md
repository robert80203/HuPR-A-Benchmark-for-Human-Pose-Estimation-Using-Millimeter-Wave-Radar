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
python main.py --saveDir YOUR_DIR --config CONFIG_DIR/YOUR_CONFIG.yaml
```
YOUR_DIR specifies the diretory that will be saved in logs/

e.g.
```
python main.py --saveDir mfc_Pad_CltRm_20210907 --config 20210906_Pad_CltRm_128Chirps/mfc_128Chirps_temporal.yaml
```

## Evaluation
```
python main.py --loadDir YOUR_DIR/model_best.pth --config CONFIG_DIR/YOUR_CONFIG.yaml --eval
```
YOUR_DIR specifies the diretory that will be loaded from logs/

Evaluate model performance and visualize the results
```
python main.py --loadDir YOUR_DIR/model_best.pth --config CONFIG_DIR/YOUR_CONFIG.yaml --visDir YOUR_VIS_DIR --eval --vis
```
YOUR_VIS_DIR specifies the directory where the results will be saved in visualization/

e.g.
```
python main.py --loadDir mfc_Pad_CltRm_20210907/model_best.pth --config 20210906_Pad_CltRm_128Chirps/mfc_128Chirps_temporal.yaml --visDir test --eval --vis
```