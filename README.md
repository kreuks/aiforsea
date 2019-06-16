# AI FOR S.E.A

## Safety
#### 1. Environment Setup
- Please install miniconda3 or anaconda3
- Open terminal and `cd` to root of directory (`aiforsea/`)
- Type `conda env update`
- Type `source activate aiforsea`
#### 2. Data Preparation
- Your features data should be in `features` folder and your label data (if you want to training) should be in `label` folder
- For example for predict:
```
aiforsea
└───safety
    └───data
        └───features
            │───part01.csv
            │───part02.csv
            │───part03.csv
            │───part04.csv
            └───part05.csv
```
- For example for training:
```
aiforsea
└───safety
    └───data
        └───features
        │   │───part01.csv
        │   │───part02.csv
        │   │───part03.csv
        │   │───part04.csv
        │   │───part05.csv
        └───label
            └───label.csv
```

#### 3. Predict
- Change directory to root dir (`aiforsea/`)
- Type `python -m safety.main -m predict -if <input_folder> -o <output_file_path>`
- Example: `python -m safety.main -m predict -if 'safety/data/features' -o result.csv`
#### 4. Training
- Change directory to root dir (`aiforsea/`)
- Type `python -m safety.main -m training -if <input_features_folder> -il <input_label_folder>`
- Example: `python -m safety.main -m training -if 'safety/data/features/' -il 'safety/data/labels/'` 
