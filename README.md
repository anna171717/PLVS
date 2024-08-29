# PLVS
Source code of PLVS for Visual Storytelling task.


## Environments

- CUDA 10.1
- python 3.8
- pytorch 2.0.1

## Run

### Datasets

Download VIST ResNet152 features and put in project directory, make sure in `src_xxx/dataset.py` the path is correct. (Acutally you can generate ResNet152 features from original dataset)


### Generate Rake plan or concept

```bash
cd src_xxx
python dataset.py
```

More details in `dataset.py`. 

### Train or Test model

```bash
cd src_xxx
python main.py
```

You can change options(including 'train' or 'test') and hyperparameters at `src_xxx/main.py`.

