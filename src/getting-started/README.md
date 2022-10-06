# ml-comp-scaffolding

## Creating an environment

The file `environment.yml` lists the dependencies required to run the code.
It is recommended to use Conda to create an environment. 

If you don't have a Conda installation, it is recommended to install [MiniConda](https://docs.conda.io/en/latest/miniconda.html).

To create an environment:
```
conda env create -f environment.yml
```

Once the environment is created, activate the environment:
```
conda activate ml-comp
```

If you don't want to install Conda, you can pip install the dependencis listed in `environment.yml` (ideally using some other form of virtual environment such as pyenv).

## Training a model

To train a model, download the ADC_Dataset folder and place it in the same folder as `train.py`, then run
```
python train.py
```

This can take a while. Once the training finishes, there should be a model saved as `resnet18_final.pth`

## Generating a CSV file with test predictions

Once you have trained the example model (`resnet18_final.pth`), run `evaluate.py` to load the test dataset and generate predictions. This script will output a `predictions.csv` file, which should have the same format as the `sample_predictions.csv` file included in the `ADC_dataset`.