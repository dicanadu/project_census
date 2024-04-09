# Income Classificator ML model predictor

## Objective

To test a Logistic Regression model to predict whether salary
of a person is less (0) or above 50k (1)

Data: https://archive.ics.uci.edu/dataset/20/census+income

## How to use this project

Install package running the command:

```bash
pip install package_code/
```

To run, build and slice a ML model, use the command adjusting parameters
metric, save, and slicing:

```bash
#default parameters --metric 'accuracy' --save True --slicing None
python create_model_slicing.py

#To slice on race
python create_model_slicing.py --slicing 'race'
```

Slicing options:
- workclass,
- education,
- marital-status,
- occupation,
- relationship,
- race,
- sex,
- native-country

Models will be saved on directory package_code/model/