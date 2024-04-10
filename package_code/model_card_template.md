# Model Card

Classification model to predict individual income above 50K based on census
characteristics of US population.

## Model Details

Owner: Diego Canales
Type of model: Logistic_regression
Hyperparameters:
 C:1
 penalty: l1
 preprocessor__cat_transformer__drop: None
Data available on: https://archive.ics.uci.edu/dataset/20/census+income

## Intended Use

The model should be used to predict whether users are expected to earn
above 50K, based on the following characteristics:

{
    "age": 35,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Protective-serv",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

## Training Data

The model was trained using a 80% without replacement subset
and y stratification of the data.

## Evaluation Data

Data was evaluated using cross-validation on train data to select best model
and later tested on the 20% test-set.

## Metrics

recall: 0.740
precision: 0.616
fbeta: 0.672
accuracy: 0.855

## Ethical Considerations

Data publicly available, may contain biases on selection.

## Caveats and Recommendations

Additional Model Card information on: https://arxiv.org/pdf/1810.03993.pdf
