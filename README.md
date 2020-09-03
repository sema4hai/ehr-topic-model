# EHR Topic Modeling Pipeline
A topic modeling pipeline for EHR notes.

## Getting Started
### Prerequisites
(These are the versions used during development, have not tested against other versions so use these to be safe.)
* Poetry 1.0.10
* SQLite 3.7.17
* Python 3.8.5
* numpy 1.19.1
* optuna 2.0.0
* pandas 1.1.1
* pyyaml 5.3.1
* scikit-learn 0.23.2
* tmtoolkit 0.10.0

For the Python packages (numpy onwards), they are handled by Poetry so just make sure the first 3 are installed.

### Installation
#### Poetry
Make sure you have [Poetry](https://python-poetry.org/) installed first. Refer to their docs for installation instructions.

#### Clone
```sh
cd /your/dir/of/choice
git clone https://github.com/minghao4/ehr-topic-model/
```

#### Setup
Use Poetry to create a virtualenv and install dependencies to it.
```sh
cd ehr-topic-model/
poetry install --no-dev
```

## Usage
### Configuration
* Create a `data/` folder and place your data CSV file within it. Make sure the file has `note_id` and `full_note_norm` columns. Point the `data_file` option to this file within the configuration YAML.
* Place your stopwords file under the `config/` folder and point the `stopwords` option within the configuration YAML to this file.
* Edit the configuration YAML as needed.

### Training with Hyperparameter Tuning
Run `main.py` with Poetry to use the virtualenv.
```sh
poetry run python main.py -c config/hpt_config.yml -p 0
```

### Inference
* Place the data CSV file to perform inference on under `data/`. As with training, ensure the file has `note_id` and `full_note_norm` columns. Point the `data_file` option to this file within the configuration YAML.
* Point `model_file` option to the serialized model file under `models/`.
* Topics require human interpretation, edit the topics file under `models/` or leave as the default names. Point the `topics_file` option to this file.
Similar to training, run `topic.py` with Poetry.
```sh
poetry run python topic.py -c config/inference_config.yml
```

## Built With
* [Poetry](https://python-poetry.org) - Python package dependency manager.
* [scikit-learn](https://scikit-learn.org/stable/) - machine learning framework.
* [Optuna](https://optuna.org/) - hyperparameter tuning framework.
* [tmtoolkit](https://github.com/WZBSocialScienceCenter/tmtoolkit) - text mining and topic modeling toolkit.

## License
TBD

(sorry, not sure what license this should be under as this project was created as part of my work - still trying to figure this out)

(if you have any suggestions I'd be happy to hear you out!)
