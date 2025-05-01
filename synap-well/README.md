# synap-well

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Machine Learning project that combines health, mood, weather, activity preferences for healthy mental health/fitness activity suggestions.

## Summary:

For some people, mental well-being is an ongoing battle that has people struggling and thriving on both sides of the spectrum. Nevertheless, it is a battle we must all face as we guide ourselves through the vicissitudes of everyday life. This is especially the case for people who experience seasonal affective disorder (SAD) and find it genuinely difficult to find a positive balance in their life. This situation becomes even further exacerbated when taking into account non-ideal weather conditions. As such, our group has decided to contrive a machine-learning project that will help alleviate that mental load through thoughtful suggestions and recommendations.  


To elaborate, this project aims to develop a personalized AI coach that optimizes daily activity recommendations based on weather conditions, mental well-being, and physical fitness. Many individuals struggle with maintaining consistent fitness routines and mental well-being due to external factors such as weather changes, seasonal affective disorder, and lifestyle habits. By leveraging machine learning and large language models (LLMs), this AI system provides personalized recommendations for workouts and mindfulness exercises, helping users make informed lifestyle choices.


## Installation Steps (Windows)
### 1. Create a Python virtual environment in your terminal with the following command:
python -m venv myenv
### 2. Activate your python virtual environment venv in your terminal with the following command:
./myenv/Scripts/activate
### 3. Download the related packages with the following pip command in your terminal:
pip install -r requirements.txt

This project requires Cuda Toolkit 12.4 to be installed.

Also, remember to create a .env file in the following format with corresponding populated values:

```
HUGGING_FACE_TOKEN=<Hugging_Face_Token_ID>
WEATHER_API_KEY=<WEATHER_API_KEY_ID>
# WEATHER_API_KEY is from weatherapi.com
```



### The main functionality of this project is found within main_workflow.ipynb

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         synap_well and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── synap_well   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes synap_well a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

