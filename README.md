# DESN2000-BINF-Multiomics

## Description
This project is a collaboration with the scientific software engineers and metabolomics researchers from the Victor Chang Cardiac Research Institute’s Innovation Centre to develop a dashboard to analyse and visualise multiomics data acquired from mass spectrometry experiments.

## Prerequisites
- Python 3.11.2

## Setting up the development environment.
    * Create a virtual environment.
'''
    python3 -m venv .venv
    source .venv/bin/activate
'''
    * install the requirements.
'''
    pip3 install -r equirements.txt
'''
    * navigate to the "vccri_dashboard" folder and run the following commands (these must be ran once only, for configuration purposes):
'''
    python manage.py makemigrations
    python manage.py migrate
'''
    * to run the local host development dashboard: 
'''
    python manage.py runserver
'''
    * navigate to the URL in the terminal. 

## Contributors
* Sasha Barisic
* Qianqian Zhang
* Dnyanda Kulkarni
* Venkata Mandadi
