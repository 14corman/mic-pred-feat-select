# Scripts
All Python scripts go here. 

## Getting libraries set up
There is a `requirements.txt` file in this folder that will allow you to have the same library versions that was used when originally running these scripts. You can run this script, after you have a venv set up for python, by running `pip install -r requirements.txt` within this directory.

## Order to call scripts
1. `python get_format_data.py`
2. (RUN XGBoost training jupyter notebook in Sagemaker to train XGBoost)
3. `python xgboost_test.py`
4. `python nn.py cv`
5. `python nn.py files`
6. `python nn.py train`
7. `python nn.py test`

For more information, look at the file comment at the top of each python file. There will be more documentation and instructions for each file there.
