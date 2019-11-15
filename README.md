# Semrel Extraction
A project focused on mining semantic relations.

## Package tree

+-- .dvc : contains config for Data Version Control   
+-- data : contains all dataset, transformed data, vector models and data preparation pipeline scripts nr_*.sh are scripts with DVC pipeline command  
|&nbsp;&nbsp;&nbsp;&nbsp;+-- scripts : scripts used to prepare data, called by .sh pipeline scripts  
+-- docker : contains Docker file for entire project environment (not finished)  
+-- relextr : contains training and testing scripts train.sh, test.sh - last scripts in DVC pipeline  
|&nbsp;&nbsp;&nbsp;&nbsp;+-- evaluation : contains scripts for visual evaluation of a model (due to major changes in project api they could not working correctly)  
|&nbsp;&nbsp;&nbsp;&nbsp;+-- model : contains code with neural network architecture, train, test scripts and utils for them  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- config : contains config used as parametrization for train and test scripts, change in this file will impact dvc pipeline  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scripts : scripts  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- model :  POJO classes  
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- utils : batches.py - batch loader, engines.py - implementation of different type of vectorizers, metrics.py - metrics holder  

## Frameworks
To mage the project with ease consider familiarize with [DVC](https://dvc.org/doc) and [mlflow](https://mlflow.org/docs/latest/index.html) frameworks.  

## FAQ

#### Where is data stored?
Data is verisioned by DVC which works like a git. All data is stored on the remote storage (https://minio.clarin-pl.eu/minio/semrel/) in semrel bucket.
To retrieve data execute:  

$ git checkout [branch_name]  
$ git dvc checkout  

DVC will download all data related to actual commit.  

#### How to train and test a model?
Make changes in config [train.yaml, test.yaml] or any other dependent script. Do not forget to pass apropriate experiment_name and tags. Then in main repository directory execute:  

$ dvc repro train.dvc  
$ dvc repro test.dvc  

Result will be automaticaly uploaded to mlflow server and visible at http://10.17.50.132:8080/ 
Please commit files after each successful run as *.dvc metrics and model will change.  

#### Do I need to setup anything on my machine?
Yes, to make mlflow log artifacts properly set environment variable otherwise mlflow try to ping orginal Amazon S3 storage.  

export MLFLOW_S3_ENDPOINT_URL=https://minio.clarin-pl.eu  

add also config file:

echo "[default]" > ~/.aws/credentials  
echo "aws_access_key_id = access_key" >> ~/.aws/credentials  
echo "aws_secret_access_key = secret_key" >> ~/.aws/credentials  

