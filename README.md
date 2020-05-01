# Semrel Extraction
Repository contains a codebase used in research on the extraction of semantic relations (brand-product). 
Research description and results are included in the paper: 
["Brand-Product Relation Extraction Using Heterogeneous Vector Space Representations"](https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction/-/blob/develop/LREC_BP.pdf) 
published in [LREC2020](https://lrec2020.lrec-conf.org/en/) conference.  

## Frameworks
Two frameworks were used in the project. [DVC](https://dvc.org/doc) for versioning the datasets and [mlflow](https://mlflow.org/docs/latest/index.html) for tracking experiments.
To manage the project with ease consider familiarize with them.  

## Setup project

To setup the project in your machine perform following commands

Download repository: \
`$ git clone https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction.git`

Enter main folder: \
`$ cd semrel-extraction`

Download datasets related to actual commit: \
`$ dvc pull`

Then enter to docker folder: \
`$ cd docker`

Copy __credentials.template__ into __credentials__ files and fill with correct access keys. \
`$ cp deps/credentials.template deps/credentials`

Start docker: \
`$ docker-compose up`


## Repository packages
Repository also contains code for additional functionalities:

__docker__ - docker configuration and execution environment for semrel package. \
__mlflow__ - configuration and execution environment for mlflow server used for tracking experiments. \
__spert__ - scripts used to prepare dataset in format required to train [SpERT](https://github.com/markus-eberts/spert) model. \
__worker__ - scripts and execution environment to use trained model as a worker.


## FAQ
#### Where is data stored?
Data is versioned by [DVC](https://dvc.org/doc) which works like a git but for data. 
All data is stored on the remote storage (https://minio.clarin-pl.eu/minio/semrel/) in dvc folder.
To retrieve data execute:  

`$ git checkout [branch_name]`  
`$ git dvc checkout`

DVC will download all data related to actual commit.  

#### How to train and test a model?
There is a script __semrel/model/train.sh__ which starts training. 
Adjust training params in __semrel/model/config.yaml__ and then execute:\
`$ ./train.sh`

Training result will be automatically uploaded to mlflow server.
   
#### Do I need to setup anything on my machine?
Yes, to make mlflow log artifacts properly set environment variable, 
otherwise mlflow try to ping original Amazon S3 storage.  

`$ export MLFLOW_S3_ENDPOINT_URL=https://minio.clarin-pl.eu`  

add also config file filled with correct credentials:

`$ echo "[default]" > ~/.aws/credentials`  
`$ echo "aws_access_key_id = access_key" >> ~/.aws/credentials`  
`$ echo "aws_secret_access_key = secret_key" >> ~/.aws/credentials`

## How it works?
![Project diagram](https://github.com/lkopocinski/semrel-extraction/blob/master/arch-diagram.svg)
