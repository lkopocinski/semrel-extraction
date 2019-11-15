# Semrel Extraction
A project focused on mining semantic relations.

### Package tree

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