# Basics

- Google cloud **console** (Open [link](https://console.cloud.google.com/) with personal account)


- Some Important **Navigation**

    - Navigation menu `>` Vertex AI `>` Workbench

        - **Vertex AI** `:` Platform for ML Applications to be built on top of google cloud

        - **Workbench** `:` Gives interface like Jupyter Lab environment (powered by backing)
        
            - **Note** : In Google colab underlying Hardware cannot be controlled
    
    </br>

    - Navigation menu `>` Cloud Storage `>` Buckets

        - Consider **Bucket** as **Folder** and **Object** as **file**
    
    </br>

    - Get Project ID

        - Open Project Picker `>` My First Project `>` ID


## Create a New Instance of Workbench

1. Navigation menu `>` Vertex AI `>` Workbench

2. Instances `>` Create New

3. New Instance
    - Continue with default paramenters for Name, Region, Zone

    - **Disable** `:` Attach 1 NVIDIA T4 GPU

    - **Disable** `:` Apache Spark and BigQuery kernels

    - Default Network setting is fine

4. Advanced Options
    - **Details** `:` Default
    
    - **Environment** `:` Default

    - **Machine type** `:`

        - Series `:` **E2**

        - Machine type `:` **e2-standard-2 (2 vCPU, 1 core, 8 GB memory)**

        - Idle shutdown `:` **60** min

    - **Disks** `:`

        - Boot disk type `:` **Standard** Persistent Disk (150 GB)

        - Data disk type `:` **Standard** Persistent Disk (100 GB)
    
    - **Networking** `:` Default

    - **IAM and security** `:` Default

    - **System Health** `:` Default

    ``>`` Create

    Pricing Summary **US$57.60** monthly estimate **if running continuously**


## Open JupyterLab

- Navigation Menu `>` Vertex AI `>` Workbench

- Open JupyterLab from the provisioned instance

- **Delete** the *notebook_template* which is there by default

- Upload the files as needed


# Week 1- Graded Assignment 1

Setting up the ML pipeline for **IRIS Classifier** in Vertex AI platform using GCS as demonstrated in the lecture (Hands-on: Introduction to Google Cloud, Vertex AI) in your GCP account.

### Prerequisite

- Activate your GCP Trial

- Setup Vertex AI Workbench (Enable appropriate services/api as required)

### Assignment Objective

1. Store Training Data in Google Storage Bucket

2. Fetch the data from Google Storage Bucket and Successfully execute the IRIS Machine Learning Training Pipeline

3. Store the Output artifacts (Models, logs, etc) in Google cloud storage bucket with folders organized by their training execution timestamp

4. Create a new script for inference and run the inference on eval set after fetching the models from GCS Output Artifacts Bucket

5. Run this Training and inference for 2 times resulting in two output artifact folders in Google cloud storage bucket

6. (Optional) Run this pipeline for two versions of data provided in github data folder

## Initial steps

#### Install Vertex AI SDK for Python and other required packages
```python
# Vertex SDK for Python
! pip3 install --upgrade --quiet  google-cloud-aiplatform
```

#### Set Google Cloud project information
```python
PROJECT_ID = "lively-nimbus-473407-m9"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
```

#### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.

```python
BUCKET_URI = f"gs://mlops-lively-nimbus-473407-m9"  # @param {type:"string"}
```
**If your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.
```python
! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}
```

#### Initialize Vertex AI SDK for Python
```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
```

## 1. Store Training Data in Google Storage Bucket

#### Fetch data from git repository
Data can not be saved directly to the cloud storage bucket from git repository, therefore we need an intermediate step i.e. to store the data locally
```python
! git clone --branch week_1 https://github.com/IITMBSMLOps/ga_resources.git
```

#### Save data to bucket
```python
! gsutil cp -r ga_resources/data/ {BUCKET_URI}/
```
## 2. IRIS Machine Learning Training Pipeline

### Import important libraries


```python
import os
import sys
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from zoneinfo import ZoneInfo
from datetime import datetime
```

### Fetch data from the bucket
```python
! gsutil cp -r {BUCKET_URI}/data/ .
```

### Import Dataset

Remember to **update the path of data csv**

```python
data = pd.read_csv('data/raw/iris.csv')
# data = pd.read_csv('data/v1/data.csv')
# data = pd.read_csv('data/v2/data.csv')
data.head(5)
```

### Train Test Split
```python
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species
```

### Eval Set
```python
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 42)
```

### Simple Decision Tree model
Build a Decision Tree model on iris data

```python
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
```

## 3. Store the Output artifacts

Store the Output artifacts (Models, logs, etc) in Google cloud storage bucket with folders organized by their training execution timestamp

### Path to your model artifacts

`MODEL_ARTIFACT_DIR` - Folder directory path to your model artifacts within a Cloud Storage bucket
```python
MODEL_ARTIFACT_DIR = f"iris_artifacts/{datetime.now(tz = ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M%S')}"
```

### Store the artifacts locally

```python
import pickle
import joblib

! mkdir -p artifacts

joblib.dump(mod_dt, "artifacts/model.joblib")
```

### Store the artifacts in Google Cloud Storage Bucket

Before you can deploy your model for serving, Vertex AI needs access to the following files in Cloud Storage:

- `model.joblib` (model artifact)
- `preprocessor.pkl` (model artifact)

Run the following commands to upload your files:
```python
# Store output artifacts to google cloud storage bucket
! gsutil cp artifacts/model.joblib {BUCKET_URI}/{MODEL_ARTIFACT_DIR}/
```

## 4. Create a new script for inference

Run the inference on eval set after fetching the models from GCS Output Artifacts Bucket

## 5. Run this Training and inference for 2 times resulting in two output artifact folders in Google cloud storage bucket
Run step 3 and 4 again

## 6. (Optional) Run this pipeline for two versions of data provided in github data folder
Change the data in step 2 and the further steps