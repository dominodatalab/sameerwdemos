# Example demos for MLOps for regular and LLM Models

## Pre-requisites

Create two conda environments and the corresponding Jupyter Kernel

```shell
./create-ray-conda.sh  #Creates the "ray" kernel
./create-tensorflow-conda.sh #Creates the "tensorboard" kernel
```

There will be several notebooks in this repo to compare the features of these Experiment Tracking Products. 

## First the basic Experiment Management Flow

There are three basic notebooks included in this repo to demonstrate the use Experiment Manager for:
1. Running [basic](./notebooks/basic/mlflow-basic.ipynb) experiments - This notebook will demonstrate how to create basic experiments, runs and model versions. It will also demonstrate how to download artifacts for a specific Model Version via the experiment run id attached to the model version. This is useful to deploy models based on the artifacts contained in the model registry
2. Running [basic hyperparameter search using Spark](./notebooks/basic/pyspark_hyperparameter_search.ipynb)
3. Running [Ray Tune based hyperparameter search using ](./notebooks/basic/ray_tune_hyperparameter_search.ipynb). This demonstrates how you can create nested runs to better organize your experiment runs. **SELECT THE KERNEL- "ray"** to run this notebook
4. [Tensorflow Example](./notebooks/llm-dl/llm-dl/03_tensorboard_example.ipynb) - This notebook demonstrates the autologging feature of MLFLOW for tensorboard. The tensorboard logs are automatically stored in the MLFLOW artifacts. The notebook demonstrates how these logs can be downloaded for any experiment run for which they have been logged to render in a local tensorboard instance.**SELECT THE KERNEL- "tensorboard"** to run this notebook


### NOTE

To clear all notebooks before commit run the following command
```
jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb
```


### Environments

Use the following two environments-
1. [Ray Compute](./RayCompute-WORKSPACE.txt)
2. [Ray Cluster](./RayCluster.txt)
