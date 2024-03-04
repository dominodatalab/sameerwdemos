# Hyperparameter Search using Ray Tune & PyTorch Lightning

This repository holds an example script for tuning hyperparameters of a PyTorch Lightning model using Ray, in Domino.

The results are also logged to the Domino experiment manager using MLflow.

## Storage Setup

On-demand clusters in Domino are ephemeral. Any data that is stored on cluster local storage and not externally will be lost upon termination of the workload and the cluster.

As a result, the code in this repository logs outputs to an external data store that is mounted to the cluster. If you do not yet have a dataset or external data volume for logging outputs to, start by creating one in the project. Note down the mounted path to your data store and use it when following the instructions to run the code below.

## Environment Setup

The example requires two Domino environments to be built - one for the workspace and one for the cluster.

### Workspace Environment

Base image: `quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.3-domino5.7-standard`

Additional Dockerfile instructions:

```
RUN pip install torchvision==0.15.2 pytorch-lightning==2.1.0 datasets==2.12.0 transformers==4.33.1 torch==2.0.1 ray[all]==2.7.1 accelerate==0.23.0 streamlit==1.26.0 tblib==2.0.0 pandas==2.1.0 --user
```

Pluggable Workspace Tools:

```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  supportedFileExtensions: [ ".ipynb" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```

### Cluster Environment

Base image: `rayproject/ray-ml:2.7.1-py39`

Addditional Dockerfile instructions:

```
RUN pip install torchvision==0.15.2 pytorch-lightning==2.1.0 datasets==2.14.5 transformers==4.33.1 torch==2.0.1 accelerate==0.23.0 scikit-learn
USER root
RUN usermod -u 12574 ray
```

## Workspace Setup

Create a workspace with the following specs:

- `Workspace environment`: Use the workspace environment created above
- `Workspace hardware tier`: Use a medium sized hardware tier (~4 cores, 12 GB RAM). GPU is not required.
- `Ray cluster environment`: Use the cluster environment create above.
- `Ray worker hardware tier`: If using default arguments, ensure the hardware tier has at least 1 GPU and 4 cores.
- `Ray head hardware tier`: Use a medium sized hardware tier (~4 cores, 12 GB RAM). GPU is not required.

The other workspace settings can be configured to your own preference/requirements.

## Run the code

To run the sample, execute the following command in the terminal

```
python train.py --storage_path <PATH TO MOUNTED DATASET OR EXTERNAL DATA VOLUME>
```

You can also change some of the default parameters:

```
python train.py --storage_path <PATH TO MOUNTED DATASET OR EXTERNAL DATA VOLUME> --num_epochs <NUMBER OF EPOCHS> --num_trials <NUMBER OF RUNS TO TRY> --cpus_per_trial <NUMBER OF CORES TO USE PER TRIAL> --gpus_per_trial <NUMBER OF GPUS TO USE PER TRIAL>
```

When changing the number of resources per trial, ensure that the Ray workers has sufficient resources to handle the request.

Once the execution has started, you can use the Domino Experiment UI to monitor the results.


