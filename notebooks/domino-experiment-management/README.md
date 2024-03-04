# Domino Experiment Management

Domino experiment management leverages [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) to enable easy logging of experiment parameters, metrics, and artifacts, while providing a Domino-native user experience to help you analyze your results.
MLflow runs as a service in your Domino cluster, fully integrated within your workspace and jobs, and honoring role-based access control. Existing MLflow experiments works right out of the box with no code changes required.

This repository holds the code and intructions that can be used to demonstrate this new capability which is available in private preview in Domino 5.4 (public preview to be released in Domino 5.5).

## Quick execution

The instructions below are if you want to create a new environment. To
execute the code quickly run the following:

```
./create-pytorch-conda.sh 
/opt/conda/envs/pytorch/bin/python train.py
```


## Prerequisite 

ML Flow is included in the Domino Standard Environments, which means experiment tracking capabilities will work right out of the box. However, this example demonstrates how to use this capability with PyTorch Lightning so you will need to ensure that your environment has the PyTorch libriaries installed. If you do not already have an environment with PyTorch, follow [these instructions](#create-pytorch-environment) to create one.

## Run Demo

### Create experiment from workspace

Start by logging an experiment run from a workspace. 

1. Create a workspace using your PyTorch Lightning environment.

2. Clone this repo into the directory and run the `train.py` script.

```
git clone https://github.com/ddl-jwu/experiment-management
python experiment-management/train.py 
```

### Monitor and evaluate the results

1. Navigate to the `Experiments` page and select the `mnist` experiment

2. Click into the most recent run to monitor and evaluate the results that were just logged.

### Create experiment from a job

As your experiment matures, we recommend logging your experiments through jobs (rather than workspaces) to guarantee reproducibility:

1. Make sure your changes in the workspace are committed and start a new job from the Domino UI

2. In the `File Name or Command` section, run the training script that was cloned into the workspace earlier.

```
python experiment-management/train.py 
```

3. Make sure your PyTorch Lightning environment is selected.

4. Click `Start` to begin the job.

5. Follow the same steps as above to monitor and evaluate the results.

## Create PyTorch Environment

1. Navigate to the `Environments` page in the Domino UI

2. Click `Create Environment`.

3. Name your new environment.

4. In the Base Environment / Image section, select `Start from a custom base image`.

5. In the FROM line, enter `quay.io/domino/compute-environment-images:latest`.

6. Set the environment’s visbility.

7. Click `Customize Before Building`

8. In the `Dockerfile Fnstructions` section, add:

```
RUN pip install torchvision==0.14.1 torch==1.13.1 pytorch-lightning==1.9.0 protobuf==4.21.12 --user
```

9. In the `Pluggable Workspace Tools` section, add:

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
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```

10. Click `Build` to create environment.