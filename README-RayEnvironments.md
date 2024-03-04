## Ray Workspace

**BASE** 
Use the standard Domino DAD


**DockerFile Instructions**

```
ENV RAY_VERSION=2.9.0
RUN sudo apt-get update
RUN sudo apt-get install -y cmake
RUN pip install ray[all]==$RAY_VERSION
RUN pip install torchvision==0.15.2 pytorch-lightning==2.1.0 datasets==2.12.0 transformers==4.33.1 torch==2.0.1 ray[all]==2.9.0 accelerate==0.23.0 streamlit==1.26.0 tblib==2.0.0 pandas==2.1.0 
RUN pip install torchmetrics
RUN pip install pytorch-lightning-bolts
RUN pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
RUN pip install torchvision
RUN pip install GPUtil
RUN pip install torch torchvision tensortrade
RUN pip install tblib
RUN pip install tensorflow
RUN pip install keras
RUN pip install nvidia-smi
USER ubuntu
```

## Ray Cluster Definition


**BASE** 
```
rayproject/ray-ml:2.9.0-py39
```

**DockerFile Instructions**

USER root
```
RUN pip install tensorflow 
RUN pip install 'jupyter-ai>=1.0,<2.0' 
RUN pip install torchvision==0.15.2 pytorch-lightning==2.1.0 datasets==2.12.0 transformers==4.33.1 torch==2.0.1 ray[all]==2.9.0 accelerate==0.23.0 streamlit==1.26.0 tblib==2.0.0 pandas==2.1.0 
RUN pip install torchmetrics
RUN pip install pytorch-lightning-bolts
RUN pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
RUN pip install torchvision
RUN pip install GPUtil
RUN pip install torch torchvision tensortrade
RUN pip install tblib
RUN pip install tensorflow
RUN pip install keras
RUN pip install nvidia-smi

RUN usermod -u 12574 ray
```
