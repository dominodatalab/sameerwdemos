conda create -n "pytorch" python=3.9.15
cd /opt/conda/envs/pytorch/bin
/opt/conda/envs/pytorch/bin/pip install ipykernel
/opt/conda/envs/pytorch/bin/python3 -m ipykernel install --user --name=pytorch
/opt/conda/envs/pytorch/bin/pip install -r /mnt/code/requirements-pytorch.txt
cd /mnt/code