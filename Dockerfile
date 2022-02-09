# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:22.01-py3
ARG WANDB_SECRET
RUN apt-get update -y && apt-get install git -y
RUN pip install pandas numpy pkbar einops jupyterlab
RUN test -n "$WANDB_SECRET" # makes WANDB_SECRET mandatory for the build
RUN pip install --upgrade wandb && \
    wandb login $WANDB_SECRET

WORKDIR /root
RUN mkdir -p ./NapoleonZero
COPY ./datasets/datasets ./NapoleonZero/datasets
# Join dataset dumps
# (we have to split them because github won't allow files bigger than 2G)
RUN cat ./NapoleonZero/datasets/ccrl5M-depth1.npz.part* > \
      ./NapoleonZero/datasets/ccrl5M-depth1.npz && \
      rm ./NapoleonZero/datasets/ccrl5M-depth1.npz.part*

COPY ./src ./NapoleonZero/src
RUN chmod +x ./NapoleonZero/src/napoleonzero-torch.py

CMD ["python3", "./NapoleonZero/src/napoleonzero-torch.py"]
# WORKDIR /root/NapoleonZero/src
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

