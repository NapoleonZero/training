# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:22.01-py3
ARG WANDB_SECRET
RUN apt-get update -y && apt-get install git -y
RUN pip install pandas numpy pkbar
RUN test -n "$WANDB_SECRET" # makes WANDB_SECRET mandatory for the build
RUN pip install --upgrade wandb && \
    wandb login $WANDB_SECRET

WORKDIR /root
RUN mkdir -p ./NapoleonZero
COPY ./datasets/datasets ./NapoleonZero/datasets

# TODO: move up
RUN pip install einops

COPY ./src ./NapoleonZero/src
RUN chmod +x ./NapoleonZero/src/napoleonzero-torch.py

CMD ["python3", "./NapoleonZero/src/napoleonzero-torch.py"]
