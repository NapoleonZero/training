# System Requirements
To run the training procedure and log an experiment you will need to have a [Weights & Biases](https://wandb.ai/site) access token.
Once you have one, place it under `~/.wandb_secret` or manually provide it to the build script by editing the `WANDB_SECRET` docker argument (TODO: pass it as a parameter).

Since the whole procedure is executed inside a docker, you will have to install it on your machine.

If you plan on training the model on a GPU, you will also have to install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) to enable GPU passthrough (this will only work with Nvidia gpus, there is currently no plan on supporting other type of accelerators).

# Usage
For starters, you will need to edit `src/napoleonzero-torch.py` and provide your own `project_name` and `entity` parameters to the `WandbCallback`object that is passed to the `TrainingLoop`. This will allow you to log the experiment on your own Weights & Biases project.

Further modifications might involve hyperparameters and dataset selection.

You can then build and run the docker container to start the experiment:
```
./build.sh
```
and
```
./run.sh
```
By default the run script tries to use any Nvidia gpu available. If you have none, please don't bother running it (TODO: will work on this).

Note that any modification to the code require you to build the container before running it again.
