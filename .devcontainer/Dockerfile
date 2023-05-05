# Base docker image
FROM --platform=linux/amd64 ubuntu:22.04

### Define (default) arguments
# Are overwritten by build arguments (if supplied while building the container)
ARG uid=1000
ARG gid=1000
ARG requirements=requirements.txt
ARG dev_requirements=dev-requirements.txt


### Change default shell
SHELL ["/bin/bash", "--login", "-c"]


### Create a non-root user
# For more information, see:
# - https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# - https://vsupalov.com/docker-shared-permissions/
ENV USER_NAME docker_user
ENV USER_UID $uid
ENV USER_GID $gid
ENV HOME_DIR /home/$USER_NAME
RUN groupadd --gid $USER_GID $USER_NAME && \
    adduser \
    --disabled-password \
    --gecos "non-root user" \
    --uid $USER_UID \
    --gid $USER_GID \
    --home $HOME_DIR \
    $USER_NAME


### Copy files
COPY $requirements /tmp/


### Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y nano


### Install miniconda
ENV CONDA_DIR /miniconda3
ENV MINICONDA_INSTALLER Miniconda3-latest-Linux-x86_64.sh
RUN wget --quiet https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm -f ~/miniconda.sh


### Configure miniconda
# # Put conda in path variable
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash -- login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bash_profile
# Make conda activate command available from /bin/bash -- interactive non-login shells
RUN conda init bash
# Automatically activate conda environment when starting a interactive non-login shell
RUN echo "conda activate venv" >> ~/.bashrc


### Create the virtual conda environment
RUN conda update --name base --channel defaults conda && \
    conda env create  --file /tmp/$requirements --force && \
    conda clean --all --yes


### Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


### Create working directory
ENV DATA_DIR /data
ENV APP_DIR $DATA_DIR/app
RUN mkdir -p $APP_DIR


### Change to non-root user and set working directory
USER $USER_NAME
WORKDIR $APP_DIR


### Entrypoint
ENTRYPOINT ["conda", "activate", "venv"]