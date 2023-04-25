# Base docker image
FROM python:3.10

### Define default arguments
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
COPY $dev_requirements /tmp/


### Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get install -y nano


### Install Python requirements
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu && \
    pip install -r /tmp/$requirements && \
    pip install -r /tmp/$dev_requirements


### Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


### Create working directory
ENV DATA_DIR $HOME_DIR
ENV APP_DIR $HOME_DIR/app
RUN mkdir -p $APP_DIR


### Change to non-root user and set working directory
USER $USER_NAME
WORKDIR $APP_DIR
