# Base docker image
FROM dolfinx/dolfinx:stable

### Define default arguments
# Are overwritten by build arguments (if supplied while building the container)
ARG uid=1000
ARG gid=1000
ARG requirements=requirements-docker.txt
ARG dev_requirements=requirements-docker-dev.txt


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


### Create directories
ENV DATA_DIR $HOME_DIR


### Copy files
COPY $requirements /tmp/
COPY $dev_requirements /tmp/


### Install base utilities
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y nano


### Install utilities
# PETSc
RUN apt-get install -y petsc-dev

# Python requirements
RUN pip install --upgrade pip && \
    pip install -r /tmp/$requirements && \
    pip install -r /tmp/$dev_requirements && \
    pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

### Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


### Change to non-root user 
USER $USER_NAME