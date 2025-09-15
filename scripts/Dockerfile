FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.11 python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends -qq cuda-nvcc-12-8

RUN mkdir /scripts
WORKDIR /scripts

COPY requirements.txt /scripts/requirements.txt
# RUN pip install --upgrade pip
RUN apt-get update && apt upgrade python3-pip -y --fix-missing
RUN pip install -U setuptools
RUN pip install -r requirements.txt
RUN pip install --force-reinstall --no-deps cryptography==3.4.8 --ignore-installed
# Default value for MODE is TRAINING
ENV MODE=TRAINING

# Number of epochs and mini batch size for each training step
ENV EPOCHS=1
ENV BATCH_SIZE=8

# Either standard or lstm for standard autoencoder or lstm autoencoder
ENV AUTOENCODER_TYPE=standard

# Default value for DETECTION_THRESHOLD is 0.4

ENV DETECTION_THRESHOLD=0.4

# Dropout rate for the autoencoder layers
ENV DROPOUT_RATE=0.2

# Lambda value for the L2 regularization
ENV L2_LAMBDA=0.001

# In case of standard autoencoder, the number of samples collected for one training epoch
# In case of LSTM autoencoder, the n umber of sequences collected for one training epoch
ENV BUFFER_LENGTH=32

ENV BUFFER_MODE=replace

ENV DISTANCE_METRIC=mse

# Only for lstm autoencoder: The length of one sequence
ENV SEQUENCE_LENGTH=10

# Precentage of validation data that should be collected
ENV VALIDATION_SPLIT=0.1

# In case of DETECTION mode, if model should be trained during detection phase (True or False)
ENV CONTINUE_TRAINING=False

ENV INCLUDE_ANOMALIES=True

ENV DETECTION_BUFFER_LENGTH=1

ENV DETECTION_BUFFER_MODE=replace

ENV REPLACEMENT_MODE=replace

ENV REPLACEMENT_VALUE=0

ENV INVALID_DATA_HANDLING=empty

# Default hostname
ENV HOSTNAME=0.0.0.0

# Port for websocket server

ENV PORT=8765

COPY . /scripts

EXPOSE 8765

CMD [ "python3", "-u", "/scripts/ws_server.py" ]