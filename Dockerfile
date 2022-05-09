# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace

# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM nvidia/cuda:10.2-base
CMD nvidia-smi
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:latest as pytorch
FROM python:3.8.7-slim-buster as builder


# Renew build args
ARG WORKSPACE

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio==1.35.0 grpcio-tools==1.35.0 protobuf==3.14.0

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. Occlusion_pose.proto


# -----------------------------
# Stage to generate final image
# -----------------------------

FROM python:3.8.7-slim-buster
# Renew build args


ARG WORKSPACE

ARG USER=runner
ARG GROUP=runner-group


# Create non-privileged user and workspace
	
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Install requirements
COPY requirements.txt .

RUN apt-get update && \
	apt-get install ffmpeg libsm6 libxext6  -y && \
	pip install --upgrade pip && \
    # Install headless version of opencv-python for server usage
    # Does not install graphical modules
    # See https://github.com/opencv/opencv-python#installation-and-usage
    pip install -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/Occlusion_pose.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} parsing.py ${WORKSPACE}/parsing.py
COPY --chown=${USER}:${GROUP} OcclusionPose_server.py ${WORKSPACE}/OcclusionPose_server.py
COPY --chown=${USER}:${GROUP} utils.py ${WORKSPACE}/utils.py
COPY --chown=${USER}:${GROUP} LantentNet.py ${WORKSPACE}/LantentNet.py
COPY --chown=${USER}:${GROUP} Latent_model_0,999.pkl ${WORKSPACE}/Latent_model_0,999.pkl


# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

CMD ["python", "OcclusionPose_server.py"]
