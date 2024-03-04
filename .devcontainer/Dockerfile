# See https://containers.dev/guide/dockerfile

FROM mcr.microsoft.com/devcontainers/base:jammy

# Install additional packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3-numpy python3-dev python3-pip python3-wheel python3-venv flex \
        bison build-essential autoconf libtool git libboost-all-dev cmake \
        libhdf5-dev g++ git make openjdk-17-jdk
