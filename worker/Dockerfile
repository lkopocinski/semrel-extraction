FROM nvidia/cuda:10.0-runtime-ubuntu16.04

# install standard ubuntu packages
RUN apt update && apt install -y \
    apt-transport-https \
    apt-utils \
    build-essential \
    cmake \
    git \
    g++ \
    iputils-ping \
    iputils-tracepath \
    libboost-all-dev \
    locales \
    locales-all \
    make \
    swig \
    unzip \
    vim \
    wget \
    zip

# install dvc
RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
RUN apt update && apt install -y dvc

# download resources
WORKDIR /home/
COPY ./deps/credentials /root/.aws/credentials
RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski/code-clean-up semrel/data/data/elmo
RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski/code-clean-up semrel/data/data/fasttext

ENV ELMO_MODEL_OPTIONS="/home/elmo/options.json"
ENV ELMO_MODEL_WEIGHTS="/home/elmo/weights.hdf5"
ENV FASTTEXT_MODEL="/home/fasttext/kgr10.plain.skipgram.dim300.neg10.bin"


# setup locale
RUN locale-gen pl_PL.UTF-8
ENV LANG='pl_PL.UTF-8'
ENV LANGUAGE='pl_PL:pl'
ENV LC_ALL='pl_PL.UTF-8'


# install python and pip
RUN apt update && apt install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt update && apt install -y \
    python3.6 \
    python3.6-dev \
    python3.6-venv

RUN update-alternatives --install \
    /usr/bin/python python /usr/bin/python3.6 10 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip


# install corpus2
RUN apt update && apt install -y \
    bison \
    flex \
    libloki-dev \
    libsfst1-1.4-dev \
    libxml++2.6-dev

WORKDIR /home/install
RUN git clone https://gitlab.clarin-pl.eu/analysers/corpus2.git && \
    mkdir corpus2/bin && \
    cd corpus2/bin && \
    git checkout --track origin/python3.6 && \
    cmake -D CORPUS2_BUILD_POLIQARP:BOOL=True .. && \
    make -j 8 && \
    make install && \
    ldconfig


# install requirements
WORKDIR /home/
ENV PIP_EXTRA_INDEX_URL "https://pypi.clarin-pl.eu/"
COPY deps/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy model
COPY deps/model.pt model.pt
ENV PREDICTION_MODEL="/home/model.pt"


# install dependencies used in worker
RUN git clone https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction.git && \
    cd semrel-extraction && \
    git checkout --track origin/lkopocinski/code-clean-up && \
    python3.6 setup.py install
