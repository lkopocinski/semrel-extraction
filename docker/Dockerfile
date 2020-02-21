FROM nvidia/cuda:10.0-runtime-ubuntu16.04
LABEL maintainer="Łukasz Kopocinski <lkopocinski@gmail.com>"

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa


RUN apt update && apt install -y \
    bison \
    build-essential \
    cmake \
    curl \
    flex \
    git \
    g++ \
    htop \
    libantlr-dev \
    libboost-all-dev \
    libedit-dev \
    libicu-dev \
    libloki-dev \
    libreadline-dev \
    libsfst1-1.4-dev \
    libxml++2.6-dev \
    locales \
    locales-all \
    pkg-config \
    python3.6 \
    python3.6-dev \
    python3.6-venv \
    ranger \
    subversion \
    swig \
    wget \
    vim

RUN update-alternatives --install \
    /usr/bin/python python /usr/bin/python3.6 10 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip

# Morfeusz
RUN wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key|sudo apt-key add - && \
    apt-add-repository http://download.sgjp.pl/apt/ubuntu && apt update && \
    apt install -y *morfeusz2*

WORKDIR /home/install
RUN mkdir morfeusz/ && cd morfeusz/ && \
    wget http://tools.clarin-pl.eu/share/morfeusz-SGJP-linux64-20130413.tar.bz2 && \
    tar -jxvf morfeusz-SGJP-linux64-20130413.tar.bz2 && \
    mv libmorfeusz* /usr/local/lib/ && \
    mv morfeusz /usr/local/bin/ && \
    mv morfeusz.h /usr/local/include/ && \
    ldconfig && \
    cd /home/install && \
    rm -rf morfeusz-sgjp

# Corpus2
WORKDIR /home/install
RUN git clone http://nlp.pwr.edu.pl/corpus2.git && \
    cd corpus2/ && \
    git checkout --track origin/python3.6 && \
    mkdir bin && \
    cd bin/ && \
    cmake -D CORPUS2_BUILD_POLIQARP:BOOL=True .. && \
    make -j 4 && \
    make install && \
    ldconfig && \
    cd /home/install && \
    rm -rf corpus2

# Toki
WORKDIR /home/install
RUN git clone http://nlp.pwr.edu.pl/toki.git && \
    cd toki/ && \
    git checkout --track origin/lkopocinski-cmake-cpp11-support && \
    mkdir bin && \
    cd bin/ && \
    cmake .. && \
    make -j 4 && \
    make install && \
    ldconfig && \
    cd /home/install && \
    rm -rf toki

# Maca
WORKDIR /home/install
RUN git clone https://gitlab.clarin-pl.eu/analysers/maca.git && \
    cd maca && \
    mkdir bin && \
    cd bin && \
    cmake .. && \
    make -j 4 && \
    make install && \
    ldconfig && \
    cd /home/install && \
    rm -rf maca

# Wccl
WORKDIR /home/install
RUN git clone http://nlp.pwr.edu.pl/wccl.git && \
    cd wccl/ && \
    git checkout --track origin/lkopocinski-add-cmake-cpp11-support && \
    mkdir bin && \
    cd bin && \
    cmake .. && \
    make -j 4 && \
    make install && \
    ldconfig && \
    cd /home/install && \
    rm -rf wccl


RUN locale-gen en_US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV MLFLOW_TRACKING_URI "http://10.17.50.132:8080"
ENV PIP_EXTRA_INDEX_URL "https://pypi.clarin-pl.eu/"

COPY requirements.txt /
RUN pip install -r /requirements.txt


# sent2vec
WORKDIR /home/install
RUN git clone https://github.com/epfml/sent2vec.git && \
    cd sent2vec/ && \
    pip install . && \
    cd /home/install && \
    rm -rf sent2vec


WORKDIR /
ENTRYPOINT ["/bin/bash"]