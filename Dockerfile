FROM ubuntu:18.04
LABEL maintainer "Łukasz Kopociński <lkopocinski@gmail.com>"

RUN apt update && apt install -y \
    bison \
    build-essential \
    cmake \
    curl \
    flex \
    git \
    g++ \
    libantlr-dev \
    libboost-all-dev \
    libedit-dev \
    libicu-dev \
    libloki-dev \
    libreadline-dev \
    libsfst1-1.4-dev \
    libxml++2.6-dev \
    python3.6 \
    python3.6-dev \
    python3.6-venv \
    software-properties-common \
    subversion \
    swig \
    wget

#RUN add-apt-repository ppa:deadsnakes/ppa

RUN cd /home/
RUN mkdir install && cd install

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.6 get-pip.py

RUN git clone http://nlp.pwr.edu.pl/corpus2.git && \
    git clone http://nlp.pwr.edu.pl/maca.git && \
    git clone http://nlp.pwr.edu.pl/toki.git && \
    git clone http://nlp.pwr.edu.pl/wccl.git && \
    git clone http://nlp.pwr.edu.pl/wcrft2.git

RUN cd corpus2/ && \
    #git apply /home/files/corpus2.diff && \
    git checkout --track origin/python3.6 && \
    mkdir bin && \
    cd bin/ && \
    cmake -D CORPUS2_BUILD_POLIQARP:BOOL=True .. && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN cd toki/ && \
    mkdir bin && \
    cd bin/ && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN cd maca && \
    #git apply /home/files/maca.diff && /
    mkdir bin && \
    cd bin && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN cd wccl/ && \
    mkdir bin && \
    cd bin && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN cd wcrft2/ && \
    #git apply /home/files/wcrft.diff && \
    mkdir bin && \
    cd bin/ && \
    cmake .. && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN mkdir crfpp && \
    cd crfpp && \
    wget http://tools.clarin-pl.eu/share/CRF++-0.58.tar.gz && \
    tar -xvzf CRF++-0.58.tar.gz && \
    cd CRF++-0.58 && \
    ./configure && \
    make -j && \
    make install && \
    ldconfig && \
    cd ../../

RUN wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key|sudo apt-key add - && \
    apt-add-repository http://download.sgjp.pl/apt/ubuntu && \
    apt update && apt install libmorfeusz2-dev && \
    dpkg -L libmorfeusz2-dev && \
    mkdir morfeusz-sgjp && \
    cd morfeusz-sgjp/ && \
    wget http://tools.clarin-pl.eu/share/morfeusz-SGJP-linux64-20130413.tar.bz2 && \
    tar -jxvf morfeusz-SGJP-linux64-20130413.tar.bz2 && \
    mv libmorfeusz* /usr/local/lib/ && \
    mv morfeusz /usr/local/bin/ && \
    mv morfeusz.h /usr/local/include/ && \
    ldconfig && \
    cd ..

#pip install dvc
#pip install mlflow

#mkdir -p home/semrel-extraction