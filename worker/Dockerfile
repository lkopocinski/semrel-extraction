FROM nvidia/cuda:10.0-runtime-ubuntu16.04

# standard ubuntu packages
RUN apt -y update && \
    apt install -y apt-utils && \
    apt install -y iputils-ping && \
    apt install -y iputils-tracepath && \
    apt install -y cmake make g++ && \
    apt install -y build-essential && \
    apt install -y git && \
    apt install -y libboost-all-dev && \
    apt install -y swig && \
    apt install -y wget && \
    apt install -y vim zip unzip &&\
    apt install -y locales locales-all && \
    apt install -y apt-transport-https


RUN locale-gen pl_PL.UTF-8
ENV LANG='pl_PL.UTF-8'
ENV LANGUAGE='pl_PL:pl'
ENV LC_ALL='pl_PL.UTF-8'

# install python and pip
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt -y update && \
    apt install -y python3.6 && \
    update-alternatives --install \
    /usr/bin/python python /usr/bin/python3.6 10 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip

COPY ./requirements.txt /
RUN pip install -r /requirements.txt

# install corpus2
RUN apt -y update && \
    apt -y install bison && \
    apt -y install flex && \
    apt -y install libloki-dev && \
    apt -y install libsfst1-1.4-dev && \
    apt -y install libxml++2.6-dev && \
    apt -y install swig

WORKDIR /home/install
RUN git clone http://nlp.pwr.wroc.pl/corpus2.git && \
    mkdir corpus2/bin && \
    cd corpus2/bin && \
    git checkout --track origin/python3.6 && \
    cmake -D CORPUS2_BUILD_POLIQARP:BOOL=True .. && \
    make -j && \
    make install && \
    ldconfig


## install NLPWorkers structure
RUN mkdir /samba
RUN mkdir /samba/requests
RUN mkdir /samba/requests/dir
RUN mkdir /samba/requests/div
RUN mkdir /home/work
RUN mkdir /home/work/models
RUN mkdir /home/work/nlpworkers

# cleaning
ADD clean.sh /etc/cron.hourly/clean.sh
RUN ["chmod", "+x", "/etc/cron.hourly/clean.sh"]

# download artifacts
COPY ./credentials /root/.aws/credentials
RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski-worker data/elmo
RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski-worker data/fasttext

COPY ./model.pt /model.pt
ENV PREDICTION_MODEL="/model.pt"
ENV ELMO_MODEL_OPTIONS="/elmo/options.json"
ENV ELMO_MODEL_WEIGHTS="/elmo/weights.hdf5"
ENV FASTTEXT_MODEL="/fasttext/kgr10.plain.skipgram.dim300.neg10.bin"


RUN git clone https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction.git && \
    cd semrel-extraction && \
    git checkout --track origin/lkopocinski-worker && \
    python3.6 setup.py install


#ENTRYPOINT ["/predict.sh", "~/115.files"]