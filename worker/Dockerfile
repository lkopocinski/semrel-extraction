FROM nvidia/cuda:10.0-runtime-ubuntu16.04

# standard ubuntu packages
RUN apt-get -y update && \
    apt-get install -y apt-utils && \
    apt-get install -y iputils-ping && \
    apt-get install -y iputils-tracepath && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get install -y git && \
    apt-get install -y libboost-all-dev && \
    apt-get install -y swig && \
    apt-get install -y python-dev && \
    apt-get install -y wget && \
    apt-get install -y software-properties-common python-software-properties &&\
    apt-get install -y vim mc zip unzip &&\
    apt-get install -y locales locales-all && \
    apt-get install -y apt-transport-https


RUN locale-gen pl_PL.UTF-8
ENV LANG='pl_PL.UTF-8' LANGUAGE='pl_PL:pl' LC_ALL='pl_PL.UTF-8'


RUN apt update && apt install -y  --no-install-recommends \
    bison \
    flex \
    htop \
    libantlr-dev \
    libboost-all-dev \
    libedit-dev \
    libicu-dev \
    libloki-dev \
    libreadline-dev \
    libsfst1-1.4-dev \
    libxml++2.6-dev \
    pkg-config \
    python3.6 \
    python3.6-dev \
    python3.6-venv \
    ranger


# install NLPWorkers structure
RUN mkdir /samba
RUN mkdir /samba/requests
RUN mkdir /samba/requests/dir
RUN mkdir /samba/requests/div
RUN mkdir /home/work
RUN mkdir /home/work/models
RUN mkdir /home/work/nlpworkers

# python libs for workers
RUN apt-get -y update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt

# corpus2
WORKDIR /home/install
RUN git clone http://nlp.pwr.wroc.pl/corpus2.git && \
    mkdir corpus2/bin && \
    cd corpus2/bin && \
    git checkout --track origin/python3.6 && \
    cmake -D CORPUS2_BUILD_POLIQARP:BOOL=True .. && \
    make -j && \
    make install && \
    ldconfig

# cleaning
ADD clean.sh /etc/cron.hourly/clean.sh
RUN ["chmod", "+x", "/etc/cron.hourly/clean.sh"]


# TODO:
COPY ./credentials /root/.aws/credentials

RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski-refactoring /data/elmo
RUN dvc get https://gitlab.clarin-pl.eu/team-semantics/semrel-extraction --rev lkopocinski-refactoring /data/fasttext

ENV PREDICTION_MODEL_PATH="/model.pt"
ENV ELMO_MODEL_PATH="/data/elmo"
ENV FASTTEXT_MODEL_PATH="/data/fasttext"


# Download repository and install it


ENTRYPOINT ["/predict.sh", "~/115.files"]