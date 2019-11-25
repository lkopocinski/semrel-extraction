FROM python:3.7
LABEL maintainer "Łukasz Kopociński <lkopoci@gmail.com>"

RUN pip install --upgrade pip && \
    pip install mlflow==1.4.0 && \
    pip install boto3==1.10.26

ENV PORT 5000
ENV FILE_DIR mlruns
ENV AWS_BUCKET semrel
ENV MLFLOW_S3_ENDPOINT_URL https://minio.clarin-pl.eu

COPY deps/credentials /root/.aws/credentials
COPY deps/init.sh /

ENTRYPOINT ["/init.sh"]