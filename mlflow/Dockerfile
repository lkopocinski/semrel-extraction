FROM python:3.7
LABEL maintainer "Łukasz Kopociński <lkopoci@gmail.com>"

RUN mkdir /mlflow/
RUN pip install --upgrade pip && \
    pip install mlflow && \
    pip install boto3

EXPOSE 5001

ENV MLFLOW_S3_ENDPOINT_URL=https://minio.clarin-pl.eu

COPY ./credentials /root/.aws/credentials

CMD mlflow server -p 5001 \
    --backend-store-uri /mlflow \
    --host 0.0.0.0 \
    --default-artifact-root s3://semrel/
