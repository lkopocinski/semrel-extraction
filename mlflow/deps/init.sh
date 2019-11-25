#!/bin/bash

set -e

if [[ -z $FILE_DIR ]]; then
  echo >&2 "FILE_DIR must be set."
  exit 1
fi

if [[ -z $AWS_BUCKET ]]; then
  echo >&2 "AWS_BUCKET must be set."
  exit 1
fi

CREDENTIALS_FILE=/root/.aws/credentials
if [[ ! -f ${CREDENTIALS_FILE} ]]; then
  echo >&2 "Credentials file must be provided."
  exit 1
fi

mlflow server \
    --backend-store-uri file:/${FILE_DIR} \
    --default-artifact-root s3://${AWS_BUCKET} \
    --host 0.0.0.0 \
    --port $PORT
