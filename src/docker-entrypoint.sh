#!/bin/bash

echo "Container is running!!!"


gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
echo "passed"
mkdir -p /mnt/gcs_bucket
gcsfuse --key-file=$GOOGLE_APPLICATION_CREDENTIALS $GCS_BUCKET_NAME /cliniq-testing-harry
echo 'GCS bucket mounted at /cliniq-testing-harry'
mkdir -p /app/cheese_dataset
mount --bind /cliniq-testing-harry /app/cheese_dataset

pipenv shell