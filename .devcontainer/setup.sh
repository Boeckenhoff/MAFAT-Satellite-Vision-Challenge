#!/bin/bash

pip3 install awscli
#aws --profile default configure set aws_access_key_id "${S3_ACESS_TOKEN}"
#aws --profile default configure set aws_secret_access_key "${S3_SECRET_KEY}"

pip3 install --user -r .devcontainer/requirements.txt