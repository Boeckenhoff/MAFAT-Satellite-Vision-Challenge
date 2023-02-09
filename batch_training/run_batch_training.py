import boto3

import argparse
import datetime


def parse_args():
    return {"job_queue": "train"}


def main():
    batch = boto3.client(
        service_name="batch",
        region_name="eu-central-1",
        )
    response = batch.submit_job(
        jobName="train1",
        jobQueue= "",
        jobDefinition="",
    )

if __name__=="__main__":
    args = parse_args()
    main()
