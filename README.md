# got_extractiveqa

# Overview

Extractive question & answering based on Game of Thrones wikipedia articles. 
This code contains everything you need to build the docker images and run the python application locally. 

The goal is to create an application that leverages an NLP model, which was trained on game of thrones wikipedia articles, to return answers to Game of Thrones related questions. 

# Installation
Docker is required.

Simply clone the repo and run the following docker command:

`docker compose up`

# Usage

After running the container for the first time, please run the following command to update the data:
`http://localhost:8777/update_docustore`

Afterwards, you can ask any GoT question like so:

`http://localhost:8777/qna/"who is the father of Arya Stark"?`


# How the application works
1. The raw data

    The raw Game of Thrones data is comprised of 517 wikipedia articles that were converted to text files and stored in the following [S3 Bucket](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip). 

    In this 


2. 

