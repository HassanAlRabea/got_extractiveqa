# import packages & requirements
import json
import logging
from flask_cors import CORS
from flask import Flask
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

# Application settings
app = Flask(__name__)
CORS(app)

# Application directory for inputs and training
app.config["data"] = "/usr/src/app/data/got_txts"
app.config["trained_models"] = "/usr/src/app/data/trained_models"

# ElasticSearch server host information
app.config["host"] = "elasticsearch"
app.config["username"] = ""
app.config["password"] = ""
app.config["port"] = "9200"


@app.route('/update_docustore')
def update_docustore():
    index = "document"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    
    # Fetches text files archived in a zip from an S3 bucket, unzips them and stores them in the output directory as text files.
    fetch_archive_from_http(url=s3_url, output_dir=app.config["data"])

    # Converts txt files in a directory into a list of documents, with some cleaning done by the cleaning function
    # The cleaning function used here will cleans wikipedia text articles by removing multiple new lines,
    # removing extremely short lines, adding paragraph breaks and removing empty paragraphs.
    all_docs = convert_files_to_dicts(
        app.config["data"],
        clean_func=clean_wiki_text,
        split_paragraphs=False)

    # Initialization of the Haystack Elasticsearch document storage to store the files
    # Retrievers operate on top of this DocumentStore to find the relevant documents for a query
    # Since we are not using a dense retriever, we don't have to worry about 
    # the embedding vector parameters and just use the default ones
    document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index)
    # Write to document store
    document_store.write_documents(all_docs)


    return json.dumps(
        {'status':'Susccess','message':
            'document available at http://'+ app.config["host"] +':'
            + app.config["port"] +'/' + index + '/_search',
            'result': []})


@app.route('/')
def home():
    """Return a friendly HTTP greeting."""
    return 'Hello, welcome to the Game of Thrones Extractive QA, please use the following format to obtain an answer: /qna/"My question goes here?"'


@app.route('/qna/<question>')
def qna(question):
    """Return the n answers."""
    index = "document"

    # Initialization of the Haystack Elasticsearch document storage
    document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index)
    
    # Initialization of the retriever in this case we will be using the
    # ElasticRetriever - a fast but simple retriever that leverages BM25 to narrow down our search
    # BM25 is a bag-of-words retrieval function that ranks a set of documents
    # based on the query terms appearing in each document, regardless of their proximity within the document.
    retriever = ElasticsearchRetriever(document_store= document_store)

    # Initialization of the reader. The Reader, also known as Open-Domain QA systems in Machine Learning speak,
    # is the core component that enables Haystack to find the answers we want.
    # In this case we are leveraging a pretrained model as training a deep learning model such as these
    # would require way too much resources for incremental gains. 
    # In the class of base sized models trained on SQuAD, RoBERTa has shown better performance than BERT and can be capably handled by any machine equipped with a single NVidia V100 GPU.
    # It is recommended for anyone wanting to create a performant and computationally reasonable instance of Haystack.
    # If speed was more important, we would have chosen the MiniLM model, and if we wanted a state of the art model
    # and had the necessary resources, we would have went with ALBERT XXL.
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)


    # Finder sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Predict answer from the question - return top 10 documents & from those, the top 5 answers
    prediction = pipe.run(query=question, top_k_retriever=10, top_k_reader=5)
    answer = []

    # Print answers
    for res in prediction['answers']:
        answer.append(res['answer'])

    return json.dumps({'status':'success','message': 'Process successfully', 'result': answer})


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return json.dumps({'status':'failed','message':
        """An internal error occurred: <pre>{}</pre>See logs for full stacktrace.""".format(e),
                       'result': []})

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='0.0.0.0', port=8777, debug=True)
