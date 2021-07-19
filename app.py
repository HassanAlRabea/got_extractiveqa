# import packages
import json
import logging
from flask_cors import CORS
from flask import Flask, request, jsonify
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

#application settings
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

    #converts txt files into a list of documents, with some cleaning done
    all_docs = convert_files_to_dicts(
        app.config["data"],
        clean_func=clean_wiki_text,
        split_paragraphs=False)

    #initialization of the Haystack Elasticsearch document storage
    document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index)
    #Write to document store
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
    #initialization of the Haystack Elasticsearch document storage
    document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index)
    
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    #initialization of ElasticRetriever
    retriever = ElasticsearchRetriever(document_store= document_store)
    # Finder sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    pipe = ExtractiveQAPipeline(reader, retriever)

    # predict n answers - ADD MORE INFOR HERE
    prediction = pipe.run(query=question, top_k_retriever=10, top_k_reader=5)
    answer = []
    # printed_answer = print_answers(prediction)
    for res in prediction['answers']:
        answer.append(res['answer'])

    return json.dumps({'status':'success','message': 'Process succesfully', 'result': answer})


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
