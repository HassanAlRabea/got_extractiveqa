# got_extractiveqa

# Overview

Extractive question & answering based on Game of Thrones wikipedia articles - built using the Haystack NLP framework.

From their [github]("https://github.com/deepset-ai/haystack"):

*Haystack is an end-to-end framework that enables you to build powerful and production-ready pipelines for different search use cases. Whether you want to perform Question Answering or semantic document search, you can use the State-of-the-Art NLP models in Haystack to provide unique search experiences and allow your users to query in natural language.*

**This code contains everything you need to build the docker images and run the python application locally.**

The goal is to create an application that leverages an NLP model trained on game of thrones wikipedia articles, to return answers to Game of Thrones related questions. 

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
1. **Retrieve the raw data**

    The raw Game of Thrones data is comprised of 517 wikipedia articles that were converted to text files and stored in the following [S3 Bucket](https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip). 

    One the prepocessing tools available in Haystack is `fetch_archive_from_http`, which
    fetches an archive (zip or tar.gz) from a url via http and extracts the content to an output directory. In our case, it was used to store the data under the following application directory: `/usr/src/app/data/got_txts`.

2. **Setup the DocumentStore**

    Haystack finds answers to queries within the documents stored in a DocumentStore. It is basically a mini server that stores your raw data in a dictionary format which makes it easy for the model to retrieve the answers. When this app is built and run, it creates a new document store instance to allow for storage of the retrieved Wikipedia articles on Game of Thrones.
    
    The current implementations of the DocumentStore is the ElasticsearchDocumentStore. It leverages Elasticsearch and it comes preloaded with features like full-text queries, BM25 retrieval, and vector storage for text embeddings. 

    When a query is made, the Retrievers will operate on top of this DocumentStore to find the relevant documents for a query. The DocumentStore is initialized using the follow haystack method:
    ```
        document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index=index)
    ```



3. **Data Preprocessing**

    Before feeding it to the model, our raw data needs to be cleaned, transformed and stored. As mentioned in our previous section, the text files need to be converted into a list of dictionaries before writing them to the document store. The format would look something like this:
    ```
    {
        'text': "<DOCUMENT_TEXT_HERE>",
        'meta': {'name': "<DOCUMENT_NAME_HERE>", ...}
    }
    ```
    Thankfully, Haystack provides two very handy methods to do so. First, the `convert_files_to_dicts` converts all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a Document Store. Additionally, it can accept a cleaning function as an argument to clean the data before storing it. 
    
    In our case, we leverage the handy `clean_wiki_text`, which cleans wikipedia text articles by removing multiple new lines, removing extremely short lines, adding paragraph breaks and removing empty paragraphs. 
    ```
        all_docs = convert_files_to_dicts(app.config["data"], clean_func=clean_wiki_text, split_paragraphs=False)
    ```
    Once the texts have been processed, they are then ready to be fed to the document store:
    ```
    document_store.write_documents(all_docs)
    ```

4. **Initialize Retriever, Reader, & Pipeline**

    The next step is to define our retriever, reader and the pipeline that brings these two components together. 
    
    **Retriever**
    
    Retrievers are simple but fast algorithms that identify candidate documents for a given query from a large collection of documents. Retrievers narrow down the search space significantly and are therefore crucial for scalable Q&A syetms. Haystack supports sparse methods (TF-IDF, BM25, custom Elasticsearch queries) and state of the art dense methods (e.g., sentence-transformers and Dense Passage Retrieval). 

    For our use case, we leverage the `ElasticsearchRetriever` which utilizes the BM25 method. BM25 is a variant of TF-IDF that is recommend if you are looking for a retrieval method that does not need a neural network for indexing. It improves upon TF-IDF in two main aspects: 

    - It saturates tf after a set number of occurrences of the given term in the document
    - It normalizes by document length so that short documents are favoured over long documents if they have the same amount of word overlap with the query

    The retriever is defined like so, only needing to pass it the document_store. 
    ```
    retriever = ElasticsearchRetriever(document_store= document_store)
    ```

    **Reader**
    
    A Reader is a neural network (e.g., BERT or RoBERTA) that reads through texts returned by the retriever in detail to find an answer. The Reader takes multiple passages of texts as input and returns the top-k answers. Haystack currently supports readers based on the FARM or Transformers frameworks. With both you can either load a local model or one from Hugging Face's model hub.

    From their [github](https://github.com/deepset-ai/FARM), FARM is defined as:
    
    *FARM makes Transfer Learning with BERT & Co simple, fast and enterprise-ready. It's built upon transformers and provides additional features to simplify the life of developers: Parallelized preprocessing, highly modular design, multi-task learning, experiment tracking, easy debugging and close integration with AWS SageMake*

    In our application we leverage the FARM framework and chose the roberta-base-squad2 model. A model that has been trained on the Squad dataset and offers a good balance between speed & accuracy. We define it like so:

    ```
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    ```

    **Pipeline**

    With a Haystack Pipeline you can stick together the building blocks for a search pipeline. Under the hood, Pipelines are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases. To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the ExtractiveQAPipeline that combines a retriever and a reader to answer questions: 

    ```
    pipe = ExtractiveQAPipeline(reader, retriever)
    ```

5. **Run the pipeline**

    Finally, once the pipeline is defined, the pipe is run with each query the model receives and returns the top k answers (which can be modified according to the use case). You can control the number of results returned by both the retriever & reader:
    ```
    prediction = pipe.run(query=question, top_k_retriever=10, top_k_reader=5)
    ```
