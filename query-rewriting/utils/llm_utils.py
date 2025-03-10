# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# credit to langchain for the original base implementation of splitting
# functionality
# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py

import logging
import os

from elasticsearch import Elasticsearch
from zenml.client import Client

from utils.openai_utils import get_openai_api_key

# Configure logging levels for specific modules
logging.getLogger("pytorch").setLevel(logging.CRITICAL)
logging.getLogger("sentence-transformers").setLevel(logging.CRITICAL)
logging.getLogger("rerankers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# Configure the logging level for the root logger
logging.getLogger().setLevel(logging.ERROR)

import re
from typing import List, Tuple

import litellm
import numpy as np
import psycopg2
import tiktoken
from constants import (
    EMBEDDINGS_MODEL,
    MODEL_NAME_MAP,
    OPENAI_MODEL,
    SECRET_NAME,
    SECRET_NAME_ELASTICSEARCH,
    ZENML_CHATBOT_MODEL_NAME,
    ZENML_CHATBOT_MODEL_VERSION,
)
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from rerankers import Reranker
from sentence_transformers import SentenceTransformer
from structures import Document

logger = logging.getLogger(__name__)


def split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """Splits a given text using a specified separator.

    This function splits the input text using the provided separator. The separator can be included or excluded
    from the resulting splits based on the value of keep_separator.

    Args:
        text (str): The text to be split.
        separator (str): The separator to use for splitting the text.
        keep_separator (bool): If True, the separator is kept in the resulting splits. If False, the separator is removed.

    Returns:
        List[str]: A list of strings resulting from splitting the input text.
    """
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [
                _splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)
            ]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


def split_text(
    document: Document,
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[Document]:
    """Splits a given text into chunks of specified size with optional overlap.

    Args:
        document (Document): The document to be split.
        separator (str, optional): The separator to use for splitting the text. Defaults to "\n\n".
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 4000.
        chunk_overlap (int, optional): The size of the overlap between consecutive chunks. Defaults to 200.
        keep_separator (bool, optional): If True, the separator is kept in the resulting splits. If False, the separator is removed. Defaults to False.
        strip_whitespace (bool, optional): If True, leading and trailing whitespace is removed from each split. Defaults to True.

    Raises:
        ValueError: If chunk_overlap is larger than chunk_size.

    Returns:
        List[Document]: A list of documents resulting from splitting the input document into chunks.
    """
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )

    separator_regex = re.escape(separator)
    splits = split_text_with_regex(
        document.page_content, separator_regex, keep_separator
    )
    _separator = "" if keep_separator else separator

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = ""

    for split in splits:
        if strip_whitespace:
            split = split.strip()

        if len(current_chunk) + len(split) + len(_separator) <= chunk_size:
            current_chunk += split + _separator
        else:
            if current_chunk:
                token_count = len(
                    encoding.encode(current_chunk.rstrip(_separator))
                )
                chunks.append(
                    Document(
                        page_content=current_chunk.rstrip(_separator),
                        filename=document.filename,
                        parent_section=document.parent_section,
                        url=document.url,
                        token_count=token_count,
                    )
                )
            current_chunk = split + _separator

    if current_chunk:
        token_count = len(encoding.encode(current_chunk.rstrip(_separator)))
        chunks.append(
            Document(
                page_content=current_chunk.rstrip(_separator),
                filename=document.filename,
                parent_section=document.parent_section,
                url=document.url,
                token_count=token_count,
            )
        )

    final_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            overlap = chunks[i - 1].page_content[-chunk_overlap:]
            token_count = len(
                encoding.encode(overlap + chunks[i].page_content)
            )
            final_chunks.append(
                Document(
                    page_content=overlap + chunks[i].page_content,
                    filename=document.filename,
                    parent_section=document.parent_section,
                    url=document.url,
                    token_count=token_count,
                )
            )

    return final_chunks


def split_documents(
    documents: List[Document],
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    keep_separator: bool = False,
    strip_whitespace: bool = True,
) -> List[Document]:
    """Splits a list of documents into chunks.

    Args:
        documents (List[str]): The list of documents to be split.
        separator (str, optional): The separator to use for splitting the documents. Defaults to "\n\n".
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 4000.
        chunk_overlap (int, optional): The size of the overlap between consecutive chunks. Defaults to 200.
        keep_separator (bool, optional): If True, the separator is kept in the resulting splits. If False, the separator is removed. Defaults to False.
        strip_whitespace (bool, optional): If True, leading and trailing whitespace is removed from each split. Defaults to True.

    Returns:
        List[str]: A list of chunked documents.
    """
    chunked_documents = []
    for doc in documents:
        chunked_documents.extend(
            split_text(
                doc,
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=keep_separator,
                strip_whitespace=strip_whitespace,
            )
        )
    return chunked_documents


def get_es_client() -> Elasticsearch:
    """Get an Elasticsearch client.

    Returns:
        Elasticsearch: An Elasticsearch client.
    """
    client = Client()
    es_host = client.get_secret(SECRET_NAME_ELASTICSEARCH).secret_values[
        "elasticsearch_host"
    ]
    es_api_key = client.get_secret(SECRET_NAME_ELASTICSEARCH).secret_values[
        "elasticsearch_api_key"
    ]

    es = Elasticsearch(
        es_host,
        api_key=es_api_key,
    )
    return es


def get_db_conn() -> connection:
    """Establishes and returns a connection to the PostgreSQL database.

    This function retrieves the password for the PostgreSQL database from a secret store,
    then uses it along with other connection details to establish a connection.

    Returns:
        connection: A psycopg2 connection object to the PostgreSQL database.
    """
    client = Client()
    try:
        secret = client.get_secret(SECRET_NAME)
        logger.debug(f"Secret keys: {list(secret.secret_values.keys())}")

        CONNECTION_DETAILS = {
            "user": os.getenv("SUPABASE_USER")
            or secret.secret_values["supabase_user"],
            "password": os.getenv("SUPABASE_PASSWORD")
            or secret.secret_values["supabase_password"],
            "host": os.getenv("SUPABASE_HOST")
            or secret.secret_values["supabase_host"],
            "port": os.getenv("SUPABASE_PORT")
            or secret.secret_values["supabase_port"],
            "dbname": "postgres",
        }
        return psycopg2.connect(**CONNECTION_DETAILS)
    except KeyError as e:
        logger.error(f"Missing key in secret: {e}")
        raise


def get_topn_similar_docs_pgvector(
    query_embedding: List[float],
    conn: psycopg2.extensions.connection,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the PostgreSQL database.

    Args:
        query_embedding (list): The query embedding to compare against.
        conn (psycopg2.extensions.connection): The database connection object.
        n (int, optional): The number of similar documents to fetch. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.
        only_urls (bool, optional): Whether to only return URLs in the results. Defaults to False.
    """
    embedding_array = np.array(query_embedding)
    register_vector(conn)
    cur = conn.cursor()

    if include_metadata:
        cur.execute(
            f"SELECT content, url, parent_section FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )
    elif only_urls:
        cur.execute(
            f"SELECT url FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )
    else:
        cur.execute(
            f"SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT {n}",
            (embedding_array,),
        )

    return cur.fetchall()


def get_topn_similar_docs_elasticsearch(
    query_embedding: List[float],
    es_client: Elasticsearch,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the Elasticsearch index.

    Args:
        query_embedding (list): The query embedding to compare against.
        es_client (Elasticsearch): The Elasticsearch client.
        n (int, optional): The number of similar documents to fetch. Defaults to 5.
        include_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.
        only_urls (bool, optional): Whether to only return URLs in the results. Defaults to False.
    """
    index_name = "zenml_docs"

    if only_urls:
        source = ["url"]
    elif include_metadata:
        source = ["content", "url", "parent_section"]
    else:
        source = ["content"]

    query = {
        "_source": source,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding},
                },
            }
        },
        "size": n,
    }

    # response = es_client.search(index=index_name, body=query)
    response = es_client.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "num_candidates": 50,
            "k": n,
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        if only_urls:
            results.append((hit["_source"]["url"],))
        elif include_metadata:
            results.append(
                (
                    hit["_source"]["content"],
                    hit["_source"]["url"],
                    hit["_source"]["parent_section"],
                )
            )
        else:
            results.append((hit["_source"]["content"],))

    return results


def get_topn_similar_docs(
    query_embedding: List[float],
    conn: psycopg2.extensions.connection = None,
    es_client: Elasticsearch = None,
    n: int = 5,
    include_metadata: bool = False,
    only_urls: bool = False,
) -> List[Tuple]:
    """Fetches the top n most similar documents to the given query embedding from the database.

    Args:
        query_embedding (list): The query embedding to compare against.
        conn (psycopg2.extensions.connection): The database connection object.
        n (int, optional): The number of similar documents to fetch. Defaults to
        5.
        include_metadata (bool, optional): Whether to include metadata in the
        results. Defaults to False.

    Returns:
        list: A list of tuples containing the content and metadata (if include_metadata is True) of the top n most similar documents.
    """
    if conn is None and es_client is None:
        raise ValueError("Either conn or es_client must be provided")

    if conn is not None:
        return get_topn_similar_docs_pgvector(
            query_embedding, conn, n, include_metadata, only_urls
        )

    if es_client is not None:
        return get_topn_similar_docs_elasticsearch(
            query_embedding, es_client, n, include_metadata, only_urls
        )


def get_completion_from_messages(
    messages, model=OPENAI_MODEL, temperature=0, max_tokens=1000
):
    """Generates a completion response from the given messages using the specified model.

    Args:
        messages (list): The list of messages to generate a completion from.
        model (str, optional): The model to use for generating the completion. Defaults to OPENAI_MODEL.
        temperature (float, optional): The temperature to use for the completion. Defaults to 0.4.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.

    Returns:
        str: The content of the completion response.
    """
    model = MODEL_NAME_MAP.get(model, model)
    completion_response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=get_openai_api_key(),
    )
    return completion_response.choices[0].message.content


def get_embeddings(text):
    """Generates embeddings for the given text using a SentenceTransformer model.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        np.ndarray: The generated embeddings.
    """
    model = SentenceTransformer(EMBEDDINGS_MODEL)
    return model.encode(text)


def find_vectorstore_name() -> str:
    """Finds the name of the vector store used for the given embeddings model."""
    from zenml.client import Client

    client = Client()
    try:
        model_version = client.get_model_version(
            model_name_or_id=ZENML_CHATBOT_MODEL_NAME,
            model_version_name_or_number_or_id=ZENML_CHATBOT_MODEL_VERSION,
        )
        return model_version.run_metadata["vector_store"]["name"]
    except KeyError:
        logger.error("Vector store metadata not found in model version")
        return "pgvector"  # Fallback to default


def rerank_documents(
    query: str, documents: List[Tuple], reranker_model: str = "flashrank"
) -> List[Tuple[str, str]]:
    """Reranks the given documents based on the given query.

    Args:
        query (str): The query to use for reranking.
        documents (List[Tuple]): The documents to rerank.
        reranker_model (str, optional): The reranker model to use.
            Defaults to "flashrank".

    Returns:
        List[Tuple[str, str]]: A list of tuples containing
            the reranked documents and their URLs.
    """
    ranker = Reranker(reranker_model)
    docs_texts = [f"{doc[0]} PARENT SECTION: {doc[2]}" for doc in documents]
    results = ranker.rank(query=query, docs=docs_texts)
    # pair the texts with the original urls in `documents`
    # `documents` is a tuple of (content, url)
    # we want the urls to be returned
    reranked_documents_and_urls = []
    for result in results.results:
        # content is a `rerankers` Result object
        index_val = result.doc_id
        doc_text = result.text
        doc_url = documents[index_val][1]
        reranked_documents_and_urls.append((doc_text, doc_url))
    return reranked_documents_and_urls


def process_input_with_retrieval(
    input: str,
    model: str = OPENAI_MODEL,
    n_items_retrieved: int = 20,
    use_reranking: bool = False,
) -> str:
    """Process the input with retrieval.

    Args:
        input (str): The input to process.
        model (str, optional): The model to use for completion. Defaults to
            OPENAI_MODEL.
        n_items_retrieved (int, optional): The number of items to retrieve from
            the database. Defaults to 5.
        use_reranking (bool, optional): Whether to use reranking. Defaults to
            False.

    Returns:
        str: The processed output.
    """
    delimiter = "```"
    es_client = None
    conn = None

    vector_store_name = find_vectorstore_name()
    if vector_store_name == "pgvector":
        conn = get_db_conn()
    else:
        es_client = get_es_client()

    # Step 1: Get documents related to the user input from database
    related_docs = get_topn_similar_docs(
        get_embeddings(input),
        conn=conn,
        es_client=es_client,
        n=n_items_retrieved,
        include_metadata=use_reranking,
    )

    if use_reranking:
        # Rerank the documents based on the input
        # and take the top 5 only
        context_content = [
            doc[0] for doc in rerank_documents(input, related_docs)[:5]
        ]
    else:
        context_content = [doc[0] for doc in related_docs[:5]]

    # Step 2: Get completion from OpenAI API
    # Set system message to help set appropriate tone and context for model
    system_message = f"""
    You are a friendly chatbot. \
    You can answer questions about ZenML, its features and its use cases. \
    You respond in a concise, technically credible tone. \
    You ONLY use the context from the ZenML documentation to provide relevant
    answers. \
    You do not make up answers or provide opinions that you don't have
    information to support. \
    If you are unsure or don't know, just say so. \
    """

    # Prepare messages to pass to model
    # We use a delimiter to help the model understand the where the user_input
    # starts and ends

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{input}{delimiter}"},
        {
            "role": "assistant",
            "content": f"Please use the input query and the following relevant ZenML documentation (in order of usefulness for this query) to answer the user query: \n"
            + "\n".join(context_content),
        },
    ]
    logger.debug("CONTEXT USED\n\n", messages[2]["content"], "\n\n")
    return get_completion_from_messages(messages, model=model)

def generate_rewritten_queries(
    input_query: str,
    model: str = OPENAI_MODEL,
    num_queries: int = 5,
) -> List[str]:
    """Generate multiple rewritten queries for a given input query using an LLM.

    Args:
        input_query (str): The original query to rewrite.
        model (str, optional): The model to use for query rewriting. Defaults to OPENAI_MODEL.
        num_queries (int, optional): Number of queries to generate. Defaults to 5.
        tracing_tags (List[str], optional): Tags for tracing. Defaults to [].

    Returns:
        List[str]: List of rewritten queries.
    """
    system_message = f"""
    You are an expert at reformulating search queries to improve document retrieval.
    Your task is to generate {num_queries} different versions of the input query.
    Each version should:
    1. Maintain the core intent of the original query
    2. Use different but semantically similar words/phrases
    3. Include relevant context that might help in document retrieval
    4. Be written as a complete, well-formed question
    5. Focus on different aspects of the original query

    Format your response as a Python list of strings, one query per line. The output should be
    exactly as shown in the example below. Don't include ` or python or anything else.
    Example:
    ["How do I configure authentication in ZenML?",
    "What are the steps to set up ZenML authentication?",
    "ZenML authentication setup guide and requirements",
    "Implementing security and auth in ZenML projects",
    "ZenML user authentication best practices and documentation"]
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_query},
    ]

    response = get_completion_from_messages(
        messages,
        model=model,
        temperature=0.7,  # Using higher temperature for more variety
    )

    # Parse the response into a list of queries
    # The response should be in a format that can be evaluated as a Python list
    try:
        queries = eval(response)
        if not isinstance(queries, list):
            raise ValueError("Response is not a list")
        return queries[:num_queries]  # Ensure we only return the requested number of queries
    except Exception as e:
        logger.error(f"Error parsing rewritten queries: {e}")
        # Fallback: return original query plus empty queries
        return [input_query] + [""] * (num_queries - 1)

def process_rewritten_queries(
    query: str,
    model: str = OPENAI_MODEL,
    n_items_per_query: int = 10,
) -> Tuple[List[Tuple[str, float]], str]:
    """Process multiple queries and combine their results for improved document retrieval.

    Args:
        query (str): Query to process.
        model (str, optional): Model to use for final response generation. Defaults to OPENAI_MODEL.
        n_items_per_query (int, optional): Number of items to retrieve per query. Defaults to 10.
        model_version_stage (str, optional): Stage of the model version. Defaults to "staging".
        tracing_tags (List[str], optional): Tags for tracing. Defaults to [].

    Returns:
        Tuple[List[Tuple[str, float]], str]: Tuple containing list of (document, score) pairs and final response.
    """
    all_docs = []

    queries = generate_rewritten_queries(query, model=model, num_queries=5)
    
    # Get the appropriate vector store client
    vector_store = find_vectorstore_name()
    if vector_store == "elasticsearch":
        client = get_es_client()
    else:  # pgvector
        client = get_db_conn()

    # Process each query
    for query in queries:
        if not query:  # Skip empty queries
            continue
            
        # Get embeddings for the query
        query_embedding = get_embeddings(query)
        
        # Get similar documents
        similar_docs = get_topn_similar_docs(
            query_embedding=query_embedding,
            es_client=client if vector_store == "elasticsearch" else None,
            conn=client if vector_store not in ["elasticsearch"] else None,  # Removed "pinecone"
            n=n_items_per_query,
            include_metadata=True,
        )
        
        all_docs.extend(similar_docs)

    # Close connection if using PostgreSQL
    if vector_store not in ["elasticsearch"]:  # Removed "pinecone"
        client.close()

    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc[0] not in seen:  # Using the document content as the key
            seen.add(doc[0])
            unique_docs.append(doc)

    # Rerank all unique documents
    ranker = Reranker("flashrank")
    docs_texts = [f"{doc[0]} PARENT SECTION: {doc[2]}" for doc in unique_docs]
    results = ranker.rank(query=queries[0], docs=docs_texts)  # Using original query for final ranking
    
    # Prepare final ranked documents
    reranked_docs = []
    for result in results.results:
        index_val = result.doc_id
        doc_text = result.text
        doc_score = result.score
        reranked_docs.append((doc_text, doc_score))

    # Generate final response using top reranked documents
    system_message = """
    You are a friendly chatbot that answers questions about ZenML.
    Use ONLY the provided context to answer the question.
    If you're unsure or don't have enough information, say so.
    Be concise and technically accurate.
    """

    context = "\n\n".join([doc[0] for doc in reranked_docs[:5]])  # Use top 5 documents
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": queries[0]},  # Using original query
        {
            "role": "assistant",
            "content": f"Here is the relevant ZenML documentation to answer your query:\n{context}",
        },
    ]

    final_response = get_completion_from_messages(
        messages,
        model=model,
    )

    return final_response
