from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
import nest_asyncio
import qdrant_client
import ollama
import base64
import os
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama

import unstructured
from qdrant_client import models
from qdrant_client import QdrantClient
from IPython.display import Markdown, display
import nest_asyncio
import shutil
nest_asyncio.apply()
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
# LLM configuration
def get_rewriting_llm(model_name: str = "gpt-4o", temperature: float = 0, max_tokens: int = 4000) -> ChatOpenAI:
    return ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)

# Prompt Template for Query Rewriting
def create_query_rewrite_prompt() -> PromptTemplate:
    template = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    Original query: {original_query}

    Rewritten query:
    """
    return PromptTemplate(input_variables=["original_query"], template=template)

# Build Query Rewriting Chain
def build_query_rewriting_chain(llm: ChatOpenAI) -> LLMChain:
    prompt_template = create_query_rewrite_prompt()
    return prompt_template | llm

# Function to Rewrite the Query
def rewrite_query(original_query: str, query_rewriter_chain: LLMChain) -> str:
    response = query_rewriter_chain.invoke(original_query)
    return response.content.strip()

def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def get_image_summary(file_path):

    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'Summarize the image in detailed:',
            'images': [file_path]
        }]
    )
    return response.message.content
def get_text_summary(text):

    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{
            'role': 'user',
            'content': f'Summarize this text {text}'
        }]
    )
    
    return response.message.content

def get_table_summary(table_html):

    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{
            'role': 'user',
            'content': f'Summarize this table: {table_html}'
        }]
    )
    
    return response.message.content


def process_chunks(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000
    )
    texts, tables, images = [], [], []
    for chunk in chunks:
        if isinstance(chunk, unstructured.documents.elements.Table):
            tables.append(chunk)
        if isinstance(chunk, unstructured.documents.elements.CompositeElement):
            texts.append(chunk)
            chunk_elements = chunk.metadata.orig_elements
            # iterate over all elements of this chunk
            for element in chunk_elements:
                if isinstance(element, unstructured.documents.elements.Image):
                    images.append(element.metadata.image_base64)
    print(len(images))
    if os.path.exists("images"):
        # Remove the existing folder and its contents
        shutil.rmtree("images")
    
    # Create a new empty folder
    os.makedirs("images")
    for idx, image in enumerate(images):
        image_data = base64.b64decode(image)
        path = f"images/image_{idx}.jpeg"
        with open(path, "wb") as f:
            f.write(image_data)
            print('saved')
    image_summaries = [get_image_summary(f"images/image_{i}.jpeg") \
                for i in tqdm(range(len(images)))]
    text_summaries = [get_text_summary(texts[i].text) \
                for i in tqdm(range(len(texts)))]
    table_summaries = [get_table_summary(tables[i].metadata.text_as_html) \
                for i in tqdm(range(len(tables)))]
    return text_summaries + image_summaries+ table_summaries


class EmbedData:

    def __init__(self, 
                 embed_model_name="nomic-ai/nomic-embed-text-v1.5",
                 batch_size=32):
        
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []
    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name,
                                           trust_remote_code=True,
                                           cache_folder='./hf_cache')
        return embed_model
    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)
    def embed(self, contexts):
        self.contexts = contexts
        
        for batch_context in tqdm(batch_iterate(contexts, self.batch_size),
                                  total=len(contexts)//self.batch_size,
                                  desc="Embedding data in batches"):
                                  
            batch_embeddings = self.generate_embedding(batch_context)
            
            self.embeddings.extend(batch_embeddings)


class QdrantVDB:

    def __init__(self, collection_name, vector_dim=768, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
    def define_client(self):
        self.client = QdrantClient(url="http://localhost:6333",
                                   prefer_grpc=True)
    def create_collection(self):
        
        if not self.client.collection_exists(collection_name=self.collection_name):

            self.client.create_collection(collection_name=self.collection_name,
                                          
                                          vectors_config=models.VectorParams(
                                                              size=self.vector_dim,
                                                              distance=models.Distance.DOT,
                                                              on_disk=True),
                                          
                                          optimizers_config=models.OptimizersConfigDiff(
                                                                            default_segment_number=5,
                                                                            indexing_threshold=0)
                                         )
    def ingest_data(self, embeddata):
    
        for batch_context, batch_embeddings in tqdm(zip(batch_iterate(embeddata.contexts, self.batch_size), 
                                                        batch_iterate(embeddata.embeddings, self.batch_size)), 
                                                    total=len(embeddata.contexts)//self.batch_size, 
                                                    desc="Ingesting in batches"):
        
            self.client.upload_collection(collection_name=self.collection_name,
                                        vectors=batch_embeddings,
                                        payload=[{"context": context} for context in batch_context])

        self.client.update_collection(collection_name=self.collection_name,
                                    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
                                    )

class Retriever:

    def __init__(self, vector_db, embeddata):
        
        self.vector_db = vector_db
        self.embeddata = embeddata
    def search(self, query):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
            
        # Start the timer
        start_time = time.time()
        
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,       
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=1000,
        )
        
        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Execution time for the search: {elapsed_time:.4f} seconds")

        return result
    
class RAG:
    def __init__(self,
                 retriever,
                 llm_name="llama3.2:1b"):
        
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.qa_prompt_tmpl_str = """Context information is below.
                                     ---------------------
                                     {context}
                                     ---------------------
                                     
                                     Given the context information above I want you
                                     to think step by step to answer the query in a
                                     crisp manner, incase case you don't know the
                                     answer say 'I don't know!'
                                     
                                     ---------------------
                                     Query: {query}
                                     ---------------------
                                     Answer: """
    def _setup_llm(self):
        return Ollama(model=self.llm_name)
    def generate_context(self, query):
    
        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context = entry["payload"]["context"]

            combined_prompt.append(context)

        return "\n\n---\n\n".join(combined_prompt)
    def query(self, query):
        context = self.generate_context(query=query)
        
        prompt = self.qa_prompt_tmpl_str.format(context=context,
                                                query=query)
        
        response = self.llm.complete(prompt)
        
        # return dict(response)['text']
        return {
        "response": dict(response)['text'],
        "context": context
        
        }