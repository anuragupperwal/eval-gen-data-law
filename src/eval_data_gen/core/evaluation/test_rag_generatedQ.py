import os
import faiss
import gzip
import json
import time
import pandas as pd
import glob
import re

from tqdm import tqdm
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv
from llama_index.core import Settings
from langchain_ollama.embeddings import OllamaEmbeddings 
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from typing import List
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key
Settings.llm = Groq(model="llama-3.1-8b-instant") # "llama3-8b-8192")

# google_api_key = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = google_api_key
# Settings.llm = Gemini(model="models/gemini-2.5-flash")

# cohere_api_key = os.getenv("COHERE_API_KEY")
# os.environ["COHERE_API_KEY"] = cohere_api_key
# Settings.llm = Cohere(model="command-r-plus")

print(f"LLM configured as the default {Settings.llm}")

ollama_embed_model = OllamaEmbeddings(model="nomic-embed-text")
Settings.embed_model = ollama_embed_model 
print("Ollama embedding model configured as the default.")


def load_nodes_from_meta(meta_path: str) -> List[TextNode]:
    """Loads text and IDs from your metadata file to create TextNode objects."""
    nodes = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            meta = json.loads(line)
            # Create a TextNode for each document. The node's id_ must match its
            # position in the FAISS index
            node = TextNode(
                text=meta["text"],
                id_=str(meta["doc_id"])
            )
            nodes.append(node)
    return nodes

nodes = load_nodes_from_meta("tmp/meta.jsonl.gz")
faiss_index = faiss.read_index("tmp/faiss.index")
vector_store = FaissVectorStore(faiss_index=faiss_index)
# Create the VectorStoreIndex directly from nodes and the vector store
index = VectorStoreIndex(
    nodes=nodes,
    vector_store=vector_store,
)

query_engine = index.as_query_engine(similarity_top_k=4)

print(f"RAG query engine loaded with {len(nodes)} documents from existing index.")



def load_all_mcqs(questions_dir="tmp/questions"):
    """Loads all MCQ strings from the generated JSON files."""
    print(f"Loading previously generated MCQs from {questions_dir}...")
    mcq_strings = []
    # glob to find all JSON files
    for f_path in glob.glob(f"{questions_dir}/*.json"):
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            mcq_strings.extend(data.get("mcqs", []))
    return mcq_strings

def parse_question_stem(mcq_string: str) -> str:
    """Extracts just the question part from a full MCQ string."""
    # This regex captures everything from the start until the first numbered option `(1)`
    match = re.match(r"(.*?)\n\s*\-?\(1\)", mcq_string, re.DOTALL)
    if match:
        # Clean up the question stem
        return match.group(1).strip()
    return "" #if parsing fails

all_mcqs = load_all_mcqs()[:15]
eval_questions = [parse_question_stem(mcq) for mcq in all_mcqs if parse_question_stem(mcq)]


print(f"Loaded and parsed {len(eval_questions)} questions to use for evaluation.")



faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

print("Faithfulness and Relevancy evaluators are ready.")

'''
For Faithfulness Evaluation: The prompt sent to Gemini contains:
    The retrieved context (the text from your documents).
    The generated answer from your RAG pipeline.
    It then asks Gemini, "Is the answer supported by the given context?"
For Relevancy Evaluation: The prompt sent to Gemini contains:
    The original query (the test question).
    The retrieved context.
    The generated answer.
    It then asks Gemini, "Are the answer and the context relevant to the original query?"
'''
eval_results = []

for question in tqdm(eval_questions):
    response = query_engine.query(question)
    #is the answer supported by the context?
    faithfulness_result = faithfulness_evaluator.evaluate_response(response=response)
    time.sleep(2) 
    #is the answer relevant to the query?
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question, response=response
    )
    
    # Store the results
    eval_results.append({
        "query": question,
        "response": response.response,
        "is_faithful": faithfulness_result.passing,
        "faithfulness_score": faithfulness_result.score,
        "is_relevant": relevancy_result.passing,
        "relevancy_score": relevancy_result.score
    })
    time.sleep(2) 



results_df = pd.DataFrame(eval_results)
print("Evaluation complete.")
# output_path_csv = "tmp/rag_evaluation_results.csv"
output_path_json = "tmp/rag_evaluation_results.json"
# results_df.to_csv(output_path_csv, index=False)
results_df.to_json(output_path_json, orient='records', indent=2)
# print(f"Results saved to {output_path_csv}")
