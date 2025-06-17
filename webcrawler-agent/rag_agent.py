"""Genesys agent that leverages RAG with a local ChromaDB for documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List
import asyncio
import chromadb

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
# from openai import AsyncOpenAI # This line is not used in the original code snippet provided
import openai # Import openai

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context,
    # REMOVED: list_collections # This function does not exist in your utils.py and is not used
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

# Initialize OpenAI client (Assuming the agent uses this implicitly or via pydantic_ai)
# If your pydantic_ai setup requires passing the client, you might need to adjust RAGDeps
# For now, assume it uses the env var and the underlying library handles client creation.
# If not, you might need something like:
# openai_client = AsyncOpenAI()


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    # collection_name is now a list of names
    collection_names: List[str]
    embedding_model: str


# Create the RAG agent
# Note: The agent is created here at module level.
# Its dependencies will be provided dynamically when run_stream or run is called.
agent = Agent(
    os.getenv("MODEL_CHOICE", "gpt-4o"), # Switched to a common model name, check your .env
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the documentation before answering. "
                  "When using the retrieve tool, your search query should be concise and focused on the user's question. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response."
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 10) -> str:
    """Retrieve relevant documents from multiple ChromaDB collections based on a search query.

    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Maximum number of *final* results to return after combining (default: 10).

    Returns:
        Formatted context information from the top relevant retrieved documents.
    """
    all_query_results = []
    
    # Iterate through all specified collections
    for collection_name in context.deps.collection_names:
        try:
            # Get the collection object (assuming it exists)
            collection = get_or_create_collection(
                context.deps.chroma_client,
                collection_name,
                embedding_model_name=context.deps.embedding_model # Pass the model name again
            )

            # Query the current collection. Fetch more results initially per collection
            # than the final n_results, to have a larger pool for re-ranking.
            # Let's fetch n_results * 2 per collection, up to a reasonable limit (e.g., 20)
            results_per_collection = min(n_results * 2, 20)
            
            query_results = query_collection(
                collection,
                search_query,
                n_results=results_per_collection # Fetch more per collection
            )
            
            # Add results from this collection to the main list
            # Assuming query_collection returns results in a structure that includes distance,
            # documents, and metadatas lists, each containing a list for the single query.
            if query_results and query_results.get('distances') and query_results.get('documents') and query_results.get('metadatas'):
                 # Structure is likely: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
                 # Process results for the first query (index 0)
                 if query_results['distances'] and query_results['documents'] and query_results['metadatas']:
                      for i in range(len(query_results['distances'][0])):
                           all_query_results.append({
                                'distance': query_results['distances'][0][i],
                                'document': query_results['documents'][0][i],
                                'metadata': query_results['metadatas'][0][i]
                           })

        except Exception as e:
            print(f"Warning: Could not query collection '{collection_name}': {e}", file=sys.stderr)
            # Continue to the next collection if one fails

    # Sort all collected results by distance (lower distance is better)
    all_query_results.sort(key=lambda x: x['distance'])

    # Take the top N results overall
    top_n_results = all_query_results[:n_results]
    
    # Reformat the results back into a structure format_results_as_context expects
    # which is similar to the original query_collection output format for a single query.
    formatted_for_context = {
        'ids': [[""] * len(top_n_results)], # IDs might not be strictly needed by formatter, provide placeholder
        'documents': [[res['document'] for res in top_n_results]],
        'metadatas': [[res['metadata'] for res in top_n_results]],
        'distances': [[res['distance'] for res in top_n_results]]
    }

    # Format the results as context
    return format_results_as_context(formatted_for_context)


# The standalone run_rag_agent and main functions below are for command-line usage
# and also need updates if you intend to use them with multiple collections from CLI args.
# For the Streamlit app, only the RAGDeps and retrieve tool modifications are strictly necessary.
# However, let's update run_rag_agent to accept a list of collection names for completeness.

async def run_rag_agent(
    question: str,
    # collection_name is now collection_names (list)
    collection_names: List[str],
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    n_results: int = 10 # Increased default
) -> str:
    """Run the RAG agent to answer a question about Genesys using multiple collections.

    Args:
        question: The question to answer.
        collection_names: List of names of the ChromaDB collections to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.

    Returns:
        The agent's response.
    """
    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_names=collection_names, # Pass the list
        embedding_model=embedding_model
    )

    # Run the agent
    result = await agent.run(question, deps=deps)

    return result.data


# Modifying main to handle a comma-separated list of collections from CLI
def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Genesys Docs agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Genesys")
    # Change --collection to accept a comma-separated list
    parser.add_argument("--collections", default="docs", help="Comma-separated list of ChromaDB collection names")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=10, help="Number of results to return from the retrieval")

    args = parser.parse_args()

    # Parse the comma-separated collection names into a list
    collection_names_list = [name.strip() for name in args.collections.split(',')]

    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_names=collection_names_list, # Pass the list
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    # If you want to run this script directly from the command line
    # with multiple collections, use --collections collection1,collection2,...
    # Example: python rag_agent.py --question "What is a skill?" --db-dir ./genesys_docs --collections pages,glossary
    main()