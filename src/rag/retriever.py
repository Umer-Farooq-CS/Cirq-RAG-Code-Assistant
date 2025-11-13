"""
RAG Retriever Module

This module implements the retrieval component of the RAG system.
It handles semantic search, context retrieval, and relevance ranking
for Cirq code examples and documentation.

Author: Umer Farooq, Hussain Waseem Syed, Muhammad Irtaza Khan
Email: umerfarooqcs0891@gmail.com

Purpose:
    - Retrieve relevant Cirq code examples from knowledge base
    - Perform semantic similarity search using embeddings
    - Rank and filter results based on relevance scores
    - Provide context for code generation

Input:
    - Natural language queries (user requests)
    - Algorithm type (optional)
    - Similarity threshold
    - Top-k results count

Output:
    - Ranked list of relevant code snippets
    - Metadata (algorithm type, complexity, tags)
    - Similarity scores
    - Contextual information

Dependencies:
    - VectorStore: For similarity search
    - EmbeddingModel: For query embeddings
    - KnowledgeBase: For accessing code examples
    - PyTorch: For GPU-accelerated embeddings

Links to other modules:
    - Used by: DesignerAgent, OptimizerAgent, ValidatorAgent, EducationalAgent
    - Uses: VectorStore, EmbeddingModel, KnowledgeBase
    - Part of: RAG system pipeline
"""

from typing import List, Dict, Optional, Any
from .knowledge_base import KnowledgeBase
from ..cirq_rag_code_assistant.config import get_config
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant context from the knowledge base using semantic search.
    
    Handles query processing, similarity search, and result ranking to provide
    high-quality context for code generation.
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the Retriever.
        
        Args:
            knowledge_base: KnowledgeBase instance
            top_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initialized Retriever with top_k={top_k}, threshold={similarity_threshold}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        algorithm: Optional[str] = None,
        framework: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return (uses self.top_k if None)
            algorithm: Optional algorithm filter
            framework: Optional framework filter (defaults to "Cirq")
            similarity_threshold: Minimum similarity score (uses self.similarity_threshold if None)
            
        Returns:
            List of retrieved entries with scores and metadata
        """
        top_k = top_k or self.top_k
        threshold = similarity_threshold or self.similarity_threshold
        framework = framework or "Cirq"
        
        # Build filter dictionary
        filter_dict = {"framework": framework}
        if algorithm:
            filter_dict["algorithms"] = algorithm
        
        # Search knowledge base
        results = self.knowledge_base.search(
            query=query,
            top_k=top_k * 2,  # Retrieve more, then filter by threshold
            filter_dict=filter_dict if filter_dict else None,
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result["score"] >= threshold
        ]
        
        # Limit to top_k
        filtered_results = filtered_results[:top_k]
        
        logger.debug(f"Retrieved {len(filtered_results)} results for query: {query[:50]}...")
        
        return filtered_results
    
    def retrieve_code_examples(
        self,
        query: str,
        top_k: Optional[int] = None,
        algorithm: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve code examples as strings.
        
        Args:
            query: Natural language query
            top_k: Number of examples to return
            algorithm: Optional algorithm filter
            
        Returns:
            List of code strings
        """
        results = self.retrieve(query, top_k=top_k, algorithm=algorithm)
        
        code_examples = []
        for result in results:
            entry = result["entry"]
            code = entry.get("code", "")
            if code:
                code_examples.append(code)
        
        return code_examples
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        algorithm: Optional[str] = None,
        include_descriptions: bool = True,
    ) -> str:
        """
        Retrieve and format context as a single string.
        
        Args:
            query: Natural language query
            top_k: Number of examples to include
            algorithm: Optional algorithm filter
            include_descriptions: Whether to include descriptions
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k, algorithm=algorithm)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            entry = result["entry"]
            
            part = f"Example {i}:\n"
            
            if include_descriptions and "description" in entry:
                part += f"Description: {entry['description']}\n"
            
            if "code" in entry:
                part += f"Code:\n{entry['code']}\n"
            
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
        algorithm: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results with full metadata.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            algorithm: Optional algorithm filter
            
        Returns:
            List of dictionaries with entry, score, and metadata
        """
        return self.retrieve(query, top_k=top_k, algorithm=algorithm)
