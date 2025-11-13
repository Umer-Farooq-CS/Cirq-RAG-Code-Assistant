"""
RAG Generator Module

This module implements the generation component of the RAG system.
It combines retrieved context with LLM capabilities to generate
Cirq code from natural language descriptions.

Author: Umer Farooq, Hussain Waseem Syed, Muhammad Irtaza Khan
Email: umerfarooqcs0891@gmail.com

Purpose:
    - Generate Cirq code from natural language + retrieved context
    - Combine RAG retrieval with LLM generation
    - Apply code templates and best practices
    - Ensure syntax correctness and Cirq conventions

Input:
    - Natural language description
    - Retrieved context (code examples, documentation)
    - Algorithm type and parameters
    - Generation parameters (temperature, max_tokens, etc.)

Output:
    - Generated Cirq code
    - Code metadata (algorithm, complexity, imports)
    - Generation confidence scores
    - Explanation of code structure

Dependencies:
    - Retriever: For context retrieval
    - LLM API (OpenAI/Anthropic): For code generation
    - Code templates: For structured generation
    - Cirq: For code validation

Links to other modules:
    - Used by: DesignerAgent
    - Uses: Retriever, LLM APIs, Code templates
    - Part of: RAG system pipeline
"""

import os
import re
from typing import Dict, Optional, Any, List
from .retriever import Retriever

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..cirq_rag_code_assistant.config import get_config
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class Generator:
    """
    Generates Cirq code using RAG (Retrieval-Augmented Generation).
    
    Combines retrieved context from the knowledge base with LLM capabilities
    to generate accurate, executable Cirq code.
    """
    
    # Code generation prompt template
    PROMPT_TEMPLATE = """You are an expert quantum computing programmer specializing in Google's Cirq framework.

Your task is to generate syntactically correct, executable Cirq code based on the user's request and the provided examples.

Context from knowledge base:
{context}

User request: {query}

Instructions:
1. Generate complete, executable Cirq code that fulfills the user's request
2. Follow Cirq best practices and conventions
3. Include necessary imports
4. Add comments explaining key steps
5. Ensure the code is syntactically correct and can be executed

Generated code:"""

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4",
        provider: str = "openai",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ):
        """
        Initialize the Generator.
        
        Args:
            retriever: Retriever instance for context retrieval
            model: LLM model name
            provider: LLM provider ("openai" or "anthropic")
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.retriever = retriever
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM client
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library is required. "
                    "Install it with: pip install openai"
                )
            config = get_config()
            api_key = config.get("api_keys.openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in config or environment")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "Anthropic library is required. "
                    "Install it with: pip install anthropic"
                )
            config = get_config()
            api_key = config.get("api_keys.anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in config or environment")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized Generator with {provider}/{model}")
    
    def generate(
        self,
        query: str,
        algorithm: Optional[str] = None,
        top_k: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Cirq code from a natural language query.
        
        Args:
            query: Natural language description of desired code
            algorithm: Optional algorithm type (e.g., "vqe", "qaoa")
            top_k: Number of examples to retrieve for context
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated code, metadata, and confidence
        """
        # Retrieve relevant context
        logger.info(f"Retrieving context for query: {query[:50]}...")
        context_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            algorithm=algorithm,
        )
        
        # Format context
        context = self.retriever.retrieve_context(
            query=query,
            top_k=top_k,
            algorithm=algorithm,
        )
        
        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(
            context=context if context else "No relevant examples found.",
            query=query,
        )
        
        # Generate code using LLM
        logger.info(f"Generating code using {self.provider}/{self.model}")
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert Cirq quantum computing programmer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                generated_text = response.choices[0].message.content
            else:  # anthropic
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                )
                generated_text = response.content[0].text
            
            # Extract code from response
            code = self._extract_code(generated_text)
            
            # Extract metadata
            metadata = self._extract_metadata(code, algorithm)
            
            result = {
                "code": code,
                "raw_response": generated_text,
                "metadata": metadata,
                "context_used": len(context_results),
                "algorithm": algorithm,
            }
            
            logger.info("✅ Code generation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code block from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted code string
        """
        # Try to find code blocks
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            # Return the first code block
            return matches[0].strip()
        
        # If no code block, try to find import statements and code
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see an import
            if 'import' in line and ('cirq' in line.lower() or 'numpy' in line.lower()):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: return the entire text
        return text.strip()
    
    def _extract_metadata(self, code: str, algorithm: Optional[str]) -> Dict[str, Any]:
        """
        Extract metadata from generated code.
        
        Args:
            code: Generated code
            algorithm: Algorithm type if known
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "code_length": len(code),
            "num_lines": len(code.split('\n')),
            "has_imports": "import" in code,
            "has_cirq": "cirq" in code.lower(),
        }
        
        # Extract imports
        import_pattern = r"(?:^|\n)(?:import|from)\s+([^\s]+)"
        imports = re.findall(import_pattern, code, re.MULTILINE)
        if imports:
            metadata["imports"] = imports
        
        # Detect algorithm if not provided
        if not algorithm:
            algorithm_patterns = {
                "vqe": r"vqe|variational|eigensolver",
                "qaoa": r"qaoa|max.?cut",
                "grover": r"grover|amplitude",
                "qft": r"qft|fourier",
                "teleportation": r"teleport|bell",
            }
            
            code_lower = code.lower()
            for algo, pattern in algorithm_patterns.items():
                if re.search(pattern, code_lower):
                    metadata["detected_algorithm"] = algo
                    break
        
        return metadata
