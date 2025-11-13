"""
Educational Agent Module

This module implements the Educational Agent, responsible for providing
explanations, learning materials, and educational content for generated
code and quantum algorithms.
"""

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from ..rag.retriever import Retriever
from ..tools.analyzer import CircuitAnalyzer
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class EducationalAgent(BaseAgent):
    """Provides educational explanations and learning materials."""
    
    def __init__(
        self,
        retriever: Retriever,
        analyzer: Optional[CircuitAnalyzer] = None,
    ):
        """
        Initialize the EducationalAgent.
        
        Args:
            retriever: Retriever instance for educational content
            analyzer: CircuitAnalyzer instance
        """
        super().__init__(name="EducationalAgent")
        self.retriever = retriever
        self.analyzer = analyzer or CircuitAnalyzer()
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate educational explanations.
        
        Args:
            task: Task dictionary with 'code' or 'circuit' and optional 'depth', 'algorithm'
            
        Returns:
            Result dictionary with explanations and learning materials
        """
        code = task.get("code", "")
        circuit = task.get("circuit")
        depth = task.get("depth", "intermediate")
        algorithm = task.get("algorithm")
        
        if not code and not circuit:
            return {
                "success": False,
                "error": "Either 'code' or 'circuit' is required",
            }
        
        try:
            # Analyze circuit if provided
            analysis = None
            if circuit:
                analysis = self.analyzer.analyze(circuit)
            
            # Generate explanations
            explanations = self._generate_explanations(
                code=code,
                circuit=circuit,
                analysis=analysis,
                depth=depth,
                algorithm=algorithm,
            )
            
            # Retrieve learning materials
            learning_materials = self._retrieve_learning_materials(algorithm)
            
            return {
                "success": True,
                "explanations": explanations,
                "learning_materials": learning_materials,
                "analysis": analysis,
            }
            
        except Exception as e:
            logger.error(f"EducationalAgent error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _generate_explanations(
        self,
        code: str,
        circuit: Any,
        analysis: Optional[Dict[str, Any]],
        depth: str,
        algorithm: Optional[str],
    ) -> Dict[str, Any]:
        """Generate educational explanations."""
        explanations = {
            "overview": "",
            "step_by_step": [],
            "concepts": [],
            "code_structure": "",
        }
        
        # Overview
        if algorithm:
            explanations["overview"] = f"This code implements the {algorithm.upper()} algorithm using Cirq."
        else:
            explanations["overview"] = "This code implements a quantum circuit using Cirq."
        
        # Step-by-step explanation
        if code:
            lines = code.split('\n')
            explanations["step_by_step"] = [
                f"Line {i+1}: {line.strip()}" for i, line in enumerate(lines[:10])
                if line.strip() and not line.strip().startswith('#')
            ]
        
        # Concepts
        if analysis:
            metrics = analysis.get("metrics", {})
            explanations["concepts"] = [
                f"The circuit uses {metrics.get('num_qubits', 0)} qubits",
                f"The circuit depth is {metrics.get('depth', 0)}",
                f"The circuit has {metrics.get('num_operations', 0)} operations",
            ]
        
        return explanations
    
    def _retrieve_learning_materials(self, algorithm: Optional[str]) -> List[Dict[str, str]]:
        """Retrieve learning materials for the algorithm."""
        materials = []
        
        if algorithm:
            # Retrieve relevant educational content
            query = f"{algorithm} algorithm tutorial explanation"
            results = self.retriever.retrieve(query, top_k=3)
            
            for result in results:
                entry = result.get("entry", {})
                materials.append({
                    "title": entry.get("description", "Learning Material"),
                    "content": entry.get("code", "")[:500],
                    "source": entry.get("file", ""),
                })
        
        return materials
