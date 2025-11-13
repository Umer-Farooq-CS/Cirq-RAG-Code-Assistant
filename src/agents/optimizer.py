"""
Optimizer Agent Module

This module implements the Optimizer Agent, responsible for optimizing
quantum circuits for performance, efficiency, and hardware constraints.
"""

from typing import Dict, Any, Optional
import cirq
from .base_agent import BaseAgent
from ..tools.analyzer import CircuitAnalyzer
from ..tools.compiler import CirqCompiler
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class OptimizerAgent(BaseAgent):
    """Optimizes quantum circuits for performance and efficiency."""
    
    def __init__(
        self,
        analyzer: Optional[CircuitAnalyzer] = None,
        compiler: Optional[CirqCompiler] = None,
    ):
        """
        Initialize the OptimizerAgent.
        
        Args:
            analyzer: CircuitAnalyzer instance
            compiler: CirqCompiler instance
        """
        super().__init__(name="OptimizerAgent")
        self.analyzer = analyzer or CircuitAnalyzer()
        self.compiler = compiler or CirqCompiler()
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a quantum circuit.
        
        Args:
            task: Task dictionary with 'code' or 'circuit' and optional 'optimization_level'
            
        Returns:
            Result dictionary with optimized code and metrics
        """
        code = task.get("code")
        circuit = task.get("circuit")
        optimization_level = task.get("optimization_level", "balanced")
        
        if not code and not circuit:
            return {
                "success": False,
                "error": "Either 'code' or 'circuit' is required",
            }
        
        try:
            # Get circuit object
            if code:
                compiled = self.compiler.compile(code, execute=True)
                if not compiled["success"] or not compiled.get("circuit"):
                    return {
                        "success": False,
                        "error": f"Failed to compile code: {compiled.get('errors')}",
                    }
                circuit = compiled["circuit"]
            
            if not isinstance(circuit, cirq.Circuit):
                return {
                    "success": False,
                    "error": "Invalid circuit object",
                }
            
            # Analyze original circuit
            original_analysis = self.analyzer.analyze(circuit)
            
            # Optimize circuit
            optimized_circuit = self._optimize_circuit(circuit, optimization_level)
            
            # Analyze optimized circuit
            optimized_analysis = self.analyzer.analyze(optimized_circuit)
            
            # Compare results
            comparison = self.analyzer.compare(circuit, optimized_circuit)
            
            return {
                "success": True,
                "original_code": str(circuit) if code else None,
                "optimized_code": str(optimized_circuit),
                "original_metrics": original_analysis["metrics"],
                "optimized_metrics": optimized_analysis["metrics"],
                "improvements": comparison.get("improvements", []),
                "differences": comparison.get("differences", {}),
            }
            
        except Exception as e:
            logger.error(f"OptimizerAgent error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _optimize_circuit(self, circuit: cirq.Circuit, level: str) -> cirq.Circuit:
        """Apply optimizations to circuit."""
        optimized = circuit.copy()
        
        # Merge single-qubit gates
        optimized = cirq.merge_single_qubit_gates_to_phxz(optimized)
        
        # Drop negligible operations
        if level in ["balanced", "aggressive"]:
            optimized = cirq.drop_negligible_operations(optimized)
        
        # Eject Z gates
        if level == "aggressive":
            optimized = cirq.eject_z(optimized)
            optimized = cirq.eject_phased_paulis(optimized)
        
        return optimized
