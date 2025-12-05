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
            
            # RL optimization
            if task.get("use_rl", False):
                optimized_circuit = self._optimize_with_rl(optimized_circuit)
                # Re-analyze after RL
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
        """Apply heuristic optimizations to circuit."""
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

    def _optimize_with_rl(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """
        Apply Reinforcement Learning based optimization.
        
        This is a simplified implementation that iteratively applies transformations
        and keeps them if they improve the reward function defined in config.
        """
        from ..cirq_rag_code_assistant.config import get_config
        config = get_config()
        rl_config = config.get("agents", {}).get("optimizer", {})
        weights = rl_config.get("rl_reward_weights", {})
        iterations = rl_config.get("rl_iterations", 10)
        
        current_circuit = circuit.copy()
        current_metrics = self.analyzer.analyze(current_circuit)["metrics"]
        current_reward = self._calculate_reward(current_metrics, weights)
        
        logger.info(f"Starting RL optimization (iterations={iterations})")
        
        # List of available transformations
        transformations = [
            cirq.merge_single_qubit_gates_to_phxz,
            cirq.drop_negligible_operations,
            cirq.eject_z,
            cirq.eject_phased_paulis,
            cirq.expand_composite,
            cirq.defer_measurements,
        ]
        
        for i in range(iterations):
            # Try each transformation
            best_next_circuit = current_circuit
            best_next_reward = current_reward
            improved = False
            
            for transform in transformations:
                try:
                    candidate = transform(current_circuit)
                    metrics = self.analyzer.analyze(candidate)["metrics"]
                    reward = self._calculate_reward(metrics, weights)
                    
                    if reward > best_next_reward:
                        best_next_circuit = candidate
                        best_next_reward = reward
                        improved = True
                except Exception:
                    continue
            
            if improved:
                current_circuit = best_next_circuit
                current_reward = best_next_reward
                logger.debug(f"RL Iteration {i+1}: Improved reward to {current_reward:.4f}")
            else:
                logger.debug(f"RL Iteration {i+1}: No improvement")
                # Early stopping if no improvement found in this iteration
                # (Could add exploration here in full RL)
                break
                
        return current_circuit

    def _calculate_reward(self, metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Calculate reward based on circuit metrics and weights."""
        reward = 0.0
        
        # Depth (lower is better)
        if "depth" in metrics:
            reward += weights.get("circuit_depth", -0.1) * metrics["depth"]
            
        # Gate count (lower is better)
        if "total_gate_count" in metrics:
            reward += weights.get("total_gate_count", -0.1) * metrics["total_gate_count"]
            
        # Two-qubit gates (lower is better)
        if "2_qubit_gate_count" in metrics:
            reward += weights.get("two_qubit_gates", -0.5) * metrics["2_qubit_gate_count"]
            
        return reward
