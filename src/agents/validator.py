"""
Validator Agent Module

This module implements the Validator Agent, responsible for testing,
validating, and ensuring quality of generated Cirq code.
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..tools.compiler import CirqCompiler
from ..tools.simulator import QuantumSimulator
from ..tools.analyzer import CircuitAnalyzer
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class ValidatorAgent(BaseAgent):
    """Validates and tests quantum circuits."""
    
    def __init__(
        self,
        compiler: Optional[CirqCompiler] = None,
        simulator: Optional[QuantumSimulator] = None,
        analyzer: Optional[CircuitAnalyzer] = None,
    ):
        """
        Initialize the ValidatorAgent.
        
        Args:
            compiler: CirqCompiler instance
            simulator: QuantumSimulator instance
            analyzer: CircuitAnalyzer instance
        """
        super().__init__(name="ValidatorAgent")
        self.compiler = compiler or CirqCompiler()
        self.simulator = simulator or QuantumSimulator()
        self.analyzer = analyzer or CircuitAnalyzer()
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Cirq code.
        
        Args:
            task: Task dictionary with 'code' and optional 'validation_level'
            
        Returns:
            Result dictionary with validation results
        """
        code = task.get("code", "")
        validation_level = task.get("validation_level", "comprehensive")
        
        if not code:
            return {
                "success": False,
                "error": "Code is required",
            }
        
        try:
            # Compile and validate syntax
            compilation = self.compiler.compile(code, execute=True)
            
            if not compilation["success"]:
                return {
                    "success": False,
                    "validation_passed": False,
                    "errors": compilation["errors"],
                    "compilation": compilation,
                }
            
            circuit = compilation.get("circuit")
            if not circuit:
                return {
                    "success": False,
                    "validation_passed": False,
                    "error": "No circuit found in code",
                }
            
            # Validate circuit structure
            circuit_validation = self.compiler.validate_circuit(circuit)
            
            # Analyze circuit
            analysis = self.analyzer.analyze(circuit)
            
            # Run simulation if comprehensive validation
            simulation_result = None
            if validation_level == "comprehensive":
                simulation_result = self.simulator.simulate(circuit, repetitions=100)
            
            # Determine overall validation status
            validation_passed = (
                compilation["success"] and
                circuit_validation["valid"] and
                (simulation_result is None or simulation_result["success"])
            )
            
            return {
                "success": True,
                "validation_passed": validation_passed,
                "compilation": compilation,
                "circuit_validation": circuit_validation,
                "analysis": analysis,
                "simulation": simulation_result,
                "errors": compilation.get("errors", []) + circuit_validation.get("errors", []),
            }
            
        except Exception as e:
            logger.error(f"ValidatorAgent error: {e}")
            return {
                "success": False,
                "validation_passed": False,
                "error": str(e),
            }
