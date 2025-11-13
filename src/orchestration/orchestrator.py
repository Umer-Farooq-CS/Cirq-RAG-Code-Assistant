"""
Orchestrator Module

This module implements the main orchestrator that coordinates
all agents and manages the overall workflow.
"""

from typing import Dict, Any, Optional, List
from ..agents.designer import DesignerAgent
from ..agents.optimizer import OptimizerAgent
from ..agents.validator import ValidatorAgent
from ..agents.educational import EducationalAgent
from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Coordinates multi-agent system for code generation workflow."""
    
    def __init__(
        self,
        designer: DesignerAgent,
        optimizer: Optional[OptimizerAgent] = None,
        validator: Optional[ValidatorAgent] = None,
        educational: Optional[EducationalAgent] = None,
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            designer: DesignerAgent instance
            optimizer: Optional OptimizerAgent instance
            validator: Optional ValidatorAgent instance
            educational: Optional EducationalAgent instance
        """
        self.designer = designer
        self.optimizer = optimizer
        self.validator = validator
        self.educational = educational
        
        logger.info("Initialized Orchestrator")
    
    def generate_code(
        self,
        query: str,
        algorithm: Optional[str] = None,
        optimize: bool = True,
        validate: bool = True,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate code with optional optimization, validation, and explanation.
        
        Args:
            query: Natural language query
            algorithm: Optional algorithm type
            optimize: Whether to optimize the code
            validate: Whether to validate the code
            explain: Whether to generate explanations
            
        Returns:
            Complete result dictionary
        """
        result = {
            "success": False,
            "query": query,
            "algorithm": algorithm,
            "code": None,
            "optimized_code": None,
            "validation": None,
            "explanation": None,
            "errors": [],
        }
        
        try:
            # Step 1: Generate code
            logger.info(f"Generating code for query: {query[:50]}...")
            design_result = self.designer.run({"query": query, "algorithm": algorithm})
            
            if not design_result.get("success"):
                result["errors"].append(f"Design failed: {design_result.get('error')}")
                return result
            
            result["code"] = design_result.get("code")
            
            # Step 2: Optimize (if requested)
            if optimize and self.optimizer and result["code"]:
                logger.info("Optimizing code...")
                optimize_result = self.optimizer.run({"code": result["code"]})
                
                if optimize_result.get("success"):
                    result["optimized_code"] = optimize_result.get("optimized_code")
                    result["optimization_metrics"] = optimize_result.get("differences", {})
                else:
                    result["errors"].append(f"Optimization failed: {optimize_result.get('error')}")
            
            # Step 3: Validate (if requested)
            code_to_validate = result.get("optimized_code") or result["code"]
            if validate and self.validator and code_to_validate:
                logger.info("Validating code...")
                validate_result = self.validator.run({"code": code_to_validate})
                
                result["validation"] = validate_result
                if not validate_result.get("validation_passed"):
                    result["errors"].append("Validation failed")
            
            # Step 4: Generate explanation (if requested)
            if explain and self.educational and code_to_validate:
                logger.info("Generating explanation...")
                explain_result = self.educational.run({
                    "code": code_to_validate,
                    "algorithm": algorithm,
                })
                
                if explain_result.get("success"):
                    result["explanation"] = explain_result.get("explanations", {})
                    result["learning_materials"] = explain_result.get("learning_materials", [])
            
            result["success"] = True
            logger.info("✅ Code generation workflow completed")
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            result["errors"].append(str(e))
        
        return result
