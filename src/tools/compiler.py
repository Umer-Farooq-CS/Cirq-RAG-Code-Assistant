"""
Cirq Compiler Tool Module

This module implements the Cirq code compiler tool for real-time
code compilation, syntax checking, and validation.

Author: Umer Farooq, Hussain Waseem Syed, Muhammad Irtaza Khan
Email: umerfarooqcs0891@gmail.com

Purpose:
    - Compile and validate Cirq code
    - Check syntax correctness
    - Resolve imports and dependencies
    - Report compilation errors
    - Provide error suggestions

Input:
    - Cirq code (string)
    - Compilation options
    - Dependency requirements

Output:
    - Compilation status (success/failure)
    - Compiled circuit object (if successful)
    - Error messages and suggestions
    - Import resolution results

Dependencies:
    - Cirq: For code compilation
    - AST: For syntax analysis
    - Import resolution: For dependency checking

Links to other modules:
    - Used by: ValidatorAgent, DesignerAgent
    - Uses: Cirq, Python AST
    - Part of: Tool suite
"""

import ast
import sys
import traceback
from typing import Dict, Any, Optional, Tuple, List
from io import StringIO

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

from ..cirq_rag_code_assistant.config.logging import get_logger

logger = get_logger(__name__)


class CirqCompiler:
    """
    Compiles and validates Cirq code.
    
    Provides syntax checking, import resolution, and error reporting
    for Cirq quantum circuit code.
    """
    
    def __init__(self):
        """Initialize the CirqCompiler."""
        if not CIRQ_AVAILABLE:
            logger.warning("Cirq not available. Compilation will be limited.")
    
    def compile(
        self,
        code: str,
        execute: bool = False,
    ) -> Dict[str, Any]:
        """
        Compile Cirq code and check for errors.
        
        Args:
            code: Python code string containing Cirq code
            execute: Whether to execute the code (default: False, just validate)
            
        Returns:
            Dictionary with compilation status, errors, and results
        """
        result = {
            "success": False,
            "errors": [],
            "warnings": [],
            "circuit": None,
            "output": None,
        }
        
        # Step 1: Syntax validation
        syntax_valid, syntax_error = self._validate_syntax(code)
        if not syntax_valid:
            result["errors"].append({
                "type": "syntax_error",
                "message": syntax_error,
            })
            return result
        
        # Step 2: Import checking
        import_errors = self._check_imports(code)
        if import_errors:
            result["errors"].extend(import_errors)
            if not execute:
                return result
        
        # Step 3: Execute code if requested
        if execute:
            exec_result = self._execute_code(code)
            result.update(exec_result)
        else:
            result["success"] = True
        
        return result
    
    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Args:
            code: Code string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            return False, error_msg
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def _check_imports(self, code: str) -> List[Dict[str, str]]:
        """
        Check if required imports are available.
        
        Args:
            code: Code string to check
            
        Returns:
            List of import errors (empty if all imports available)
        """
        errors = []
        
        # Parse AST to find imports
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            # Check if imports are available
            for imp in set(imports):
                if imp == "cirq" and not CIRQ_AVAILABLE:
                    errors.append({
                        "type": "import_error",
                        "module": imp,
                        "message": f"Module '{imp}' is not installed",
                    })
                else:
                    try:
                        __import__(imp)
                    except ImportError:
                        errors.append({
                            "type": "import_error",
                            "module": imp,
                            "message": f"Module '{imp}' cannot be imported",
                        })
        
        except Exception as e:
            errors.append({
                "type": "import_check_error",
                "message": f"Error checking imports: {str(e)}",
            })
        
        return errors
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code and capture results.
        
        Args:
            code: Code string to execute
            
        Returns:
            Dictionary with execution results
        """
        result = {
            "success": False,
            "errors": [],
            "circuit": None,
            "output": None,
        }
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Create execution namespace
        namespace = {}
        if CIRQ_AVAILABLE:
            namespace["cirq"] = cirq
        
        try:
            exec(code, namespace)
            
            # Try to find circuit in namespace
            for key, value in namespace.items():
                if isinstance(value, cirq.Circuit):
                    result["circuit"] = value
                    break
            
            result["output"] = captured_output.getvalue()
            result["success"] = True
            
        except Exception as e:
            error_msg = traceback.format_exc()
            result["errors"].append({
                "type": "execution_error",
                "message": str(e),
                "traceback": error_msg,
            })
        
        finally:
            sys.stdout = old_stdout
        
        return result
    
    def extract_circuit(self, code: str) -> Optional[Any]:
        """
        Extract circuit object from code.
        
        Args:
            code: Code string
            
        Returns:
            Circuit object or None
        """
        if not CIRQ_AVAILABLE:
            return None
        
        result = self.compile(code, execute=True)
        return result.get("circuit")
    
    def validate_circuit(self, circuit: Any) -> Dict[str, Any]:
        """
        Validate a Cirq circuit object.
        
        Args:
            circuit: Cirq circuit object
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
        
        if not CIRQ_AVAILABLE:
            result["errors"].append("Cirq not available")
            return result
        
        if not isinstance(circuit, cirq.Circuit):
            result["errors"].append("Not a valid Cirq Circuit object")
            return result
        
        try:
            # Basic validation
            result["valid"] = True
            
            # Calculate metrics
            result["metrics"] = {
                "num_qubits": len(circuit.all_qubits()),
                "depth": len(circuit),
                "num_operations": len(list(circuit.all_operations())),
            }
            
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            result["valid"] = False
        
        return result
