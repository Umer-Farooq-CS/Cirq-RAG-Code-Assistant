"""
Data Preprocessor Module

This module handles preprocessing, cleaning, and validation of extracted
quantum code samples for the knowledge base.

Author: Umer Farooq, Hussain Waseem Syed, Muhammad Irtaza Khan
Email: umerfarooqcs0891@gmail.com

Purpose:
    - Clean and normalize code samples
    - Validate code quality and relevance
    - Remove duplicates and low-quality samples
    - Extract metadata (functions, classes, algorithms)
    - Organize data for knowledge base ingestion

Input:
    - Raw JSONL dataset files
    - Preprocessing configuration
    - Quality thresholds

Output:
    - Cleaned and validated JSONL files
    - Preprocessing statistics
    - Quality metrics
    - Metadata files

Dependencies:
    - json: For JSONL file handling
    - re: For pattern matching and extraction
    - ast: For code parsing and validation
    - cirq: For Cirq-specific validation (optional)
    - hashlib: For duplicate detection

Links to other modules:
    - Used by: Data preprocessing pipeline
    - Uses: DatasetLoader for loading raw data
    - Output feeds into: KnowledgeBase, VectorStore
"""

import json
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from tqdm import tqdm

from .dataset_loader import DatasetLoader


class DataPreprocessor:
    """
    Preprocesses and cleans quantum code datasets.
    
    Handles code validation, duplicate removal, quality filtering,
    and metadata extraction.
    """
    
    # Minimum and maximum code length thresholds
    DEFAULT_MIN_LENGTH = 50
    DEFAULT_MAX_LENGTH = 50000
    
    # Quality thresholds
    DEFAULT_MIN_LINES = 5
    DEFAULT_MAX_LINES = 1000
    
    def __init__(
        self,
        min_code_length: int = DEFAULT_MIN_LENGTH,
        max_code_length: int = DEFAULT_MAX_LENGTH,
        min_lines: int = DEFAULT_MIN_LINES,
        max_lines: int = DEFAULT_MAX_LINES,
        remove_duplicates: bool = True,
        validate_syntax: bool = True,
    ):
        """
        Initialize the DataPreprocessor.
        
        Args:
            min_code_length: Minimum code length in characters
            max_code_length: Maximum code length in characters
            min_lines: Minimum number of lines
            max_lines: Maximum number of lines
            remove_duplicates: Whether to remove duplicate code samples
            validate_syntax: Whether to validate Python syntax
        """
        self.min_code_length = min_code_length
        self.max_code_length = max_code_length
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.remove_duplicates = remove_duplicates
        self.validate_syntax = validate_syntax
        self._seen_hashes: Set[str] = set()
    
    def calculate_code_hash(self, code: str) -> str:
        """
        Calculate hash of code for duplicate detection.
        
        Args:
            code: Code content
            
        Returns:
            MD5 hash of normalized code
        """
        # Normalize code (remove extra whitespace, normalize line endings)
        normalized = re.sub(r'\s+', ' ', code.strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, code: str) -> bool:
        """
        Check if code is a duplicate.
        
        Args:
            code: Code content
            
        Returns:
            True if duplicate, False otherwise
        """
        if not self.remove_duplicates:
            return False
        
        code_hash = self.calculate_code_hash(code)
        if code_hash in self._seen_hashes:
            return True
        
        self._seen_hashes.add(code_hash)
        return False
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax of code.
        
        Args:
            code: Code content
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.validate_syntax:
            return True, None
        
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def check_code_quality(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check code quality and return issues.
        
        Args:
            code: Code content
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check length
        code_length = len(code)
        if code_length < self.min_code_length:
            issues.append(f"Code too short: {code_length} < {self.min_code_length}")
        if code_length > self.max_code_length:
            issues.append(f"Code too long: {code_length} > {self.max_code_length}")
        
        # Check line count
        lines = code.split('\n')
        num_lines = len(lines)
        if num_lines < self.min_lines:
            issues.append(f"Too few lines: {num_lines} < {self.min_lines}")
        if num_lines > self.max_lines:
            issues.append(f"Too many lines: {num_lines} > {self.max_lines}")
        
        # Check for empty code
        if not code.strip():
            issues.append("Code is empty")
        
        # Check for only comments
        non_comment_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]
        if len(non_comment_lines) < 3:
            issues.append("Code contains mostly comments")
        
        # Validate syntax
        is_valid_syntax, syntax_error = self.validate_python_syntax(code)
        if not is_valid_syntax:
            issues.append(f"Syntax error: {syntax_error}")
        
        return len(issues) == 0, issues
    
    def extract_metadata(self, code: str, framework: str) -> Dict:
        """
        Extract metadata from code sample.
        
        Args:
            code: Code content
            framework: Framework name
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "code_length": len(code),
            "num_lines": len(code.split('\n')),
            "num_functions": len(re.findall(r'def\s+\w+', code)),
            "num_classes": len(re.findall(r'class\s+\w+', code)),
            "num_imports": len(re.findall(r'^(import|from)\s+', code, re.MULTILINE)),
        }
        
        # Extract function names
        func_names = re.findall(r'def\s+([a-zA-Z_]\w*)', code)
        if func_names:
            metadata["functions"] = func_names[:5]  # Top 5 functions
        
        # Extract class names
        class_names = re.findall(r'class\s+([a-zA-Z_]\w*)', code)
        if class_names:
            metadata["classes"] = class_names[:5]  # Top 5 classes
        
        # Detect algorithm patterns
        algorithm_patterns = {
            "vqe": r"vqe|variational|eigensolver",
            "qaoa": r"qaoa|max.?cut",
            "grover": r"grover|amplitude.?amplification",
            "qft": r"qft|quantum.?fourier",
            "teleportation": r"teleport|bell.?state",
        }
        
        code_lower = code.lower()
        detected_algorithms = [
            algo for algo, pattern in algorithm_patterns.items()
            if re.search(pattern, code_lower)
        ]
        if detected_algorithms:
            metadata["algorithms"] = detected_algorithms
        
        return metadata
    
    def preprocess_entry(
        self,
        entry: Dict,
        add_metadata: bool = True,
    ) -> Optional[Dict]:
        """
        Preprocess a single dataset entry.
        
        Args:
            entry: Dataset entry dictionary
            add_metadata: Whether to add extracted metadata
            
        Returns:
            Preprocessed entry or None if filtered out
        """
        code = entry.get("code", "")
        framework = entry.get("framework", "Unknown")
        
        # Check for duplicates
        if self.is_duplicate(code):
            return None
        
        # Check code quality
        is_valid, issues = self.check_code_quality(code)
        if not is_valid:
            return None
        
        # Create preprocessed entry
        processed_entry = {
            "framework": framework,
            "file": entry.get("file", ""),
            "code": code,
        }
        
        # Preserve existing fields
        if "description" in entry:
            processed_entry["description"] = entry["description"]
        
        # Add metadata
        if add_metadata:
            metadata = self.extract_metadata(code, framework)
            processed_entry["metadata"] = metadata
        
        return processed_entry
    
    def preprocess_dataset(
        self,
        input_path: Path,
        output_path: Path,
        add_metadata: bool = True,
    ) -> Dict:
        """
        Preprocess an entire dataset.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            add_metadata: Whether to add extracted metadata
            
        Returns:
            Statistics dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reset duplicate tracking
        self._seen_hashes.clear()
        
        stats = {
            "total": 0,
            "processed": 0,
            "filtered": 0,
            "duplicates": 0,
            "quality_issues": 0,
        }
        
        processed_entries = []
        
        # Load and process entries
        print(f"Loading dataset from {input_path}...")
        loader = DatasetLoader(input_path)
        entries = loader.load()
        stats["total"] = len(entries)
        
        print(f"Preprocessing {stats['total']} entries...")
        for entry in tqdm(entries, desc="Preprocessing"):
            # Check for duplicates
            code = entry.get("code", "")
            if self.is_duplicate(code):
                stats["duplicates"] += 1
                continue
            
            # Preprocess entry
            processed = self.preprocess_entry(entry, add_metadata=add_metadata)
            
            if processed is None:
                stats["filtered"] += 1
                continue
            
            processed_entries.append(processed)
            stats["processed"] += 1
        
        # Write output file
        print(f"Writing preprocessed data to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Calculate quality issues
        stats["quality_issues"] = stats["filtered"] - stats["duplicates"]
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"✅ Preprocessing complete!")
        print(f"{'='*60}")
        print(f"Total entries: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Filtered out: {stats['filtered']}")
        print(f"  - Duplicates: {stats['duplicates']}")
        print(f"  - Quality issues: {stats['quality_issues']}")
        print(f"Retention rate: {stats['processed']/stats['total']:.1%}")
        print(f"\n💾 Saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess quantum code dataset")
    parser.add_argument("input", type=str, help="Input JSONL file path")
    parser.add_argument("output", type=str, help="Output JSONL file path")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum code length")
    parser.add_argument("--max-length", type=int, default=50000,
                       help="Maximum code length")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip syntax validation")
    parser.add_argument("--no-dedup", action="store_true",
                       help="Skip duplicate removal")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(
        min_code_length=args.min_length,
        max_code_length=args.max_length,
        validate_syntax=not args.no_validate,
        remove_duplicates=not args.no_dedup,
    )
    
    preprocessor.preprocess_dataset(
        Path(args.input),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
