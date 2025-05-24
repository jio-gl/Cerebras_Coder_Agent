"""Module for checking semantic equivalence between LLM outputs."""
import json
from typing import Any, Dict, List, Union
import difflib
from dataclasses import dataclass
import re

@dataclass
class EquivalenceResult:
    """Result of an equivalence check."""
    is_equivalent: bool
    similarity_score: float
    differences: List[str]

class EquivalenceChecker:
    """Checker for semantic equivalence between LLM outputs."""
    
    @staticmethod
    def _normalize_json(json_str: str) -> Dict[str, Any]:
        """Normalize JSON string by parsing and re-stringifying."""
        try:
            if isinstance(json_str, str):
                return json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            return json_str

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return ' '.join(text.split())

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code by removing comments and extra whitespace."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove empty lines
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def check_json_equivalence(self, json1: Union[str, Dict], json2: Union[str, Dict]) -> EquivalenceResult:
        """Check if two JSON objects are semantically equivalent."""
        try:
            normalized1 = self._normalize_json(json1)
            normalized2 = self._normalize_json(json2)
            
            # Convert to sorted string representation for comparison
            str1 = json.dumps(normalized1, sort_keys=True)
            str2 = json.dumps(normalized2, sort_keys=True)
            
            similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
            differences = list(difflib.unified_diff(
                str1.splitlines(), str2.splitlines(), lineterm=''))
            
            # They're equivalent if the normalized forms are exactly equal
            is_equivalent = normalized1 == normalized2
            
            return EquivalenceResult(
                is_equivalent=is_equivalent,
                similarity_score=similarity,
                differences=differences
            )
        except Exception as e:
            return EquivalenceResult(
                is_equivalent=False,
                similarity_score=0.0,
                differences=[f"Error comparing JSONs: {str(e)}"]
            )

    def check_text_equivalence(self, text1: str, text2: str, 
                             similarity_threshold: float = 0.95) -> EquivalenceResult:
        """Check if two text strings are semantically equivalent."""
        norm1 = self._normalize_whitespace(text1)
        norm2 = self._normalize_whitespace(text2)
        
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        differences = list(difflib.unified_diff(
            norm1.splitlines(), norm2.splitlines(), lineterm=''))
        
        return EquivalenceResult(
            is_equivalent=similarity >= similarity_threshold,
            similarity_score=similarity,
            differences=differences
        )

    def check_code_equivalence(self, code1: str, code2: str,
                             similarity_threshold: float = 0.95) -> EquivalenceResult:
        """Check if two code snippets are semantically equivalent."""
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)
        
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        differences = list(difflib.unified_diff(
            norm1.splitlines(), norm2.splitlines(), lineterm=''))
        
        return EquivalenceResult(
            is_equivalent=similarity >= similarity_threshold,
            similarity_score=similarity,
            differences=differences
        )

    def check_file_structure_equivalence(self, structure1: Dict[str, Any], 
                                       structure2: Dict[str, Any]) -> EquivalenceResult:
        """Check if two file/directory structures are equivalent."""
        def normalize_structure(struct: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize a file structure dictionary."""
            result = {}
            for k, v in struct.items():
                k = k.rstrip('/')  # Remove trailing slashes
                if isinstance(v, dict):
                    result[k] = normalize_structure(v)
                else:
                    result[k] = v
            return dict(sorted(result.items()))

        norm1 = normalize_structure(structure1)
        norm2 = normalize_structure(structure2)
        
        str1 = json.dumps(norm1, sort_keys=True)
        str2 = json.dumps(norm2, sort_keys=True)
        
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        differences = list(difflib.unified_diff(
            str1.splitlines(), str2.splitlines(), lineterm=''))
        
        return EquivalenceResult(
            is_equivalent=norm1 == norm2,
            similarity_score=similarity,
            differences=differences
        ) 