"""
=================================================================================
BREVIT.PY (brevit-py)

A high-performance Python library for semantically compressing
and optimizing data before sending it to a Large Language Model (LLM).

Project: Brevit
Author: Javian
Version: 0.1.0
=================================================================================
"""

import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Protocol, Union
from abc import ABC, abstractmethod

# region == Public Enums and Config ==

class JsonOptimizationMode(Enum):
    NONE = auto()
    Flatten = auto()
    ToYaml = auto()  # Note: Requires a YAML library like 'PyYAML'
    Filter = auto()  # Note: Requires a JSON-path library like 'jsonpath-ng'

class TextOptimizationMode(Enum):
    NONE = auto()
    Clean = auto()
    SummarizeFast = auto()
    SummarizeHighQuality = auto()

class ImageOptimizationMode(Enum):
    NONE = auto()
    Ocr = auto()
    Metadata = auto()

@dataclass
class BrevitConfig:
    """
    Configuration object for the BrevitClient.
    This defines the rules for the optimization pipeline.
    """
    json_mode: JsonOptimizationMode = JsonOptimizationMode.Flatten
    text_mode: TextOptimizationMode = TextOptimizationMode.Clean
    image_mode: ImageOptimizationMode = ImageOptimizationMode.Ocr
    json_paths_to_keep: List[str] = field(default_factory=list)
    long_text_threshold: int = 500

# endregion

# region == Extensibility: Service Protocols ==

class ITextOptimizer(Protocol):
    """Interface for a service that optimizes unstructured text."""
    
    async def optimize_text(self, long_text: str, config: BrevitConfig) -> str:
        """Optimize long text (e.g., via summarization)."""
        ...

class IImageOptimizer(Protocol):
    """Interface for a service that optimizes image data."""
    
    async def optimize_image(self, image_data: bytes, config: BrevitConfig) -> str:
        """Optimize image data (e.g., via OCR)."""
        ...

class DefaultTextOptimizer:
    """STUB implementation. Replace this by registering your own service."""
    
    async def optimize_text(self, long_text: str, config: BrevitConfig) -> str:
        if config.text_mode == TextOptimizationMode.NONE:
            return long_text
        # TODO: Implement with LangChain, Semantic Kernel for Python, or a direct model call.
        print("[Brevit] STUB: Text summarization (e.g., via LangChain) is not implemented.")
        mode = config.text_mode.name
        stub_summary = long_text[:150]
        return f"[{mode} Stub: Summary of text follows...]\n{stub_summary}...\n[End of summary]"

class DefaultImageOptimizer:
    """STUB implementation. Replace this by registering your own service."""
    
    async def optimize_image(self, image_data: bytes, config: BrevitConfig) -> str:
        if config.image_mode == ImageOptimizationMode.NONE:
            return ""
        # TODO: Implement with Azure AI Vision SDK, Tesseract, or other OCR tool.
        print("[Brevit] STUB: Image OCR (e.g., via Azure AI) is not implemented.")
        result = (
            f"[OCR Stub: Extracted text from image ({len(image_data)} bytes)]\n"
            "Sample OCR Text: INVOICE #1234\n"
            "Total: $499.99\n"
            "[End of extracted text]"
        )
        return result

# endregion

# region == Core Class: BrevitClient ==

class BrevitClient:
    """
    The main client for the Brevit.py library.
    This class orchestrates the optimization pipeline.
    """
    
    def __init__(
        self,
        config: BrevitConfig,
        text_optimizer: Optional[ITextOptimizer] = None,
        image_optimizer: Optional[IImageOptimizer] = None,
    ):
        self._config = config
        # Use default stub optimizers if none are provided
        self._text_optimizer = text_optimizer or DefaultTextOptimizer()
        self._image_optimizer = image_optimizer or DefaultImageOptimizer()
        self._strategies: Dict[str, Any] = {}  # Registry for custom strategies

    def _is_uniform_object_array(self, arr: list) -> tuple[list[str] | None, bool]:
        """Checks if an array contains uniform objects (all have same keys)."""
        if not arr or not isinstance(arr, list):
            return None, False
        
        first_item = arr[0]
        if not isinstance(first_item, dict):
            return None, False
        
        # Preserve original field order instead of sorting
        first_keys = list(first_item.keys())
        first_key_set = set(first_keys)
        
        # Check if all items have the same keys (order-independent)
        for item in arr[1:]:
            if not isinstance(item, dict):
                return None, False
            item_keys = list(item.keys())
            if len(first_keys) != len(item_keys):
                return None, False
            # Check if all keys exist (order doesn't matter for uniformity)
            if not all(key in first_key_set for key in item_keys):
                return None, False
        
        return first_keys, True

    def _is_primitive_array(self, arr: list) -> bool:
        """Checks if an array contains only primitives."""
        if not arr or not isinstance(arr, list):
            return False
        
        for item in arr:
            if isinstance(item, (dict, list)):
                return False
        
        return True

    def _escape_value(self, value: Any) -> str:
        """Escapes a value for comma-separated format."""
        if value is None:
            return "null"
        
        str_value = str(value)
        
        # Quote if contains comma, newline, or quotes
        if ',' in str_value or '\n' in str_value or '"' in str_value:
            return f'"{str_value.replace('"', '\\"')}"'
        
        return str_value

    def _format_tabular_array(self, arr: list, prefix: str, keys: list[str]) -> str:
        """Formats a uniform object array in tabular format."""
        header = f"{prefix}[{len(arr)}]{{{','.join(keys)}}}:"
        rows = []
        
        for item in arr:
            values = [self._escape_value(item.get(key)) for key in keys]
            rows.append(','.join(values))
        
        return f"{header}\n" + "\n".join(rows)

    def _format_primitive_array(self, arr: list, prefix: str) -> str:
        """Formats a primitive array in comma-separated format."""
        values = [self._escape_value(item) for item in arr]
        return f"{prefix}[{len(arr)}]:{','.join(values)}"

    def _flatten(self, node: Any, prefix: str, output: list[str]):
        """Recursive helper for flattening a JSON object/array with tabular optimization."""
        if isinstance(node, dict):
            for key, value in node.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten(value, new_prefix, output)
        elif isinstance(node, list):
            # Check for uniform object array (tabular format)
            keys, is_uniform = self._is_uniform_object_array(node)
            if is_uniform and keys:
                output.append(self._format_tabular_array(node, prefix, keys))
                return
            
            # Check for primitive array (comma-separated format)
            if self._is_primitive_array(node):
                output.append(self._format_primitive_array(node, prefix))
                return
            
            # Fall back to current format for mixed/non-uniform arrays
            for i, item in enumerate(node):
                new_prefix = f"{prefix}[{i}]"
                self._flatten(item, new_prefix, output)
        else:
            # It's a primitive value (str, int, float, bool, None)
            if not prefix:
                prefix = "value"  # Handle root-level value
            output.append(f"{prefix}:{str(node)}")

    def _flatten_object(self, obj: Any) -> str:
        """Flattens a Python dict/list into a token-efficient string with tabular optimization."""
        output: list[str] = []
        self._flatten(obj, "", output)
        return "\n".join(output)

    def _analyze_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyzes data structure to determine the best optimization strategy."""
        analysis = {
            'type': 'unknown',
            'depth': 0,
            'has_uniform_arrays': False,
            'has_primitive_arrays': False,
            'has_nested_objects': False,
            'text_length': 0,
            'array_count': 0,
            'object_count': 0,
            'complexity': 'simple'
        }
        
        def analyze_node(node: Any, depth: int = 0) -> None:
            analysis['depth'] = max(analysis['depth'], depth)
            
            if isinstance(node, str):
                analysis['text_length'] += len(node)
            elif isinstance(node, list):
                analysis['array_count'] += 1
                
                # Check for uniform object arrays
                keys, is_uniform = self._is_uniform_object_array(node)
                if is_uniform:
                    analysis['has_uniform_arrays'] = True
                
                # Check for primitive arrays
                if self._is_primitive_array(node):
                    analysis['has_primitive_arrays'] = True
                
                # Analyze each element
                for item in node:
                    analyze_node(item, depth + 1)
            elif isinstance(node, dict):
                analysis['object_count'] += 1
                if depth > 0:
                    analysis['has_nested_objects'] = True
                
                for value in node.values():
                    analyze_node(value, depth + 1)
        
        analyze_node(data)
        
        # Determine complexity
        if analysis['depth'] > 3 or analysis['array_count'] > 5 or analysis['object_count'] > 10:
            analysis['complexity'] = 'complex'
        elif analysis['depth'] > 1 or analysis['array_count'] > 0 or analysis['object_count'] > 3:
            analysis['complexity'] = 'moderate'
        
        # Determine type
        if isinstance(data, str):
            analysis['type'] = 'longText' if len(data) > self._config.long_text_threshold else 'text'
        elif isinstance(data, bytes):
            analysis['type'] = 'image'
        elif isinstance(data, list):
            analysis['type'] = 'array'
        elif isinstance(data, dict):
            analysis['type'] = 'object'
        else:
            analysis['type'] = 'primitive'
        
        return analysis

    def _select_optimal_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Selects the best optimization strategy based on data analysis."""
        strategies = []
        
        # Strategy 1: Flatten with tabular optimization
        if analysis['has_uniform_arrays'] or analysis['has_primitive_arrays']:
            strategies.append({
                'name': 'Flatten',
                'json_mode': JsonOptimizationMode.Flatten,
                'score': 100 if analysis['has_uniform_arrays'] else 80,
                'reason': 'Uniform object arrays detected - tabular format optimal' if analysis['has_uniform_arrays'] 
                         else 'Primitive arrays detected - comma-separated format optimal'
            })
        
        # Strategy 2: Standard flatten
        if analysis['has_nested_objects'] or analysis['complexity'] == 'moderate':
            strategies.append({
                'name': 'Flatten',
                'json_mode': JsonOptimizationMode.Flatten,
                'score': 70,
                'reason': 'Nested objects detected - flatten format optimal'
            })
        
        # Strategy 3: Text optimization
        if analysis['type'] == 'longText':
            strategies.append({
                'name': 'TextOptimization',
                'text_mode': self._config.text_mode,
                'score': 90,
                'reason': 'Long text detected - summarization recommended'
            })
        
        # Strategy 4: Image optimization
        if analysis['type'] == 'image':
            strategies.append({
                'name': 'ImageOptimization',
                'image_mode': self._config.image_mode,
                'score': 100,
                'reason': 'Image data detected - OCR recommended'
            })
        
        # Select highest scoring strategy
        if not strategies:
            return {
                'name': 'Flatten',
                'json_mode': JsonOptimizationMode.Flatten,
                'score': 50,
                'reason': 'Default flatten strategy'
            }
        
        return max(strategies, key=lambda s: s['score'])

    async def brevity(self, raw_data: Any, intent: Optional[str] = None) -> str:
        """
        Intelligently optimizes data by automatically selecting the best strategy.
        This method analyzes the input data structure and applies the most
        appropriate optimization methods automatically.
        
        :param raw_data: The data to optimize (dict, list, JSON str, text, bytes).
        :param intent: (Optional) A hint about the user's goal.
        :return: A single string optimized for an LLM prompt.
        """
        # Handle image data immediately
        if isinstance(raw_data, bytes):
            return await self._image_optimizer.optimize_image(raw_data, self._config)
        
        # Handle text
        if isinstance(raw_data, str):
            try:
                trimmed = raw_data.strip()
                if trimmed.startswith('{') or trimmed.startswith('['):
                    input_object = json.loads(raw_data)
                else:
                    # Plain text
                    analysis = self._analyze_data_structure(raw_data)
                    strategy = self._select_optimal_strategy(analysis)
                    
                    if strategy['name'] == 'TextOptimization':
                        return await self._text_optimizer.optimize_text(raw_data, self._config)
                    return raw_data
            except json.JSONDecodeError:
                # Not JSON - treat as text
                analysis = self._analyze_data_structure(raw_data)
                strategy = self._select_optimal_strategy(analysis)
                
                if strategy['name'] == 'TextOptimization':
                    return await self._text_optimizer.optimize_text(raw_data, self._config)
                return raw_data
        
        # Handle dict/list
        if isinstance(raw_data, (dict, list)):
            analysis = self._analyze_data_structure(raw_data)
            strategy = self._select_optimal_strategy(analysis)
            
            # Create temporary config with selected strategy
            temp_config = BrevitConfig(
                json_mode=strategy.get('json_mode', self._config.json_mode),
                text_mode=strategy.get('text_mode', self._config.text_mode),
                image_mode=strategy.get('image_mode', self._config.image_mode),
                long_text_threshold=self._config.long_text_threshold,
                json_paths_to_keep=self._config.json_paths_to_keep
            )
            
            # Temporarily override config
            original_config = self._config
            self._config = temp_config
            
            try:
                return await self.optimize(raw_data, intent)
            finally:
                self._config = original_config
        
        # Fallback to standard optimization
        return await self.optimize(raw_data, intent)

    def register_strategy(self, name: str, analyzer: Any, optimizer: Any) -> None:
        """
        Registers a custom optimization strategy for the brevity method.
        This allows extending Brevit with new optimization strategies.
        
        :param name: Strategy name
        :param analyzer: Function that analyzes data and returns score (0-100)
        :param optimizer: Function that optimizes the data
        """
        self._strategies[name] = {'analyzer': analyzer, 'optimizer': optimizer}

    async def optimize(self, raw_data: Any, intent: Optional[str] = None) -> str:
        """
        The primary method. Optimizes any Python object, JSON string,
        or data (text, image bytes) into a token-efficient string.
        
        :param raw_data: The data to optimize (dict, list, JSON str, text, bytes).
        :param intent: (Optional) A hint about the user's goal.
        :return: A single string optimized for an LLM prompt.
        """
        
        input_object = None
        
        if isinstance(raw_data, str):
            # Could be JSON string or just text
            try:
                trimmed = raw_data.strip()
                if trimmed.startswith("{") or trimmed.startswith("["):
                    input_object = json.loads(raw_data)
            except json.JSONDecodeError:
                # It's not a JSON string, treat as text
                pass
            
            if input_object is None:
                # It's text
                if len(raw_data) > self._config.long_text_threshold:
                    # It's long text
                    return await self._text_optimizer.optimize_text(raw_data, self._config)
                # It's short text
                return raw_data
        
        elif isinstance(raw_data, (dict, list)):
            # It's a plain Python object
            input_object = raw_data
            
        elif isinstance(raw_data, bytes):
            # It's image data
            return await self._image_optimizer.optimize_image(raw_data, self._config)
            
        else:
            # Other primitives
            return str(raw_data)

        # If we're here, we have an object (from JSON or dict/list)
        # Now apply the configured JSON optimization
        mode = self._config.json_mode
        
        if mode == JsonOptimizationMode.Flatten:
            return self._flatten_object(input_object)
            
        elif mode == JsonOptimizationMode.ToYaml:
            # STUB: Requires a 'PyYAML' library
            # import yaml
            # return yaml.dump(input_object)
            print("[Brevit] ToYaml mode requires 'pip install PyYAML'.")
            return json.dumps(input_object, indent=2)  # Fallback
            
        elif mode == JsonOptimizationMode.Filter:
            # STUB: Requires a JSON-path library
            print("[Brevit] Filter mode is not implemented in this stub.")
            return json.dumps(input_object)  # Fallback
            
        elif mode == JsonOptimizationMode.NONE:
            return json.dumps(input_object)  # Return as unformatted JSON
        
        return json.dumps(input_object)

# endregion

