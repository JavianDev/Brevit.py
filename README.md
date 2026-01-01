# Brevit.py

A high-performance Python library for semantically compressing and optimizing data before sending it to a Large Language Model (LLM). Dramatically reduce token costs while maintaining data integrity and readability.

## Table of Contents

- [Why Brevit.py?](#why-brevitpy)
- [Key Features](#key-features)
- [When Not to Use Brevit.py](#when-not-to-use-brevitpy)
- [Benchmarks](#benchmarks)
- [Installation & Quick Start](#installation--quick-start)
- [Playgrounds](#playgrounds)
- [CLI](#cli)
- [Format Overview](#format-overview)
- [API](#api)
- [Using Brevit.py in LLM Prompts](#using-brevitpy-in-llm-prompts)
- [Syntax Cheatsheet](#syntax-cheatsheet)
- [Other Implementations](#other-implementations)
- [Full Specification](#full-specification)

## Why Brevit.py?

### Python-Specific Advantages

- **Async/Await**: Built with modern Python async/await patterns
- **Type Hints**: Full type annotations for better IDE support
- **LangChain Integration**: Ready for LangChain workflows
- **FastAPI/Flask Compatible**: Works seamlessly with popular web frameworks
- **Pydantic Support**: Integrates with Pydantic models

### Performance Benefits

- **40-60% Token Reduction**: Dramatically reduce LLM API costs
- **Async Operations**: Non-blocking I/O for better concurrency
- **Memory Efficient**: Processes data in-place where possible
- **Fast Execution**: Optimized algorithms for minimal overhead

### Example Cost Savings

```python
# Before: 234 tokens = $0.000468 per request
json_str = json.dumps(complex_order)

# After: 127 tokens = $0.000254 per request (46% reduction)
optimized = await brevit.brevity(complex_order)  # Automatic optimization

# Or with explicit configuration
explicit = await brevit.optimize(complex_order)

# Savings: $0.000214 per request
# At 1M requests/month: $214/month savings
```

### Automatic Strategy Selection

Brevit.py now includes the `.brevity()` method that automatically analyzes your data and selects the optimal optimization strategy:

```python
data = {
    "friends": ["ana", "luis", "sam"],
    "hikes": [
        {"id": 1, "name": "Blue Lake Trail", "distanceKm": 7.5},
        {"id": 2, "name": "Ridge Overlook", "distanceKm": 9.2}
    ]
}

# Automatically detects uniform arrays and applies tabular format
optimized = await brevit.brevity(data)
# No configuration needed - Brevit analyzes and optimizes automatically!
```

## Key Features

- **JSON Optimization**: Flatten nested JSON structures into token-efficient key-value pairs
- **Text Optimization**: Clean and summarize long text documents
- **Image Optimization**: Extract text from images via OCR
- **Async/Await**: Built with modern Python async/await patterns
- **Extensible**: Plugin architecture for custom optimizers
- **Lightweight**: Minimal dependencies, high performance
- **Type Hints**: Full type annotations for better IDE support

## Installation

### Prerequisites

- Python 3.8 or later
- pip or poetry

### Install via pip

```bash
pip install brevit
```

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/JavianDev/Brevit.py.git
cd Brevit.py
```

2. Install in development mode:
```bash
pip install -e .
```

### Optional Dependencies

For YAML support:
```bash
pip install brevit[yaml]
# or
pip install PyYAML
```

For JSON path filtering:
```bash
pip install brevit[jsonpath]
# or
pip install jsonpath-ng
```

## Quick Start

### Basic Usage

```python
from brevit import BrevitClient, BrevitConfig, JsonOptimizationMode
import asyncio

async def main():
    # 1. Create configuration
    config = BrevitConfig(
        json_mode=JsonOptimizationMode.Flatten,
        text_mode=TextOptimizationMode.Clean,
        image_mode=ImageOptimizationMode.Ocr,
        long_text_threshold=1000  # Summarize text over 1000 chars
    )

    # 2. Create client
    brevit = BrevitClient(config)

    # 3. Optimize data
    order = {
        "orderId": "o-456",
        "status": "SHIPPED",
        "items": [
            {"sku": "A-88", "name": "Brevit Pro License", "quantity": 1}
        ]
    }

    optimized = await brevit.optimize(order)
    # Result (with abbreviations enabled by default):
    # "@o=order\n@o.orderId:o-456\n@o.status:SHIPPED\n@o.items[1]{name,quantity,sku}:\nBrevit Pro License,1,A-88"
    print(optimized)

asyncio.run(main())
```

#### Abbreviation Feature (New in v0.1.2)

Brevit automatically creates abbreviations for frequently repeated prefixes, reducing token usage by 10-25%:

```python
from brevit import BrevitClient, BrevitConfig, JsonOptimizationMode

config = BrevitConfig(
    json_mode=JsonOptimizationMode.Flatten,
    enable_abbreviations=True,    # Enabled by default
    abbreviation_threshold=2           # Minimum occurrences to abbreviate
)
brevit = BrevitClient(config)

data = {
    "user": {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    },
    "order": {
        "id": "o-456",
        "status": "SHIPPED"
    }
}

optimized = await brevit.brevity(data)
# Output with abbreviations:
# @u=user
# @o=order
# @u.name:John Doe
# @u.email:john@example.com
# @u.age:30
# @o.id:o-456
# @o.status:SHIPPED
```

**Token Savings**: The abbreviation feature reduces tokens by replacing repeated prefixes like "user." and "order." with short aliases like "@u" and "@o", saving 10-25% on typical nested JSON structures.

## Complete Usage Examples

Brevit.py supports three main data types: **JSON objects/strings**, **text files/strings**, and **images**. Here's how to use each:

### 1. JSON Optimization Examples

#### Example 1.1: Simple JSON Object

```python
from brevit import BrevitClient, BrevitConfig, JsonOptimizationMode

brevit = BrevitClient(BrevitConfig(json_mode=JsonOptimizationMode.Flatten))

data = {
    "user": {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    }
}

# Method 1: Automatic optimization (recommended)
optimized = await brevit.brevity(data)
# Output (with abbreviations enabled by default):
# @u=user
# @u.name:John Doe
# @u.email:john@example.com
# @u.age:30

# Method 2: Explicit optimization
explicit = await brevit.optimize(data)
```

#### Example 1.2: JSON String

```python
json_string = '{"order": {"id": "o-456", "status": "SHIPPED"}}'

# Brevit automatically detects JSON strings
optimized = await brevit.brevity(json_string)
# Output (with abbreviations enabled by default):
# @o=order
# @o.id:o-456
# @o.status:SHIPPED
```

#### Example 1.2a: Abbreviations Disabled

```python
config_no_abbr = BrevitConfig(
    json_mode=JsonOptimizationMode.Flatten,
    enable_abbreviations=False  # Disable abbreviations
)
brevit_no_abbr = BrevitClient(config_no_abbr)

json_string = '{"order": {"id": "o-456", "status": "SHIPPED"}}'
optimized = await brevit_no_abbr.brevity(json_string)
# Output (without abbreviations):
# order.id:o-456
# order.status:SHIPPED
```

#### Example 1.3: Complex Nested JSON with Arrays

```python
complex_data = {
    "context": {
        "task": "Our favorite hikes together",
        "location": "Boulder",
        "season": "spring_2025"
    },
    "friends": ["ana", "luis", "sam"],
    "hikes": [
        {
            "id": 1,
            "name": "Blue Lake Trail",
            "distanceKm": 7.5,
            "elevationGain": 320,
            "companion": "ana",
            "wasSunny": True
        },
        {
            "id": 2,
            "name": "Ridge Overlook",
            "distanceKm": 9.2,
            "elevationGain": 540,
            "companion": "luis",
            "wasSunny": False
        }
    ]
}

optimized = await brevit.brevity(complex_data)
# Output (with abbreviations enabled by default):
# @c=context
# @c.task:Our favorite hikes together
# @c.location:Boulder
# @c.season:spring_2025
# friends[3]:ana,luis,sam
# hikes[2]{companion,distanceKm,elevationGain,id,name,wasSunny}:
# ana,7.5,320,1,Blue Lake Trail,True
# luis,9.2,540,2,Ridge Overlook,False
```

#### Example 1.3a: Complex Data with Abbreviations Disabled

```python
config_no_abbr = BrevitConfig(
    json_mode=JsonOptimizationMode.Flatten,
    enable_abbreviations=False  # Disable abbreviations
)
brevit_no_abbr = BrevitClient(config_no_abbr)

complex_data = {
    "context": {
        "task": "Our favorite hikes together",
        "location": "Boulder",
        "season": "spring_2025"
    },
    "friends": ["ana", "luis", "sam"],
    "hikes": [
        {
            "id": 1,
            "name": "Blue Lake Trail",
            "distanceKm": 7.5,
            "elevationGain": 320,
            "companion": "ana",
            "wasSunny": True
        },
        {
            "id": 2,
            "name": "Ridge Overlook",
            "distanceKm": 9.2,
            "elevationGain": 540,
            "companion": "luis",
            "wasSunny": False
        }
    ]
}

optimized = await brevit_no_abbr.brevity(complex_data)
# Output (without abbreviations):
# context.task:Our favorite hikes together
# context.location:Boulder
# context.season:spring_2025
# friends[3]:ana,luis,sam
# hikes[2]{companion,distanceKm,elevationGain,id,name,wasSunny}:
# ana,7.5,320,1,Blue Lake Trail,True
# luis,9.2,540,2,Ridge Overlook,False
```

#### Example 1.4: Different JSON Optimization Modes

```python
# Flatten Mode (Default)
flatten_config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
# Converts nested JSON to flat key-value pairs

# YAML Mode
yaml_config = BrevitConfig(json_mode=JsonOptimizationMode.ToYaml)
# Converts JSON to YAML format (requires PyYAML)

# Filter Mode
filter_config = BrevitConfig(
    json_mode=JsonOptimizationMode.Filter,
    json_paths_to_keep=["user.name", "order.id"]
)
# Keeps only specified paths, removes everything else
```

### 2. Text Optimization Examples

#### Example 2.1: Long Text String

```python
long_text = "This is a very long document..." * 100

config = BrevitConfig(
    json_mode=JsonOptimizationMode.None,
    text_mode=TextOptimizationMode.Clean,
    long_text_threshold=500
)
brevit = BrevitClient(config)

# Automatic detection
optimized = await brevit.brevity(long_text)

# Explicit text optimization
cleaned = await brevit.optimize(long_text)
```

#### Example 2.2: Reading Text from File

```python
# Read text file
with open('document.txt', 'r', encoding='utf-8') as f:
    text_content = f.read()

# Optimize the text
optimized = await brevit.brevity(text_content)
```

#### Example 2.3: Text Optimization Modes

```python
# Clean Mode (Remove Boilerplate)
clean_config = BrevitConfig(text_mode=TextOptimizationMode.Clean)
# Removes signatures, headers, repetitive content

# Summarize Fast
fast_config = BrevitConfig(text_mode=TextOptimizationMode.SummarizeFast)
# Fast summarization (requires custom text optimizer implementation)

# Summarize High Quality
quality_config = BrevitConfig(text_mode=TextOptimizationMode.SummarizeHighQuality)
# High-quality summarization (requires custom text optimizer with LLM integration)
```

### 3. Image Optimization Examples

#### Example 3.1: Image from File (OCR)

```python
# Read image file
with open('receipt.jpg', 'rb') as f:
    image_bytes = f.read()

# Brevit automatically detects bytes as image data
extracted_text = await brevit.brevity(image_bytes)
# Output: OCR-extracted text from the image
```

#### Example 3.2: Image from URL

```python
import requests

# Fetch image from URL
response = requests.get('https://example.com/invoice.png')
image_bytes = response.content

# Optimize image
extracted_text = await brevit.brevity(image_bytes)
```

#### Example 3.3: Image Optimization Modes

```python
# OCR Mode (Extract Text)
ocr_config = BrevitConfig(image_mode=ImageOptimizationMode.Ocr)
# Extracts text from images using OCR (requires custom image optimizer)

# Metadata Mode
metadata_config = BrevitConfig(image_mode=ImageOptimizationMode.Metadata)
# Extracts only image metadata (dimensions, format, etc.)
```

### 4. Method Comparison: `.brevity()` vs `.optimize()`

#### `.brevity()` - Automatic Strategy Selection

**Use when:** You want Brevit to automatically analyze and select the best optimization strategy.

```python
# Automatically detects data type and applies optimal strategy
result = await brevit.brevity(data)
# - JSON objects → Flatten with tabular optimization
# - Long text → Text optimization
# - Images → OCR extraction
```

**Advantages:**
- Zero configuration needed
- Intelligent strategy selection
- Works with any data type
- Best for general-purpose use

#### `.optimize()` - Explicit Configuration

**Use when:** You want explicit control over optimization mode.

```python
config = BrevitConfig(
    json_mode=JsonOptimizationMode.Flatten,
    text_mode=TextOptimizationMode.Clean,
    image_mode=ImageOptimizationMode.Ocr
)
brevit = BrevitClient(config)

# Uses explicit configuration
result = await brevit.optimize(data)
```

**Advantages:**
- Full control over optimization
- Predictable behavior
- Best for specific use cases

### 5. Custom Optimizers

You can provide custom optimizers for text and images:

```python
# Custom text optimizer
class CustomTextOptimizer:
    async def optimize_text(self, text: str, config: BrevitConfig) -> str:
        # Call your summarization service
        return await summarize_service.summarize(text)

# Custom image optimizer
class CustomImageOptimizer:
    async def optimize_image(self, image_data: bytes, config: BrevitConfig) -> str:
        # Call your OCR service (e.g., Azure AI Vision, Tesseract)
        return await ocr_service.extract_text(image_data)

brevit = BrevitClient(
    config,
    text_optimizer=CustomTextOptimizer(),
    image_optimizer=CustomImageOptimizer()
)
```

### 6. Complete Workflow Examples

#### Example 6.1: E-Commerce Order Processing

```python
# Step 1: Optimize order JSON
order = {
    "orderId": "o-456",
    "customer": {"name": "John", "email": "john@example.com"},
    "items": [
        {"sku": "A-88", "quantity": 2, "price": 29.99},
        {"sku": "B-22", "quantity": 1, "price": 49.99}
    ]
}

optimized_order = await brevit.brevity(order)

# Step 2: Send to LLM
prompt = f"Analyze this order:\n\n{optimized_order}\n\nExtract total amount."
# Send prompt to OpenAI, Anthropic, etc.
```

#### Example 6.2: Document Processing Pipeline

```python
# Step 1: Read and optimize text document
with open('contract.txt', 'r') as f:
    contract_text = f.read()

optimized_text = await brevit.brevity(contract_text)

# Step 2: Process with LLM
prompt = f"Summarize this contract:\n\n{optimized_text}"
# Send to LLM for summarization
```

#### Example 6.3: Receipt OCR Pipeline

```python
# Step 1: Read receipt image
with open('receipt.jpg', 'rb') as f:
    receipt_image = f.read()

# Step 2: Extract text via OCR
extracted_text = await brevit.brevity(receipt_image)

# Step 3: Optimize extracted text (if it's long)
optimized = await brevit.brevity(extracted_text)

# Step 4: Send to LLM for analysis
prompt = f"Extract items and total from this receipt:\n\n{optimized}"
# Send to LLM
```

### Flask/FastAPI Example

```python
from flask import Flask, request, jsonify
from brevit import BrevitClient, BrevitConfig, JsonOptimizationMode

app = Flask(__name__)

# Initialize Brevit client
config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
brevit = BrevitClient(config)

@app.route('/optimize', methods=['POST'])
async def optimize_data():
    data = request.json
    
    # Optimize the data
    optimized = await brevit.optimize(data)
    
    # Send to LLM API
    prompt = f"Context:\n{optimized}\n\nTask: Summarize the data."
    
    # response = await call_llm_api(prompt)
    
    return jsonify({"optimized": optimized, "prompt": prompt})

if __name__ == '__main__':
    app.run()
```

### FastAPI Example

```python
from fastapi import FastAPI
from brevit import BrevitClient, BrevitConfig, JsonOptimizationMode
from pydantic import BaseModel

app = FastAPI()

config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
brevit = BrevitClient(config)

class OrderData(BaseModel):
    orderId: str
    status: str
    items: list

@app.post("/optimize")
async def optimize_order(order: OrderData):
    optimized = await brevit.optimize(order.dict())
    return {"optimized": optimized}
```

## Configuration Options

### BrevitConfig

```python
config = BrevitConfig(
    json_mode=JsonOptimizationMode.Flatten,      # JSON optimization strategy
    text_mode=TextOptimizationMode.Clean,        # Text optimization strategy
    image_mode=ImageOptimizationMode.Ocr,        # Image optimization strategy
    json_paths_to_keep=[],                       # Paths to keep for Filter mode
    long_text_threshold=500,                     # Character threshold for text optimization
    enable_abbreviations=True,                    # Enable abbreviation feature (default: True)
    abbreviation_threshold=2                      # Minimum occurrences to create abbreviation (default: 2)
)
```

### JsonOptimizationMode

- **NONE**: No optimization, pass JSON as-is
- **Flatten**: Convert nested JSON to flat key-value pairs (most token-efficient)
- **ToYaml**: Convert JSON to YAML format (requires PyYAML)
- **Filter**: Keep only specified JSON paths

### TextOptimizationMode

- **NONE**: No optimization
- **Clean**: Remove boilerplate and excessive whitespace
- **SummarizeFast**: Use a fast model for summarization (requires custom ITextOptimizer)
- **SummarizeHighQuality**: Use a high-quality model for summarization (requires custom ITextOptimizer)

### ImageOptimizationMode

- **NONE**: Skip image processing
- **Ocr**: Extract text from images (requires custom IImageOptimizer)
- **Metadata**: Extract basic metadata only

## Advanced Usage

### Custom Text Optimizer

Implement `ITextOptimizer` to use LangChain or your own LLM service:

```python
from brevit import ITextOptimizer, BrevitConfig
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LangChainTextOptimizer:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text: {text}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def optimize_text(self, long_text: str, config: BrevitConfig) -> str:
        result = await self.chain.arun(text=long_text)
        return result

# Use custom optimizer
config = BrevitConfig(text_mode=TextOptimizationMode.SummarizeFast)
brevit = BrevitClient(config, text_optimizer=LangChainTextOptimizer())
```

### Custom Image Optimizer

Implement `IImageOptimizer` to use Azure AI Vision or Tesseract:

```python
from brevit import IImageOptimizer, BrevitConfig
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

class AzureVisionImageOptimizer:
    def __init__(self, endpoint: str, key: str):
        self.client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    async def optimize_image(self, image_data: bytes, config: BrevitConfig) -> str:
        result = self.client.analyze(
            image_data=image_data,
            visual_features=["read"]
        )
        return result.read.text

# Use custom optimizer
config = BrevitConfig(image_mode=ImageOptimizationMode.Ocr)
brevit = BrevitClient(
    config,
    image_optimizer=AzureVisionImageOptimizer(endpoint="...", key="...")
)
```

### Using Tesseract OCR

```python
from brevit import IImageOptimizer, BrevitConfig
from PIL import Image
import pytesseract
import io

class TesseractImageOptimizer:
    async def optimize_image(self, image_data: bytes, config: BrevitConfig) -> str:
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text

config = BrevitConfig(image_mode=ImageOptimizationMode.Ocr)
brevit = BrevitClient(config, image_optimizer=TesseractImageOptimizer())
```

### YAML Mode

To use YAML mode, install PyYAML:

```bash
pip install PyYAML
```

Then update the `ToYaml` case in `brevit.py`:

```python
import yaml

# In the optimize method:
elif mode == JsonOptimizationMode.ToYaml:
    return yaml.dump(input_object)
```

### Filter Mode

Use Filter mode to keep only specific JSON paths:

```python
config = BrevitConfig(
    json_mode=JsonOptimizationMode.Filter,
    json_paths_to_keep=[
        "user.name",
        "order.orderId",
        "order.items[*].sku"
    ]
)
```

## Examples

### Example 1: Optimize Complex Object

```python
user = {
    "id": "u-123",
    "name": "Javian",
    "isActive": True,
    "contact": {
        "email": "support@javianpicardo.com",
        "phone": None
    },
    "orders": [
        {"orderId": "o-456", "status": "SHIPPED"}
    ]
}

optimized = await brevit.optimize(user)
# Output:
# id: u-123
# name: Javian
# isActive: True
# contact.email: support@javianpicardo.com
# contact.phone: None
# orders[0].orderId: o-456
# orders[0].status: SHIPPED
```

### Example 2: Optimize JSON String

```python
json_str = """{
    "order": {
        "orderId": "o-456",
        "status": "SHIPPED",
        "items": [
            {"sku": "A-88", "name": "Brevit Pro", "quantity": 1}
        ]
    }
}"""

optimized = await brevit.optimize(json_str)
```

### Example 3: Process Long Text

```python
with open("document.txt", "r") as f:
    long_document = f.read()

optimized = await brevit.optimize(long_document)
# Will trigger text optimization if length > long_text_threshold
```

### Example 4: Process Image

```python
with open("receipt.jpg", "rb") as f:
    image_data = f.read()

optimized = await brevit.optimize(image_data)
# Will trigger image optimization
```

## When Not to Use Brevit.py

Consider alternatives when:

1. **API Responses**: If returning JSON to HTTP clients, use standard JSON
2. **Data Contracts**: When strict JSON schema validation is required
3. **Small Objects**: Objects under 100 tokens may not benefit significantly
4. **Real-Time APIs**: For REST APIs serving JSON, standard formatting is better
5. **Database Storage**: Databases expect standard JSON format

**Best Use Cases:**
- ✅ LLM prompt optimization
- ✅ Reducing OpenAI/Anthropic API costs
- ✅ Processing large datasets for AI
- ✅ Document summarization workflows
- ✅ OCR and image processing pipelines
- ✅ LangChain integrations

## Benchmarks

### Token Reduction

| Object Type | Original Tokens | Brevit (No Abbr) | Brevit (With Abbr) | Total Reduction |
|-------------|----------------|------------------|-------------------|-----------------|
| Simple Dict | 45 | 28 | 26 | 42% |
| Complex Dict | 234 | 127 | 105 | 55% |
| Nested Lists | 156 | 89 | 75 | 52% |
| API Response | 312 | 178 | 145 | 54% |
| Deeply Nested | 95 | 78 | 65 | 32% |

**Note**: Abbreviations are enabled by default and provide additional 10-25% savings on top of base optimization.

### Performance

| Operation | Objects/sec | Avg Latency | Memory |
|-----------|-------------|-------------|--------|
| Flatten (1KB) | 1,600 | 0.6ms | 2.1MB |
| Flatten (10KB) | 380 | 2.6ms | 8.5MB |
| Flatten (100KB) | 48 | 21ms | 45MB |

*Benchmarks: Python 3.11, Intel i7-12700K, asyncio*

## Playgrounds

### Interactive Playground

```bash
# Clone and run
git clone https://github.com/JavianDev/Brevit.git
cd Brevit/Brevit.py
pip install -e .
python playground.py
```

### Online Playground

- **Web Playground**: [https://brevit.dev/playground](https://brevit.dev/playground) (Coming Soon)
- **Replit**: [https://replit.com/@brevit/playground](https://replit.com/@brevit/playground) (Coming Soon)
- **Colab**: [https://colab.research.google.com/brevit](https://colab.research.google.com/brevit) (Coming Soon)

## CLI

### Installation

```bash
pip install brevit-cli
```

### Usage

```bash
# Optimize a JSON file
brevit optimize input.json -o output.txt

# Optimize from stdin
cat data.json | brevit optimize

# Optimize with custom config
brevit optimize input.json --mode flatten --threshold 1000

# Help
brevit --help
```

### Examples

```bash
# Flatten JSON
brevit optimize order.json --mode flatten

# Convert to YAML
brevit optimize data.json --mode yaml

# Filter paths
brevit optimize data.json --mode filter --paths "user.name,order.id"
```

## Format Overview

### Flattened Format (Hybrid Optimization)

Brevit intelligently converts Python dictionaries to flat key-value pairs with automatic tabular optimization:

**Input:**
```python
order = {
    "orderId": "o-456",
    "friends": ["ana", "luis", "sam"],
    "items": [
        {"sku": "A-88", "quantity": 1},
        {"sku": "T-22", "quantity": 2}
    ]
}
```

**Output (with tabular optimization and abbreviations enabled by default):**
```
orderId: o-456
friends[3]: ana,luis,sam
@i=items
@i[2]{quantity,sku}:
1,A-88
2,T-22
```

**Output (with abbreviations disabled):**
```
orderId: o-456
friends[3]: ana,luis,sam
items[2]{quantity,sku}:
1,A-88
2,T-22
```

**For non-uniform arrays (fallback):**
```python
mixed = {
    "items": [
        {"sku": "A-88", "quantity": 1},
        "special-item",
        {"sku": "T-22", "quantity": 2}
    ]
}
```

**Output (fallback to indexed format):**
```
items[0].sku: A-88
items[0].quantity: 1
items[1]: special-item
items[2].sku: T-22
items[2].quantity: 2
```

### Key Features

- **Dictionary Keys**: Uses Python dictionary keys as-is
- **Nested Dicts**: Dot notation for nested dictionaries
- **Tabular Arrays**: Uniform object arrays automatically formatted in compact tabular format (`items[2]{field1,field2}:`)
- **Primitive Arrays**: Comma-separated format (`friends[3]: ana,luis,sam`)
- **Abbreviation System** (Default: Enabled): Automatically creates short aliases for repeated prefixes (`@u=user`, `@o=order`)
- **Hybrid Approach**: Automatically detects optimal format, falls back to indexed format for mixed data
- **None Handling**: Explicit `None` values
- **Type Preservation**: Numbers, booleans preserved as strings

### Abbreviation System (Default: Enabled)

Brevit automatically creates abbreviations for frequently repeated key prefixes, placing definitions at the top of the output:

**Example:**
```
@u=user
@o=order
@u.name:John Doe
@u.email:john@example.com
@o.id:o-456
@o.status:SHIPPED
```

**Benefits:**
- **10-25% additional token savings** on nested data
- **Self-documenting**: Abbreviations are defined at the top
- **LLM-friendly**: Models easily understand the mapping
- **Configurable**: Can be disabled with `enable_abbreviations=False`

**When Abbreviations Help Most:**
- Deeply nested JSON structures
- Arrays of objects with repeated field names
- API responses with consistent schemas
- Data with many repeated prefixes (e.g., `user.profile.settings.theme`)

**Disable Abbreviations:**
```python
config = BrevitConfig(
    enable_abbreviations=False  # Disable abbreviation feature
)
```

## API

### BrevitClient

Main client class for optimization.

```python
class BrevitClient:
    def __init__(
        self,
        config: BrevitConfig,
        text_optimizer: Optional[ITextOptimizer] = None,
        image_optimizer: Optional[IImageOptimizer] = None,
    ):
    
    # Automatic optimization - analyzes data and selects best strategy
    async def brevity(self, raw_data: Any, intent: Optional[str] = None) -> str:
    
    # Explicit optimization with configured settings
    async def optimize(self, raw_data: Any, intent: Optional[str] = None) -> str:
    
    # Register custom optimization strategy
    def register_strategy(self, name: str, analyzer: Any, optimizer: Any) -> None:
```

**Example - Automatic Optimization:**
```python
# Automatically analyzes data structure and selects best strategy
optimized = await brevit.brevity(order)
# Automatically detects uniform arrays, long text, etc.
```

**Example - Explicit Optimization:**
```python
# Use explicit configuration
optimized = await brevit.optimize(order, "extract_total")
```

**Example - Custom Strategy:**
```python
# Register custom optimization strategy
brevit.register_strategy('custom', custom_analyzer, custom_optimizer)
```

### BrevitConfig

Configuration dataclass for BrevitClient.

```python
@dataclass
class BrevitConfig:
    json_mode: JsonOptimizationMode = JsonOptimizationMode.Flatten
    text_mode: TextOptimizationMode = TextOptimizationMode.Clean
    image_mode: ImageOptimizationMode = ImageOptimizationMode.Ocr
    json_paths_to_keep: List[str] = field(default_factory=list)
    long_text_threshold: int = 500
    enable_abbreviations: bool = True      # Default: True
    abbreviation_threshold: int = 2          # Default: 2
```

### Enums

#### JsonOptimizationMode
- `NONE` - No optimization
- `Flatten` - Flatten to key-value pairs (default)
- `ToYaml` - Convert to YAML
- `Filter` - Keep only specified paths

#### TextOptimizationMode
- `NONE` - No optimization
- `Clean` - Remove boilerplate
- `SummarizeFast` - Fast summarization
- `SummarizeHighQuality` - High-quality summarization

#### ImageOptimizationMode
- `NONE` - Skip processing
- `Ocr` - Extract text via OCR
- `Metadata` - Extract metadata only

## Using Brevit.py in LLM Prompts

### Best Practices

1. **Context First**: Provide context before optimized data
2. **Clear Instructions**: Tell the LLM what format to expect
3. **Examples**: Include format examples in prompts

### Example Prompt Template

```python
optimized = await brevit.optimize(order)

prompt = f"""You are analyzing order data. The data is in Brevit flattened format:

Context:
{optimized}

Task: Extract the order total and shipping address.

Format your response as JSON with keys: total, address"""
```

### Real-World Example

```python
async def analyze_order(order: dict):
    optimized = await brevit.optimize(order)
    
    prompt = f"""Analyze this order:

{optimized}

Questions:
1. What is the order total?
2. How many items?
3. Average item price?

Respond in JSON."""
    
    # Call OpenAI API
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

## Syntax Cheatsheet

### Python to Brevit Format

| Python Structure | Brevit Format | Example |
|------------------|---------------|---------|
| Dictionary key | `key: value` | `orderId: o-456` |
| Nested key | `parent.child: value` | `customer.name: John` |
| Primitive list | `list[count]: val1,val2,val3` | `friends[3]: ana,luis,sam` |
| Uniform object list | `list[count]{field1,field2}:`<br>`  val1,val2`<br>`  val3,val4` | `items[2]{sku,qty}:`<br>`  A-88,1`<br>`  T-22,2` |
| List element (fallback) | `list[index].key: value` | `items[0].sku: A-88` |
| Nested list | `parent[index].child[index]` | `orders[0].items[1].sku` |
| None value | `key: None` | `phone: None` |
| Boolean | `key: True` | `isActive: True` |
| Number | `key: 123` | `quantity: 5` |

### Special Cases

- **Empty Lists**: `items: []` → `items: []`
- **Empty Dicts**: `metadata: {}` → `metadata: {}`
- **None**: Explicit `None` values
- **Datetime**: Converted to ISO string
- **Tabular Arrays**: Automatically detected when all dicts have same keys
- **Primitive Arrays**: Automatically detected when all elements are primitives

## Other Implementations

Brevit is available in multiple languages:

| Language | Package | Status |
|----------|---------|--------|
| Python | `brevit` | ✅ Stable (This) |
| C# (.NET) | `Brevit` | ✅ Stable |
| JavaScript | `brevit` | ✅ Stable |

## Full Specification

### Format Specification

1. **Key-Value Pairs**: One pair per line
2. **Separator**: `: ` (colon + space)
3. **Key Format**: Dictionary keys with dot/bracket notation
4. **Value Format**: String representation of values
5. **Line Endings**: `\n` (newline)

### Grammar

```
brevit := line*
line := key ": " value "\n"
key := identifier ("." identifier | "[" number "]")*
value := string | number | boolean | None
identifier := [a-zA-Z_][a-zA-Z0-9_]*
```

### Examples

**Simple Dict:**
```
orderId: o-456
status: SHIPPED
```

**Nested Dict:**
```
customer.name: John Doe
customer.email: john@example.com
```

**List:**
```
items[0].sku: A-88
items[0].quantity: 1
items[1].sku: T-22
items[1].quantity: 2
```

**Complex Structure:**
```
orderId: o-456
customer.name: John Doe
items[0].sku: A-88
items[0].price: 29.99
items[1].sku: T-22
items[1].price: 39.99
shipping.address.street: 123 Main St
shipping.address.city: Toronto
```

## Performance Considerations

- **Flatten Mode**: Reduces token count by 40-60% compared to standard JSON
- **Async/Await**: All operations are asynchronous for better scalability
- **Memory Efficient**: Processes data in-place where possible
- **Type Hints**: Full type annotations for better performance with type checkers

## Best Practices

1. **Use Async/Await**: Always use `await` when calling `optimize()`
2. **Implement Custom Optimizers**: Replace default stubs with real LLM integrations
3. **Configure Thresholds**: Adjust `long_text_threshold` based on your use case
4. **Monitor Token Usage**: Track token counts before/after optimization
5. **Error Handling**: Wrap optimize calls in try-except blocks
6. **Use Type Hints**: Leverage type hints for better IDE support

## Troubleshooting

### Issue: "ToYaml mode requires 'pip install PyYAML'"

**Solution**: Install PyYAML: `pip install PyYAML` and update the code as shown in Advanced Usage.

### Issue: Text summarization returns stub

**Solution**: Implement a custom `ITextOptimizer` using LangChain, Semantic Kernel, or your LLM service (see Advanced Usage).

### Issue: Image OCR returns stub

**Solution**: Implement a custom `IImageOptimizer` using Azure AI Vision, Tesseract, or your OCR service (see Advanced Usage).

### Issue: "Filter mode is not implemented"

**Solution**: Install `jsonpath-ng` and implement JSON path filtering logic.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: [https://brevit.dev/docs](https://brevit.dev/docs)
- **Issues**: [https://github.com/JavianDev/Brevit.py/issues](https://github.com/JavianDev/Brevit.py/issues)
- **Email**: support@javianpicardo.com

## Version History

- **0.1.0** (Current): Initial release with core optimization features

