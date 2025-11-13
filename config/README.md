# Configuration Directory

This directory contains all configuration files for the Cirq-RAG-Code-Assistant project.

## Files

- **`config.json`** - Main configuration file with all settings
- **`config.dev.json`** - Development-specific configuration
- **`config.template.json`** - Template file for creating your own config
- **`config_loader.py`** - Python module that loads and manages configuration
- **`__init__.py`** - Package initialization file

## Usage

### In Python Files

```python
from config import get_config, get_config_loader

# Get config dictionary
config = get_config()
model_name = config.get("models.embedding.model_name")

# Get config loader for advanced operations
config_loader = get_config_loader()
rag_section = config_loader.get_section("rag")
```

### In Notebooks

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

# Import config
from config import get_config, get_config_loader

config = get_config()
```

## Configuration Structure

See `config.template.json` for the complete configuration structure with all available options.

## Environment Variables

The following environment variables can override config values:

- `OPENAI_API_KEY` - OpenAI API key
- `HUGGINGFACE_API_KEY` - Hugging Face API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `ENVIRONMENT` - Environment name (development, production)
- `DEBUG` - Debug mode (true/false)
- `LOG_LEVEL` - Logging level

## Setup

1. Copy the template:
   ```bash
   cp config/config.template.json config/config.json
   ```

2. Edit `config/config.json` and add your API keys

3. The config loader will automatically:
   - Load the appropriate config file based on environment
   - Apply environment variable overrides
   - Create all necessary directories
   - Setup PyTorch/CUDA configuration

