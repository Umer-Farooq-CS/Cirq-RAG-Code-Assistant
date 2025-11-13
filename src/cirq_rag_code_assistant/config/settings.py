"""Application settings and configuration management."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Settings(PydanticBaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Cirq-RAG-Code-Assistant"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    testing: bool = Field(default=False, env="TESTING")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///data/cirq_rag.db", env="DATABASE_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="outputs/logs/cirq_rag.log", env="LOG_FILE")
    log_format: str = Field(default="text", env="LOG_FORMAT")
    enable_console_logging: bool = Field(default=True, env="ENABLE_CONSOLE_LOGGING")
    enable_file_logging: bool = Field(default=True, env="ENABLE_FILE_LOGGING")
    max_log_file_size: str = Field(default="10MB", env="MAX_LOG_FILE_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # PyTorch
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    torch_device: str = Field(default="auto", env="TORCH_DEVICE")  # auto, cpu, cuda, cuda:0
    torch_deterministic: bool = Field(default=False, env="TORCH_DETERMINISTIC")
    torch_benchmark: bool = Field(default=True, env="TORCH_BENCHMARK")
    torch_memory_fraction: float = Field(default=0.8, env="TORCH_MEMORY_FRACTION")
    
    # Models
    default_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="DEFAULT_MODEL_NAME"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    vector_db_type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    max_retrieval_results: int = Field(default=5, env="MAX_RETRIEVAL_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Agents
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=300, env="TIMEOUT_SECONDS")
    agent_parallel_execution: bool = Field(default=True, env="AGENT_PARALLEL_EXECUTION")
    agent_retry_attempts: int = Field(default=3, env="AGENT_RETRY_ATTEMPTS")
    
    # RAG
    knowledge_base_path: str = Field(default="data/knowledge_base", env="KNOWLEDGE_BASE_PATH")
    vector_index_path: str = Field(default="data/models/vector_index", env="VECTOR_INDEX_PATH")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    # Evaluation
    evaluation_dataset_path: str = Field(
        default="data/datasets/evaluation", 
        env="EVALUATION_DATASET_PATH"
    )
    benchmark_dataset_path: str = Field(
        default="data/datasets/benchmark", 
        env="BENCHMARK_DATASET_PATH"
    )
    metrics_output_path: str = Field(
        default="outputs/reports/metrics.json", 
        env="METRICS_OUTPUT_PATH"
    )
    evaluation_batch_size: int = Field(default=32, env="EVALUATION_BATCH_SIZE")
    
    # Performance
    batch_size: int = Field(default=16, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    # Security
    secret_key: str = Field(default="your_secret_key_here", env="SECRET_KEY")
    api_key_length: int = Field(default=32, env="API_KEY_LENGTH")
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")
    
    # External Services (Optional)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    elasticsearch_url: Optional[str] = Field(default=None, env="ELASTICSEARCH_URL")
    mongo_url: Optional[str] = Field(default=None, env="MONGO_URL")
    
    # Monitoring (Optional)
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=False, env="ENABLE_METRICS")
    metrics_interval: int = Field(default=60, env="METRICS_INTERVAL")
    
    # QCanvas Integration (Future Enhancement)
    qcanvas_api_url: Optional[str] = Field(default=None, env="QCANVAS_API_URL")
    qcanvas_api_key: Optional[str] = Field(default=None, env="QCANVAS_API_KEY")
    qcanvas_timeout: int = Field(default=30, env="QCANVAS_TIMEOUT")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()
    
    @validator("vector_db_type")
    def validate_vector_db_type(cls, v):
        """Validate vector database type."""
        valid_types = ["faiss", "chroma", "pinecone", "weaviate"]
        if v.lower() not in valid_types:
            raise ValueError(f"Vector DB type must be one of {valid_types}")
        return v.lower()
    
    @validator("log_format")
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ["text", "json"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of {valid_formats}")
        return v.lower()
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory."""
        return self.project_root / "outputs"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.outputs_dir / "logs"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.data_dir / "models"
    
    @property
    def knowledge_base_dir(self) -> Path:
        """Get knowledge base directory."""
        return self.data_dir / "knowledge_base"
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.outputs_dir,
            self.logs_dir,
            self.models_dir,
            self.knowledge_base_dir,
            self.outputs_dir / "reports",
            self.outputs_dir / "artifacts",
            self.data_dir / "datasets",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_pytorch(self) -> None:
        """Setup PyTorch configuration."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        
        try:
            import torch
            
            # Set deterministic behavior if requested
            if self.torch_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            else:
                torch.backends.cudnn.benchmark = self.torch_benchmark
            
            # Configure device
            if self.torch_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.torch_device
            
            # Configure GPU memory if using CUDA
            if device.startswith("cuda") and torch.cuda.is_available():
                # Set memory fraction
                if self.torch_memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.torch_memory_fraction
                    )
                
                # Print GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"PyTorch GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        except ImportError:
            print("PyTorch not installed. GPU acceleration will not be available.")
        except Exception as e:
            print(f"PyTorch GPU configuration error: {e}")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Initialize directories and PyTorch on import
settings.create_directories()
if not settings.testing:
    settings.setup_pytorch()
