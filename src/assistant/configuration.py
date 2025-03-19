import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass


class SearchAPI(Enum):
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    LINKUP = "linkup"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    
    
@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))
    search_api_config: Optional[Dict[str, Any]] = None
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", 3))
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        
        return cls(**{k: v for k, v in values.items() if v})