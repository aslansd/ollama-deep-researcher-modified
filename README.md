# Modified Ollama Deep Researcher

Modified Ollama Deep Researcher is an extension of [Ollama Deep Researcher](https://github.com/langchain-ai/ollama-deep-researcher), a local web research assistant built upon the multi-agent framework of LangGraph that uses any LLM hosted by [Ollama](https://ollama.com/search). Give it a topic and it will generate a web search query, gather web search results (via seven different search APIs such as [DuckDuckGo](https://duckduckgo.com/) by default), summarize the results of web search, reflect on the summary to examine knowledge gaps, generate a new search query to address the gaps, search, and improve the summary for a user-defined number of cycles. It will provide the user a final markdown summary with all sources used. For more complete description of this app, please refer to [the original implementation](https://github.com/langchain-ai/ollama-deep-researcher)

### Mac

1. Download the Ollama app for Mac [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/llama3.2:latest):

```bash
ollama pull llama3.2:latest
```

3. Clone the repository:

```bash
git clone https://github.com/aslansd/ollama-deep-researcher-modified.git
cd ollama-deep-researcher-modified
```

4. Select a web search tool:

By default, it will use [DuckDuckGo](https://duckduckgo.com/) for web search, which does not require an API key. But you can also use the following serach API, by adding their API keys to the environment file and setting them as the default search API manually:

  * [Tavily](https://tavily.com/),
  * [Perplexity](https://www.perplexity.ai/),
  * [Linkup](https://www.linkup.so/),
  * [Exa](https://exa.ai/),
  * [ArXiv](https://arxiv.org/),
  * [PubMed](https://pubmed.ncbi.nlm.nih.gov/).

5. 

```bash
cp .env.example .env
```

6. The following environment variables are supported:

  * `OLLAMA_BASE_URL` - the endpoint of the Ollama service, defaults to `http://localhost:11434` if not set. 
  * `OLLAMA_MODEL` - the model to use, defaults to `llama3.2` if not set.
  * `SEARCH_API` - the search API to use, either `duckduckgo` (default), or `tavily` or `perplexity` or `linkup` or `exa`, or `arxiv`, or `pubmed`. You need to set the corresponding API key if it is required by the selected API. 
  * `TAVILY_API_KEY` - the tavily API key to use.
  * `PERPLEXITY_API_KEY` - the perplexity API key to use.
  * `LINKUP_API_KEY` - the linkup API key to use.
  * `EXA_API_KEY` - the exa API key to use.
  * `NCBI_API_KEY` - the pubmed API key to use.
  * `MAX_WEB_RESEARCH_LOOPS` - the maximum number of research loop steps, defaults to `3`.

5. (Recommended) Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

6. Launch the assistant with the LangGraph server:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

### Windows

1. Download the Ollama app for Windows [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/llama3.2:latest):

```powershell
ollama pull llama3.2:latest
```

3. Clone the repository:

```bash
git clone https://github.com/langchain-ai/ollama-deep-researcher-modified.git
cd ollama-deep-researcher-modified
```
 
4. Select a web search tool and set the required environment variables, as above.

5. (Recommended) Create a virtual environment: Install `Python 3.11` (and add to PATH during installation). Restart your terminal to ensure Python is available, then create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

6. Launch the assistant with the LangGraph server:

```powershell
# Install dependencies
pip install -e .
pip install langgraph-cli[inmem]

# Start the LangGraph server
langgraph dev
```

### Model Compatibility Note

When selecting a local LLM, note that this application relies on the model's ability to produce structured JSON output. Some models may have difficulty with this requirement:

- **Working well**: 
  - [Llama2 3.2](https://ollama.com/library/llama3.2)
  - [DeepSeek R1 (8B)](https://ollama.com/library/deepseek-r1:8b)
  
- **Known issues**:
  - [DeepSeek R1 (7B)](https://ollama.com/library/deepseek-llm:7b) - Currently has difficulty producing required JSON output
  
If you [encounter JSON-related errors](https://github.com/langchain-ai/ollama-deep-researcher/issues/18) (e.g., `KeyError: 'query'`), try switching to one of the confirmed working models.

### Browser Compatibility Note

When accessing the LangGraph Studio UI:
- Firefox is recommended for the best experience
- Safari users may encounter security warnings due to mixed content (HTTPS/HTTP)
- If you encounter issues, try:
  1. Using Firefox or another browser,
  2. Disabling ad-blocking extensions,
  3. Checking browser console for specific error messages,

## How it works

Modified Ollama Deep Researcher like its ancestor, Ollama Deep Researcher, is inspired by [IterDRAG](https://arxiv.org/html/2410.04343v1#:~:text=To%20tackle%20this%20issue%2C%20we,used%20to%20generate%20intermediate%20answers.). This approach will decompose a query into sub-queries, retrieve documents for each one, answer the sub-query, and then build on the answer by retrieving docs for the second sub-query. Here, I do similar thing:

- Given a user-provided topic, use a local LLM (via [Ollama](https://ollama.com/search)) to generate a web search query,
- Uses a search engine (such as [DuckDuckGo](https://duckduckgo.com/)) to find relevant sources,
- Uses LLM to summarize the findings from web search related to the user-provided research topic,
- Then, it uses the LLM to reflect on the summary, identifying knowledge gaps,
- It generates a new search query to address the knowledge gaps,
- The process repeats, with the summary being iteratively updated with new information from web search,
- It will repeat down the research rabbit hole,
- Runs for a configurable number of iterations (see `configuration` tab).