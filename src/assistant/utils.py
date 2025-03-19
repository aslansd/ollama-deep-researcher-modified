import asyncio
import os
import requests
from typing import Any, Dict, List, Optional

from duckduckgo_search import DDGS
from tavily import TavilyClient
from exa_py import Exa
from linkup import LinkupClient

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper

from langsmith import traceable


def get_config_value(value):
    """Helper function to handle both string and enum cases of configuration values."""
    
    return value if isinstance(value, str) else value.value


def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "tavily", "exa").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "duckduckgo": [],  # Duckduckgo currently accepts no additional parameters    
        "tavily": [],  # Tavily currently accepts no additional parameters
        "perplexity": [],  # Perplexity accepts no additional parameters
        "linkup": ["depth"],
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],

    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source. Include_raw_content specifies whether 
    to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: 
        Either:
            - A dict with a 'results' key containing a list of search results.
            - A list of dicts, each containing search results.
            
    Returns:
        str: Formatted string with deduplicated sources.
    """
    
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()


@traceable
def duckduckgo_search(search_queries):
    """Perform searches using DuckDuckGo
    
    Args:
        search_queries (List[str]): List of search queries to process.
        
    Returns:
        List[dict]: List of search results.
    """

    results = []
    with DDGS() as ddgs:
        ddg_results = list(ddgs.text(search_queries, max_results=5))
                
        # Format results
        for i, result in enumerate(ddg_results):
            results.append({
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'content': result.get('body', ''),
                'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                'raw_content': result.get('body', '')
            })
            
    search_docs = {'query': search_queries,
                   'follow_up_questions': None,
                   'answer': None,
                   'images': [],
                   'results': results}
    
    return search_docs


@traceable
def tavily_search(search_queries):
    """
    Performs web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process.

    Returns:
        List[dict]: List of search responses from Tavily API, one per query.
    """
    
    tavily_client = TavilyClient(api_key = f"{os.getenv('TAVILY_API_KEY')}")
    search_docs = tavily_client.search(
                    search_queries,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                  )

    return search_docs


@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process.
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
                    {'query': str,   # The original search query
                     'follow_up_questions': None,      
                     'answer': None,
                     'images': list,
                     'results': [              # List of search results
                    {'title': str,             # Title of the search result
                     'url': str,               # URL of the result
                     'content': str,           # Summary/snippet of content
                     'score': float,           # Relevance score
                     'raw_content': str|None   # Full content or None for secondary citations
                    },
                    ...
                    ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
    }

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources.",
            },
            {
                "role": "user",
                "content": search_queries,
            }
        ]
        }
        
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()  # Raise exception for bad status codes
        
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations", ["https://perplexity.ai"])
        
    # Create results list for this query
    results = []
        
    # First citation gets the full content
    results.append({
        "title": f"Perplexity Search, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content,
        "score": 1.0,  # Adding score to match Perplexity format
    })
        
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search, Source {i}",
            "url": citation,
            "content": "See primary source for full content",
            "raw_content": None,
            "score": 0.5,  # Lower score for secondary sources
        })
        
    # Format response to match Perplexity structure
    search_docs = {
        "query": search_queries,
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": results,
    }
    
    return search_docs


@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """Performs web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process.
        depth (str, optional): "standard" (default) or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts.

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {'results': [# List of search results
                        {'title': str,      # Title of the search result
                         'url': str,        # URL of the result
                         'content': str,    # Summary/snippet of content
                        },
                        ...
                        ]
            }
    """

    client = LinkupClient(api_key = f"{os.getenv('LINKUP_API_KEY')}")
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {"results": [
             {"title": result.name, 
              "url": result.url, 
              "content": result.content}
            for result in response.results
            ],
            }
        )

    return search_results


@traceable
def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, include_domains: Optional[List[str]] = None, exclude_domains: Optional[List[str]] = None, subpages: Optional[int] = None):
    """Search the web using the Exa API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process.
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                        If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results. 
                                               When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
                                               Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.
        
    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {'query': str, # The original search query
             'follow_up_questions': None,      
             'answer': None,
             'images': list,
             'results': [ # List of search results
                        {'title': str,            # Title of the search result
                         'url': str,              # URL of the result
                         'content': str,          # Summary/snippet of content
                         'score': float,          # Relevance score
                         'raw_content': str|None  # Full content or None for secondary citations
                        },
                        ...
                        ]
                        }
    """
    
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")
    
    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
        
    # Define the function for the executor with all parameters
    def exa_search_fn(search_queries):
        # Build parameters dictionary
        kwargs = {
            # Set text to True if max_characters is None, otherwise use an object with max_characters
            "text": True if max_characters is None else {"max_characters": max_characters},
            "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
            "num_results": num_results,
        }
            
        # Add optional parameters only if they are provided
        if subpages is not None:
            kwargs["subpages"] = subpages
                
        if include_domains:
            kwargs["include_domains"] = include_domains
        elif exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
                
        return exa.search_and_contents(search_queries, **kwargs)
        
    search_docs = []
    
    try:
        response = exa_search_fn(search_queries)
        
        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        
        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # Access the results from the SearchResponse object
        results_list = get_value(response, 'results', [])
        
        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, 'score', 0.0)
            
            # Combine summary and text for content if both are available
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content,
            }
            
            # Add the main result to the formatted results
            formatted_results.append(result_entry)
        
        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text,
                    })
        
        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # Avoid duplicate images
                images.append(image)
                
        search_docs.append({
            "query": search_queries,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results,
        })
    
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error processing query '{search_queries}': {str(e)}")
        # Add a placeholder result for failed queries to maintain index alignment
        search_docs.append({
            "query": search_queries,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": [],
            "error": str(e),
        })
            
        # If we hit a rate limit error
        if "429" in str(e):
            print("Rate limit exceeded...")
    
    return search_docs


@traceable
def arxiv_search(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """Performs searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs.
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
                   {'query': str,   # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': [],
                    'results': [# List of search results
                                {'title': str,            # Title of the paper
                                 'url': str,              # URL (Entry ID) of the paper
                                 'content': str,          # Formatted summary with metadata
                                 'score': float,          # Relevance score (approximated)
                                 'raw_content': str|None  # Full paper content if available
                                },
                                ...
                                ]
                                }
    """
    
    def process_single_query(search_queries):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta,
            )
            
            # Run the retriever
            docs = retriever.invoke(search_queries)
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata
                
                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')
                
                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None,
                }
                results.append(result)
                
            return {
                'query': search_queries,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results,
            }
            
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{search_queries}': {str(e)}")
            return {
                'query': search_queries,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e),
            }
    
    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    
    try:
        result = process_single_query(search_queries)
        search_docs.append(result)
        
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error processing arXiv query '{search_queries}': {str(e)}")
        search_docs.append({
            'query': search_queries,
            'follow_up_questions': None,
            'answer': None,
            'images': [],
            'results': [],
            'error': str(e),
        })
            
        # If we hit a rate limit error
        if "429" in str(e) or "Too Many Requests" in str(e):
            print("ArXiv rate limit exceeded...")
    
    return search_docs


@traceable
def pubmed_search(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """Performs searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries.
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
                    {'query': str,   # The original search query
                     'follow_up_questions': None,      
                     'answer': None,
                     'images': [],
                     'results': [# List of search results
                                 {'title': str,            # Title of the paper
                                  'url': str,              # URL to the paper on PubMed
                                  'content': str,          # Formatted summary with metadata
                                  'score': float,          # Relevance score (approximated)
                                  'raw_content': str       # Full abstract content
                                 },
                                 ...
                                ]
                    }
    """
    
    def process_single_query(seach_queries):
        try:
            # print(f"Processing PubMed query: '{query}'")
            
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "aslansatary@gmail.com",
                api_key=api_key if api_key else f"{os.getenv('NCBI_API_KEY_API_KEY')}",
            )
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = list(wrapper.lazy_load(search_queries))
            
            print(f"Query '{search_queries}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', ''),
                }
                results.append(result)
            
            return {
                'query': search_queries,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results,
            }
            
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{search_queries}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            
            return {
                'query': search_queries,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e),
            }
    
    # Process all queries with a reasonable delay between them
    search_docs = []
    
    try:
        result = process_single_query(search_queries)
        search_docs.append(result)
    
    except Exception as e:
        # Handle exceptions gracefully
        error_msg = f"Error in main loop processing PubMed query '{search_queries}': {str(e)}"
        print(error_msg)
            
        search_docs.append({
            'query': search_queries,
            'follow_up_questions': None,
            'answer': None,
            'images': [],
            'results': [],
            'error': str(e),
        })
    
    return search_docs