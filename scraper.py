#!/usr/bin/env python3
"""
THE AI CHRONICLE - News Scraper and Knowledge Graph Builder
Aggregates AI news from multiple sources and builds a knowledge graph.

Usage:
    python scraper.py                    # Full scrape with AI summarization (requires GEMINI_API_KEY)
    python scraper.py --no-ai            # Scrape without AI (for testing)
    python scraper.py --days 7           # Scrape last 7 days

Environment Variables:
    GEMINI_API_KEY                       # Optional: For AI-generated article summaries
"""

import os
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse
import feedparser
import requests
from dataclasses import dataclass, asdict
from collections import defaultdict

# ============================================
# Configuration
# ============================================

RSS_FEEDS = {
    "arxiv_cs_ai": "https://arxiv.org/rss/cs.AI",
    "arxiv_cs_lg": "https://arxiv.org/rss/cs.LG",
    "arxiv_cs_cl": "https://arxiv.org/rss/cs.CL",
    "openai_blog": "https://openai.com/blog/rss/",
    "google_ai": "https://blog.google/technology/ai/rss/",
    "huggingface": "https://huggingface.co/blog/feed.xml",
    "anthropic": "https://www.anthropic.com/feed.xml",
    "mit_tech_review_ai": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
}

HACKER_NEWS_API = "https://hn.algolia.com/api/v1/search_by_date"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Known entities for extraction
KNOWN_ORGS = [
    "OpenAI", "Anthropic", "Google", "Google DeepMind", "DeepMind", "Meta", 
    "Meta AI", "Microsoft", "NVIDIA", "Hugging Face", "Stability AI",
    "Mistral", "Cohere", "AI21", "xAI", "Amazon", "AWS", "Apple", "IBM",
    "Baidu", "Alibaba", "Tencent", "ByteDance"
]

KNOWN_MODELS = [
    "GPT-4", "GPT-4o", "GPT-5", "ChatGPT", "o1", "o3",
    "Claude", "Claude 3", "Claude 3.5", "Opus", "Sonnet", "Haiku",
    "Gemini", "Gemini 2", "Gemini 3", "Gemini Pro", "Gemini Ultra", "Gemini Flash",
    "Llama", "Llama 2", "Llama 3", "Llama 3.3", "Llama 4",
    "Mistral", "Mixtral", "DALL-E", "Midjourney",
    "Stable Diffusion", "Sora", "Copilot", "Grok", "Grok 3", "PaLM", "Bard",
    "Nemotron", "Phi", "Phi-4", "Qwen", "DeepSeek", "Command R"
]

# Model-to-Organization mappings (which company made which model)
MODEL_TO_ORG = {
    "GPT-4": "OpenAI", "GPT-4o": "OpenAI", "GPT-5": "OpenAI", "ChatGPT": "OpenAI",
    "o1": "OpenAI", "o3": "OpenAI", "DALL-E": "OpenAI", "Sora": "OpenAI",
    "Claude": "Anthropic", "Claude 3": "Anthropic", "Claude 3.5": "Anthropic",
    "Opus": "Anthropic", "Sonnet": "Anthropic", "Haiku": "Anthropic",
    "Gemini": "Google", "Gemini 2": "Google", "Gemini 3": "Google",
    "Gemini Pro": "Google", "Gemini Ultra": "Google", "Gemini Flash": "Google",
    "PaLM": "Google", "Bard": "Google",
    "Llama": "Meta", "Llama 2": "Meta", "Llama 3": "Meta", "Llama 3.3": "Meta", "Llama 4": "Meta",
    "Mistral": "Mistral", "Mixtral": "Mistral",
    "Grok": "xAI", "Grok 3": "xAI",
    "Copilot": "Microsoft", "Phi": "Microsoft", "Phi-4": "Microsoft",
    "Nemotron": "NVIDIA",
    "Stable Diffusion": "Stability AI",
    "Qwen": "Alibaba",
    "Command R": "Cohere",
    "DeepSeek": "DeepSeek",
}

# Competing models (for COMPETES_WITH edges)
MODEL_COMPETITORS = [
    ("Claude", "GPT-4"), ("Claude", "Gemini"), ("GPT-4", "Gemini"),
    ("Claude 3.5", "GPT-4o"), ("Claude 3.5", "Gemini Flash"),
    ("Llama 3", "Mistral"), ("Llama 3", "Claude"),
    ("Grok", "Claude"), ("Grok", "GPT-4"),
    ("o1", "Claude 3.5"), ("o3", "Opus"),
]

# Competing organizations (for COMPETES_WITH edges)
ORG_COMPETITORS = [
    ("OpenAI", "Anthropic"), ("OpenAI", "Google"), ("Anthropic", "Google"),
    ("OpenAI", "Meta"), ("Anthropic", "Meta"), ("Google", "Meta"),
    ("OpenAI", "xAI"), ("Microsoft", "Google"),
    ("NVIDIA", "Google"), ("NVIDIA", "AMD"),
]

KNOWN_TOPICS = [
    ("Large Language Models", ["LLM", "language model", "transformer"]),
    ("AI Reasoning", ["reasoning", "chain-of-thought", "CoT", "logical"]),
    ("Multimodal AI", ["multimodal", "vision-language", "VLM", "audio-visual"]),
    ("AI Agents", ["agent", "agentic", "tool use", "autonomous"]),
    ("AI Safety", ["safety", "alignment", "RLHF", "constitutional AI", "jailbreak"]),
    ("RAG", ["retrieval", "RAG", "retrieval-augmented"]),
    ("Diffusion Models", ["diffusion", "DDPM", "stable diffusion", "image generation"]),
    ("Model Efficiency", ["efficiency", "quantization", "pruning", "distillation", "MoE"]),
    ("Prompt Engineering", ["prompting", "prompt", "few-shot", "zero-shot"]),
    ("Computer Vision", ["vision", "image", "video", "object detection", "segmentation"]),
    ("NLP", ["NLP", "natural language", "text", "sentiment", "NER"]),
    ("Reinforcement Learning", ["RL", "reinforcement", "reward", "policy"]),
    ("Fine-tuning", ["fine-tuning", "fine-tune", "PEFT", "LoRA", "adapter"]),
]

# ============================================
# Data Classes
# ============================================

@dataclass
class Article:
    id: str
    title: str
    summary: str
    url: str
    source: str
    date: str
    trending_score: int = 50

@dataclass
class Node:
    id: str
    type: str  # article, topic, organization, model
    title: str
    summary: str
    url: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    trending_score: Optional[int] = None
    connection_count: Optional[int] = None

@dataclass
class Edge:
    source: str
    target: str
    relationship: str  # COVERS, MENTIONS, CREATED_BY, RELATED_TO

# ============================================
# Scraping Functions
# ============================================

def fetch_rss_feeds(days: int = 7) -> List[Dict]:
    """Fetch articles from RSS feeds."""
    articles = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            print(f"Fetching {feed_name}...")
            feed = feedparser.parse(feed_url)
            
            source_type = "arxiv" if "arxiv" in feed_name else "blogs"
            
            for entry in feed.entries[:20]:  # Limit per feed
                # Parse date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                else:
                    pub_date = datetime.now()
                
                # Skip if too old
                if pub_date < cutoff_date:
                    continue
                
                # Extract summary
                summary = ""
                if hasattr(entry, 'summary'):
                    summary = clean_html(entry.summary)[:500]
                elif hasattr(entry, 'description'):
                    summary = clean_html(entry.description)[:500]
                
                articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "summary": summary,
                    "source": source_type,
                    "date": pub_date.strftime("%Y-%m-%d"),
                    "feed": feed_name
                })
                
        except Exception as e:
            print(f"Error fetching {feed_name}: {e}")
    
    return articles


def fetch_hacker_news(days: int = 7) -> List[Dict]:
    """Fetch AI-related posts from Hacker News."""
    articles = []
    
    queries = ["AI", "LLM", "GPT", "machine learning", "neural network", "Claude", "Gemini"]
    
    for query in queries:
        try:
            print(f"Fetching HN: {query}...")
            params = {
                "query": query,
                "tags": "story",
                "numericFilters": f"created_at_i>{int((datetime.now() - timedelta(days=days)).timestamp())}",
                "hitsPerPage": 20
            }
            
            response = requests.get(HACKER_NEWS_API, params=params, timeout=10)
            data = response.json()
            
            for hit in data.get("hits", []):
                # Filter for relevance
                title_lower = hit.get("title", "").lower()
                if not any(kw in title_lower for kw in ["ai", "llm", "gpt", "ml", "neural", "model", "claude", "gemini", "anthropic", "openai"]):
                    continue
                
                pub_date = datetime.fromtimestamp(hit.get("created_at_i", 0))
                
                articles.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                    "summary": "",
                    "source": "hackernews",
                    "date": pub_date.strftime("%Y-%m-%d"),
                    "points": hit.get("points", 0),
                    "comments": hit.get("num_comments", 0)
                })
                
        except Exception as e:
            print(f"Error fetching HN for '{query}': {e}")
    
    # Deduplicate
    seen = set()
    unique = []
    for article in articles:
        if article["url"] not in seen:
            seen.add(article["url"])
            unique.append(article)
    
    return unique


def clean_html(text: str) -> str:
    """Remove HTML tags and clean text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================
# AI Summarization (Gemini API)
# ============================================

def summarize_with_gemini(articles: List[Dict], api_key: str) -> List[Dict]:
    """Use Gemini API to generate summaries for articles."""
    if not api_key:
        print("No API key provided, skipping AI summarization")
        return articles

    url = f"{GEMINI_API_URL}?key={api_key}"

    for article in articles:
        if article.get("summary") and len(article["summary"]) > 100:
            continue  # Already has a good summary

        try:
            prompt = f"""Summarize this AI news article in 1-2 sentences (max 200 characters). Focus on the key technical contribution or announcement.

Title: {article['title']}
Content: {article.get('summary', 'No content available')}

Summary:"""

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 150
                }
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                summary = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if summary:
                    article["summary"] = summary
                    print(f"  Summarized: {article['title'][:50]}...")
            else:
                print(f"  API error for {article['title'][:30]}: {response.status_code}")

        except Exception as e:
            print(f"  Error summarizing: {e}")

    return articles

# ============================================
# Entity Extraction
# ============================================

def extract_entities(article: Dict) -> Dict:
    """Extract organizations, models, and topics from article."""
    text = f"{article['title']} {article.get('summary', '')}".lower()

    entities = {
        "organizations": [],
        "models": [],
        "topics": []
    }

    # Extract organizations
    for org in KNOWN_ORGS:
        if org.lower() in text:
            entities["organizations"].append(org)

    # Extract models
    for model in KNOWN_MODELS:
        if model.lower() in text:
            entities["models"].append(model)
            # Also add the organization that created this model
            if model in MODEL_TO_ORG:
                org = MODEL_TO_ORG[model]
                if org not in entities["organizations"]:
                    entities["organizations"].append(org)

    # Extract topics
    for topic_name, keywords in KNOWN_TOPICS:
        if any(kw.lower() in text for kw in keywords):
            entities["topics"].append(topic_name)

    # If no topics found but article is AI-related, assign default topic
    if not entities["topics"]:
        title_lower = article['title'].lower()
        # Fallback topic detection for HN articles with sparse content
        if any(kw in title_lower for kw in ["llm", "gpt", "claude", "gemini", "chatgpt"]):
            entities["topics"].append("Large Language Models")
        elif any(kw in title_lower for kw in ["agent", "agentic", "autonomous"]):
            entities["topics"].append("AI Agents")
        elif any(kw in title_lower for kw in ["vision", "image", "video", "diffusion"]):
            entities["topics"].append("Computer Vision")
        elif "ai" in title_lower or "ml" in title_lower:
            entities["topics"].append("NLP")  # Default fallback

    return entities

# ============================================
# Knowledge Graph Builder
# ============================================

def build_knowledge_graph(articles: List[Dict]) -> Dict:
    """Build knowledge graph from articles."""
    nodes = []
    edges = []
    
    # Track unique entities
    topics_seen = {}
    orgs_seen = {}
    models_seen = {}
    
    # Topic descriptions
    topic_descriptions = {
        "Large Language Models": "Foundation models trained on massive text corpora that can generate and understand natural language.",
        "AI Reasoning": "Methods to improve logical reasoning, mathematical problem-solving, and multi-step thinking in AI systems.",
        "Multimodal AI": "Systems that process and understand multiple types of input including text, images, audio, and video.",
        "AI Agents": "Autonomous AI systems that can plan, use tools, and take actions to accomplish goals.",
        "AI Safety": "Research focused on making AI systems safe, aligned with human values, and beneficial.",
        "RAG": "Retrieval-Augmented Generation: combining LLMs with external knowledge retrieval for more accurate responses.",
        "Diffusion Models": "Generative models that create content by iteratively denoising random noise into structured outputs.",
        "Model Efficiency": "Techniques to reduce computational costs and improve inference speed of AI models.",
        "Prompt Engineering": "Methods for crafting effective prompts to guide AI model behavior and outputs.",
        "Computer Vision": "AI systems for understanding and processing visual information from images and video.",
        "NLP": "Natural Language Processing: AI techniques for understanding and generating human language.",
        "Reinforcement Learning": "Training AI through rewards and penalties to learn optimal behaviors.",
        "Fine-tuning": "Adapting pre-trained models to specific tasks or domains.",
    }
    
    # Process articles
    for i, article in enumerate(articles):
        article_id = f"article-{hashlib.md5(article['url'].encode()).hexdigest()[:8]}"
        
        # Calculate trending score
        trending_score = 50
        if article.get("points"):
            trending_score = min(100, 50 + article["points"] // 10)
        
        # Create article node
        nodes.append({
            "id": article_id,
            "type": "article",
            "title": article["title"],
            "summary": article.get("summary", ""),
            "url": article["url"],
            "source": article["source"],
            "date": article["date"],
            "trendingScore": trending_score
        })
        
        # Extract and link entities
        entities = extract_entities(article)
        
        # Topics
        for topic in entities["topics"]:
            topic_id = f"topic-{topic.lower().replace(' ', '-')}"
            if topic_id not in topics_seen:
                topics_seen[topic_id] = {
                    "id": topic_id,
                    "type": "topic",
                    "title": topic,
                    "summary": topic_descriptions.get(topic, f"Articles related to {topic}."),
                    "connectionCount": 0
                }
            topics_seen[topic_id]["connectionCount"] += 1
            edges.append({"source": article_id, "target": topic_id, "relationship": "COVERS"})
        
        # Organizations
        for org in entities["organizations"]:
            org_id = f"org-{org.lower().replace(' ', '-')}"
            if org_id not in orgs_seen:
                orgs_seen[org_id] = {
                    "id": org_id,
                    "type": "organization",
                    "title": org,
                    "summary": f"{org} - AI research and development.",
                    "connectionCount": 0
                }
            orgs_seen[org_id]["connectionCount"] += 1
            edges.append({"source": article_id, "target": org_id, "relationship": "MENTIONS"})
        
        # Models
        for model in entities["models"]:
            model_id = f"model-{model.lower().replace(' ', '-')}"
            if model_id not in models_seen:
                models_seen[model_id] = {
                    "id": model_id,
                    "type": "model",
                    "title": model,
                    "summary": f"{model} AI model.",
                    "connectionCount": 0
                }
            models_seen[model_id]["connectionCount"] += 1
            edges.append({"source": article_id, "target": model_id, "relationship": "MENTIONS"})
    
    # Add entity nodes
    nodes.extend(topics_seen.values())
    nodes.extend(orgs_seen.values())
    nodes.extend(models_seen.values())
    
    # Add topic relationships (expanded for better connectivity)
    topic_relations = [
        # LLM connections
        ("topic-large-language-models", "topic-ai-reasoning"),
        ("topic-large-language-models", "topic-ai-agents"),
        ("topic-large-language-models", "topic-rag"),
        ("topic-large-language-models", "topic-nlp"),
        ("topic-large-language-models", "topic-fine-tuning"),
        ("topic-large-language-models", "topic-prompt-engineering"),
        ("topic-large-language-models", "topic-model-efficiency"),
        # Safety connections
        ("topic-ai-safety", "topic-large-language-models"),
        ("topic-ai-safety", "topic-ai-agents"),
        ("topic-ai-safety", "topic-ai-reasoning"),
        # Vision connections
        ("topic-multimodal-ai", "topic-computer-vision"),
        ("topic-multimodal-ai", "topic-large-language-models"),
        ("topic-diffusion-models", "topic-computer-vision"),
        ("topic-diffusion-models", "topic-multimodal-ai"),
        # Agent connections
        ("topic-ai-agents", "topic-prompt-engineering"),
        ("topic-ai-agents", "topic-rag"),
        ("topic-ai-agents", "topic-ai-reasoning"),
        # Efficiency connections
        ("topic-model-efficiency", "topic-fine-tuning"),
        # RL connections
        ("topic-reinforcement-learning", "topic-ai-agents"),
        ("topic-reinforcement-learning", "topic-ai-reasoning"),
    ]

    for source, target in topic_relations:
        if source in topics_seen and target in topics_seen:
            edges.append({"source": source, "target": target, "relationship": "RELATED_TO"})

    # Add Model-to-Organization edges (CREATED_BY)
    for model_id, model_data in models_seen.items():
        model_name = model_data["title"]
        if model_name in MODEL_TO_ORG:
            org_name = MODEL_TO_ORG[model_name]
            org_id = f"org-{org_name.lower().replace(' ', '-')}"
            if org_id in orgs_seen:
                edges.append({"source": model_id, "target": org_id, "relationship": "CREATED_BY"})

    # Add Model competitor edges (COMPETES_WITH)
    for model1, model2 in MODEL_COMPETITORS:
        id1 = f"model-{model1.lower().replace(' ', '-')}"
        id2 = f"model-{model2.lower().replace(' ', '-')}"
        if id1 in models_seen and id2 in models_seen:
            edges.append({"source": id1, "target": id2, "relationship": "COMPETES_WITH"})

    # Add Org competitor edges (COMPETES_WITH)
    for org1, org2 in ORG_COMPETITORS:
        id1 = f"org-{org1.lower().replace(' ', '-')}"
        id2 = f"org-{org2.lower().replace(' ', '-')}"
        if id1 in orgs_seen and id2 in orgs_seen:
            edges.append({"source": id1, "target": id2, "relationship": "COMPETES_WITH"})
    
    return {
        "metadata": {
            "lastUpdated": datetime.now().isoformat() + "Z",
            "totalArticles": len([n for n in nodes if n["type"] == "article"]),
            "totalNodes": len(nodes),
            "totalEdges": len(edges),
            "dateRange": {
                "start": min(a["date"] for a in articles) if articles else "",
                "end": max(a["date"] for a in articles) if articles else ""
            }
        },
        "nodes": nodes,
        "edges": edges
    }

# ============================================
# Output Generation
# ============================================

def generate_js_file(graph_data: Dict, output_path: str):
    """Generate JavaScript data file."""
    js_content = f"""// ================================================
// THE AI CHRONICLE - Knowledge Graph Data
// Auto-generated and updated daily via GitHub Actions
// Last updated: {datetime.now().strftime('%Y-%m-%d')}
// ================================================

const AIChronicleData = {json.dumps(graph_data, indent=4)};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = AIChronicleData;
}}
"""
    
    with open(output_path, 'w') as f:
        f.write(js_content)
    
    print(f"Generated: {output_path}")


def generate_json_file(graph_data: Dict, output_path: str):
    """Generate JSON data file for archiving."""
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Generated: {output_path}")

# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="AI Chronicles News Scraper")
    parser.add_argument("--days", type=int, default=7, help="Number of days to scrape")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI summarization")
    parser.add_argument("--output", default="aichronicle-data.js", help="Output file path")
    args = parser.parse_args()
    
    print("=" * 50)
    print("THE AI CHRONICLE - News Scraper")
    print("=" * 50)
    
    # Fetch articles
    print("\n[1/4] Fetching RSS feeds...")
    rss_articles = fetch_rss_feeds(days=args.days)
    print(f"  Found {len(rss_articles)} articles from RSS")
    
    print("\n[2/4] Fetching Hacker News...")
    hn_articles = fetch_hacker_news(days=args.days)
    print(f"  Found {len(hn_articles)} articles from HN")
    
    # Combine and deduplicate
    all_articles = rss_articles + hn_articles
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article["url"] not in seen_urls:
            seen_urls.add(article["url"])
            unique_articles.append(article)
    
    print(f"\n  Total unique articles: {len(unique_articles)}")
    
    # AI Summarization
    print("\n[3/4] AI Summarization...")
    if not args.no_ai:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            unique_articles = summarize_with_gemini(unique_articles, api_key)
        else:
            print("  GEMINI_API_KEY not set, skipping summarization")
    else:
        print("  Skipped (--no-ai flag)")
    
    # Build knowledge graph
    print("\n[4/4] Building knowledge graph...")
    graph_data = build_knowledge_graph(unique_articles)
    print(f"  Nodes: {graph_data['metadata']['totalNodes']}")
    print(f"  Edges: {graph_data['metadata']['totalEdges']}")
    
    # Generate output files
    print("\nGenerating output files...")
    generate_js_file(graph_data, args.output)
    generate_json_file(graph_data, args.output.replace('.js', '.json'))
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
