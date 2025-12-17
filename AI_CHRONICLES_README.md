# AI Chronicles - Technical Overview

## What It Is
AI Chronicles is a self-updating knowledge graph that visualizes the AI news landscape. It scrapes articles daily, extracts entities (organizations, models, topics), and renders an interactive force-directed graph.

## Architecture

```
RSS Feeds + HN API → Python Scraper → Gemini Summarization → Knowledge Graph → Canvas Visualization
                           ↓
                    GitHub Actions (daily 6 AM UTC)
                           ↓
                    Auto-commit to repo → GitHub Pages
```

## Key Files

| File | Purpose |
|------|---------|
| `aichronicle.html` | Main page - UI, styling, layout |
| `aichronicle-app.js` | App controller - populates UI, handles events |
| `aichronicle-graph.js` | Canvas visualization - physics simulation, rendering |
| `aichronicle-data.js` | Generated data file - nodes and edges (auto-updated) |
| `scraper.py` | Python scraper - fetches articles, builds graph |
| `.github/workflows/update-chronicle.yml` | GitHub Actions - daily automation |

## Data Pipeline (scraper.py)

### Sources
- **RSS Feeds (8)**: arXiv (cs.AI, cs.LG, cs.CL), OpenAI, Google AI, Hugging Face, Anthropic, MIT Tech Review
- **Hacker News API**: Searches for AI/LLM/GPT topics

### Entity Extraction
- **Organizations (22)**: OpenAI, Anthropic, Google, Meta, Microsoft, NVIDIA, etc.
- **Models (20+)**: GPT-4, Claude, Gemini, Llama, Mistral, etc.
- **Topics (13)**: LLMs, AI Reasoning, Multimodal AI, AI Agents, AI Safety, RAG, etc.

### Graph Structure
- **Node types**: article, topic, organization, model
- **Edge types**: COVERS (article→topic), MENTIONS (article→org/model), RELATED_TO (topic→topic)

### Commands
```bash
python scraper.py --days 7 --no-ai    # Scrape without AI summaries
python scraper.py --days 7            # Scrape with Gemini summaries (needs GEMINI_API_KEY)
```

## Visualization (aichronicle-graph.js)

### Physics
- Force-directed layout with repulsion (5000), attraction (0.01), damping (0.85), center gravity (0.02)
- Max 50 nodes displayed (scored by recency + trending + connections)

### Interactions
- **Click node**: Select, show details in sidebar
- **Drag node**: Pin position
- **Pan**: Click + drag empty space
- **Zoom**: Mousewheel (0.3x to 3x)

### Node Sizing
- Articles: `12 + (trendingScore / 20)`
- Topics: `18 + (connectionCount * 1.5)`
- Orgs/Models: `14-16 + connectionCount`

### Colors
- Articles: `#e07b53` (coral)
- Topics: `#5b8a72` (green)
- Organizations: `#6b8cae` (blue)
- Models: `#9b7bb8` (purple)

## App Controller (aichronicle-app.js)

- Populates stats bar (nodes, edges, articles, last updated)
- Populates trending topics (top 5 by connection count)
- Populates recent articles (top 5 by date)
- Handles node selection → updates details card
- Controls: reset view, toggle labels, filter trending

## GitHub Actions

**Schedule**: Daily at 6 AM UTC
**Secrets needed**: `GEMINI_API_KEY` (optional, for AI summaries)

Workflow:
1. Checkout repo
2. Install Python 3.11 + dependencies (feedparser, requests)
3. Run scraper
4. Commit + push if data changed

## Quick Fixes

### Scraper not working
- Check RSS feed URLs are still valid
- Check Hacker News API endpoint
- Run with `--no-ai` to bypass Gemini

### Graph not rendering
- Check browser console for JS errors
- Verify `aichronicle-data.js` has valid data
- Check canvas sizing (should be `position: absolute` with `top/left: 0`)

### Summaries not generating
- Set `GEMINI_API_KEY` environment variable
- Model: `gemini-2.5-flash`
- API URL: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`

## Dependencies

**Python** (scraper):
- feedparser >= 6.0.0
- requests >= 2.28.0

**Frontend**: Vanilla JS, no frameworks

## Recent Changes (Dec 2025)
- Switched from Claude API to Gemini 2.5 Flash API
- Added time to "Last Updated" display
- Fixed canvas whitespace issue with absolute positioning
- Updated page content with accurate tech details
