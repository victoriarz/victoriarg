# Victoria Ruiz Griffith - AI Engineer Portfolio

A portfolio website showcasing interactive AI-powered demos. Each project demonstrates practical applications of knowledge graphs, LLM integration, and data visualization.

**Live Site**: [victoriarg.com](https://victoriarg.com)

---

## Projects

### AI Chronicles
An auto-updating knowledge graph that visualizes the AI news landscape.

- **What it does**: Scrapes AI news from 8+ RSS feeds and Hacker News daily, extracts entities (companies, models, topics), and renders an interactive force-directed graph
- **Tech**: Python scraper, Gemini API for summarization, Canvas-based visualization, GitHub Actions for daily automation
- **Key files**: `aichronicle.html`, `aichronicle-graph.js`, `scraper.py`

### Saponify AI
An AI-powered soap making assistant with a built-in lye calculator.

- **What it does**: Chat naturally to get safe, accurate soap recipes with exact lye calculations based on verified SAP values
- **Tech**: Gemini 2.5 Flash, custom calculation engine with 19+ oils, RAG knowledge base for soap chemistry
- **Key files**: `saponifyai.html`, `soap-chat.js`, `soap-calculator.js`

### Pantry Atlas
An ingredient substitution assistant with an interactive knowledge graph.

- **What it does**: Find ingredient swaps, explore flavor pairings, and get cooking advice while respecting dietary restrictions
- **Tech**: Gemini 2.5 Flash, graph-based ontology with substitution relationships, dietary filtering
- **Key files**: `pantryatlas.html`, `pantry-atlas-chat.js`, `pantry-atlas-data.js`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Pages (Frontend)                      │
│                     victoriarg.com                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ AI          │  │ Saponify    │  │ Pantry      │              │
│  │ Chronicles  │  │ AI          │  │ Atlas       │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          │         ┌──────┴────────────────┴──────┐
          │         │  Render.com (API Proxy)      │
          │         │  Securely holds Gemini key   │
          │         └──────────────┬───────────────┘
          │                        │
          │                        ▼
          │              ┌─────────────────┐
          └─────────────►│  Gemini 2.5     │
           (scraper)     │  Flash API      │
                         └─────────────────┘
```

**Key design decisions**:
- LLMs handle reasoning; knowledge graphs provide structured context
- API keys stored server-side on Render.com (not in frontend)
- Zero-cost stack: GitHub Pages + Render free tier + Gemini free tier
- Vanilla JS with no frameworks for simplicity

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualization | Canvas API, D3.js force simulation concepts |
| AI | Gemini 2.5 Flash API |
| Backend Proxy | Node.js, Express (Render.com) |
| Automation | GitHub Actions (daily scraper) |
| Hosting | GitHub Pages |

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/yourusername/victoriarg.git
cd victoriarg

# Start local backend (for AI chat features)
cd server
npm install
npm start
# Backend runs at http://localhost:3000

# Open any HTML file in browser, or use a simple server:
python -m http.server 8000
# Visit http://localhost:8000
```

**Environment variables** (for server/.env):
```
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

---

## Project Structure

```
victoriarg/
├── index.html                 # Portfolio landing page
├── aichronicle.html           # AI Chronicles project
├── saponifyai.html            # Saponify AI project
├── pantryatlas.html           # Pantry Atlas project
├── style.css                  # Global styles (3800+ lines)
│
├── aichronicle-*.js           # AI Chronicles app, graph, data
├── soap-*.js                  # Saponify calculator, chat, knowledge
├── pantry-atlas-*.js          # Pantry Atlas engine, chat, data, viz
├── ai-config.js               # Gemini config (Saponify)
├── pantry-atlas-ai-config.js  # Gemini config (Pantry Atlas)
│
├── scraper.py                 # AI Chronicles news scraper
├── server/                    # Express proxy server
├── tests/                     # Test files
│
├── .github/workflows/         # GitHub Actions (daily scraper)
└── CLAUDE.md                  # AI assistant instructions
```

---

## Deployment

**Frontend**: Push to `main` branch → GitHub Pages auto-deploys

**Backend**: Render.com watches `server/` directory → auto-deploys on push

**Scraper**: GitHub Actions runs daily at 6 AM UTC → commits new data

See [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) for detailed setup instructions.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions for this codebase |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Backend proxy architecture details |
| [AI_CHRONICLES_README.md](AI_CHRONICLES_README.md) | AI Chronicles technical overview |
| [SAPONIFY-README.md](SAPONIFY-README.md) | Saponify AI technical overview |
| [PANTRYATLAS-README.md](PANTRYATLAS-README.md) | Pantry Atlas technical overview |

---

## License

This project is for portfolio demonstration purposes.

---

Built by Victoria Ruiz Griffith | AI Engineer
