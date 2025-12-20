# CLAUDE.md - Victoria Ruiz Griffith Portfolio

## Philosophy

Our applications are built with LLMs at the core for the reasoning. Knowledge graphs should be reviewed for correctness.

## Project Overview

Portfolio website with interactive AI-powered project demos. Hosted on GitHub Pages at victoriarg.com.

**Stack**: HTML5, CSS3, JavaScript, D3.js (force graphs), Gemini AI API, Node.js/Express (local dev server)

## Repository Structure

```
victoriarg/
├── index.html              # Main portfolio landing page
├── aichronicle.html        # AI Chronicles - interactive AI history knowledge graph
├── saponifyai.html         # Saponify AI - soap calculator with AI chat
├── pantryatlas.html        # Pantry Atlas - ingredient/cooking knowledge graph with AI chat
├── style.css               # Global styles (3800+ lines, organized by sections)
├── *-app.js, *-data.js     # App logic and data for each project
├── *-graph.js, *-viz.js    # D3.js visualization code
├── *-chat.js               # Chat functionality
├── ai-config.js            # Gemini AI configuration (Saponify)
├── pantry-atlas-ai-config.js  # Gemini AI configuration (Pantry Atlas)
├── server/                 # Local Express dev server
├── tests/                  # Test files
└── .claude/commands/       # Custom Claude Code skills
```

## Key Documentation (read these for deep context)

- [README.md](README.md) - Project overview, architecture diagram, and local dev setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - Saponify AI backend architecture (proxy server setup)
- [AI_CHRONICLES_README.md](AI_CHRONICLES_README.md) - AI Chronicles project details
- [SAPONIFY-README.md](SAPONIFY-README.md) - Saponify AI project details
- [PANTRYATLAS-README.md](PANTRYATLAS-README.md) - Pantry Atlas project details
- [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) - GitHub Pages deployment

## Code Conventions

- **CSS**: Organized with section headers (`/* --- 1. Section Name --- */`). See `style.css:1-50` for structure.
- **Colors**: Earthy tones - reference existing CSS rather than hardcoding. Primary browns/beiges throughout.
- **HTML**: Semantic HTML5, 4-space indentation
- **JS**: Each project has separate files for app logic, data, visualization, and chat

## Development Workflow

### Cache Busting (IMPORTANT)

All JS files use version query strings for cache busting. **Increment version when modifying any JS file:**

```html
<script src="soap-chat.js?v=1.0.5"></script>
```

This applies to all project JS files in saponifyai.html, aichronicle.html, and pantryatlas.html.

### Local Development

Run the Express server for local testing with AI features:

```bash
cd server && npm install && npm start
```

### Adding New Pages

1. Copy structure from an existing project page (e.g., aichronicle.html)
2. Add page-specific styles as a new section in style.css
3. Update navigation in index.html

## Deployment

GitHub Pages auto-deploys from main branch. See [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) for details.

## Guidelines

### Critical Rules

- IMPORTANT: Always read files before modifying them
- IMPORTANT: Increment JS version strings when changing any JavaScript file
- IMPORTANT: Maintain CSS section organization - don't break the commented structure
- Prefer simple solutions; avoid over-engineering for this portfolio site
- Ask before making significant architectural changes

### Avoid

- Adding frameworks/libraries unless explicitly requested
- Hardcoding colors - reference existing CSS patterns instead
- Creating new files when editing existing ones works

---

**Last Updated**: 2025-12-20
