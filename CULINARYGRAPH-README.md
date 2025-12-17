# Culinary Graph - Architecture Overview

## What It Is
An AI-powered ingredient substitution assistant with an interactive knowledge graph. Users can find ingredient swaps, explore flavor pairings, and get cooking advice while respecting dietary restrictions.

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   culinarygraph.html                     │
│              (Main UI + Pixel-Art Styles)                │
└─────────────────────────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐
│culinary-     │  │culinary-graph-  │  │culinary-graph-  │
│  chat.js     │  │  data.js        │  │  engine.js      │
│(Chat + RAG)  │  │(Ontology + Data)│  │(Query Engine)   │
└──────────────┘  └─────────────────┘  └─────────────────┘
     │                     │                     │
     └─────────────────────┼─────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ culinary-graph-viz.js   │
              │ (Graph Visualization)   │
              └─────────────────────────┘
              ┌─────────────────────────┐
              │culinary-substitution.js │
              │ (Swap Logic)            │
              └─────────────────────────┘
              ┌─────────────────────────┐
              │ culinary-ai-config.js   │
              │ (Gemini Config)         │
              └─────────────────────────┘
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `culinarygraph.html` | Main page with chat interface, graph canvas, dietary restrictions UI |
| `culinary-chat.js` | Chat controller: message handling, LLM calls, RAG context injection |
| `culinary-graph-data.js` | Ontology definition + ingredient nodes + relationship edges |
| `culinary-graph-engine.js` | Query engine: finds substitutes, pairings, paths between ingredients |
| `culinary-graph-viz.js` | Canvas rendering for ingredient graph visualization |
| `culinary-substitution.js` | Substitution logic with ratios and context rules |
| `culinary-ai-config.js` | Gemini 2.5 Flash config and system prompt |

## Key Data Structures

### Ontology (culinary-graph-data.js)
```javascript
culinaryOntology = {
    classes: { 'Ingredient', 'Protein', 'Recipe', 'Technique' },
    relationshipTypes: {
        'SUBSTITUTES': { ratio, context, confidence },
        'PAIRS_WITH': { strength, confidence },
        'SIMILAR_TO': { similarity_type }
    },
    dietaryRestrictions: ['vegan', 'gluten-free', 'nut-free', ...],
    cuisineTypes: ['italian', 'mexican', 'asian', ...]
}
```

### Ingredient Node
```javascript
{
    id: 'butter',
    label: 'Butter',
    category: 'dairy',
    cuisine: ['all'],
    dietary: ['vegetarian']
}
```

### Substitution Edge
```javascript
{
    source: 'butter',
    target: 'coconut_oil',
    type: 'SUBSTITUTES',
    ratio: '1:1',
    context: 'baking',
    confidence: 0.85
}
```

## How It Works

1. **User asks** → "What can I use instead of butter?"
2. **Graph query** → Engine traverses edges to find SUBSTITUTES relationships
3. **Filter** → Applies user's dietary restrictions (stored in localStorage)
4. **RAG context** → Relevant substitutes + ratios injected into LLM prompt
5. **LLM response** → Gemini formats answer with context and cooking tips

## Dietary Restrictions System

- User adds restrictions via chips UI (e.g., "dairy-free", "vegan")
- Saved to localStorage, persists across sessions
- Filters out incompatible substitutes before showing results
- Injected into LLM system prompt for context-aware responses

## Relationship Types

| Type | Description | Properties |
|------|-------------|------------|
| SUBSTITUTES | One ingredient can replace another | ratio, context, confidence |
| PAIRS_WITH | Flavor pairing (bidirectional) | strength, confidence |
| USED_WITH | Commonly cooked together | context, frequency |
| SIMILAR_TO | Similar properties/flavor | similarity_type |

## Backend Integration

- **Proxy**: Same Render.com backend as Saponify AI
- **Model**: Gemini 2.5 Flash
- **Retry**: Exponential backoff on failures

## Styling

- Same pixel-art theme as portfolio
- Graph visualization with colored category nodes
- Interactive: click nodes, drag to explore relationships
