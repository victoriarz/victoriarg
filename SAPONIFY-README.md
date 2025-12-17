# Saponify AI - Architecture Overview

## What It Is
An AI-powered soap making assistant with a built-in SoapCalc-style calculator. Users can chat naturally to get safe, accurate soap recipes with exact lye calculations.

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    saponifyai.html                       │
│                   (Main UI + Styles)                     │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  soap-chat.js   │ │ ai-config.js    │ │soap-knowledge-  │
│ (Chat Logic)    │ │ (Gemini Config) │ │  bank.js        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                │                    │
          └────────────────┼────────────────────┘
                           ▼
               ┌─────────────────────┐
               │ soap-calculator.js  │
               │ (SAP Value Engine)  │
               └─────────────────────┘
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `saponifyai.html` | Main page with pixel-art UI, chat interface, all CSS inline |
| `soap-chat.js` | Chat controller: message handling, LLM calls, recipe state management |
| `ai-config.js` | Gemini 2.5 Flash config, system prompt, backend proxy URL |
| `soap-calculator.js` | Core engine: 19+ oils with SAP values, fatty acid profiles, property calculations |
| `soap-knowledge-bank.js` | RAG knowledge base: saponification chemistry, methods, troubleshooting |

## Key Data Structures

### Recipe State (soap-chat.js:16-22)
```javascript
let recipeState = {
    active: false,
    batchSizeGrams: 0,
    oils: [],           // {name, grams, percentage}
    superfat: 5
};
```

### Oil Database (soap-calculator.js)
```javascript
oilsDatabase = {
    'olive': {
        sapNaOH: 0.1353,    // Lye needed per gram
        sapKOH: 0.1899,
        properties: { hardness: 15, cleansing: 0, conditioning: 82, ... }
    }
}
```

## How It Works

1. **User asks for recipe** → `soap-chat.js` detects intent
2. **Gather requirements** → Batch size, oils, superfat percentage
3. **Calculate** → `SoapCalculator.calculate()` with verified SAP values
4. **Augment response** → Knowledge bank adds chemistry context
5. **LLM formats** → Gemini structures the final response with markdown

## Backend Integration

- **Proxy**: `https://saponify-ai-backend.onrender.com`
- **Why proxy**: Gemini API key stored server-side for security
- **Retry logic**: Exponential backoff (2s, 4s, 8s, 16s) on failures

## Safety Features

- Never guesses lye amounts - always calculates from SAP values
- Warns about high coconut oil (drying), missing hard oils (soft bars)
- Recipe draft auto-saves to localStorage (restores within 1 hour)

## Styling

- Pixel-art theme with VT323 and Press Start 2P fonts
- Bubble animations for soap-like aesthetic
- All CSS inline in `saponifyai.html` (no external stylesheet)
