# Saponify AI Architecture

## How It Works Now (Secure Setup)

```
┌─────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  https://victoriarg.com/saponifyai.html            │    │
│  │                                                     │    │
│  │  - User types message in chat                      │    │
│  │  - JavaScript calls your backend API               │    │
│  │  - NO API key in browser (secure!)                 │    │
│  └─────────────────┬──────────────────────────────────┘    │
│                    │                                         │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     │ HTTPS Request
                     │ POST /api/chat
                     │ (No API key exposed)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          YOUR BACKEND (Render.com)                          │
│     https://saponify-ai-backend.onrender.com                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  server.js (Node.js/Express)                       │    │
│  │                                                     │    │
│  │  - Receives request from frontend                  │    │
│  │  - Gets API key from environment variable          │    │
│  │  - Adds API key to request                         │    │
│  │  - Forwards to Gemini API                          │    │
│  └─────────────────┬──────────────────────────────────┘    │
│                    │                                         │
│  Environment Variables (Secure):                            │
│  GEMINI_API_KEY=AIza...                                     │
│  GEMINI_MODEL=gemini-2.5-flash                              │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     │ HTTPS Request
                     │ With API key
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               GOOGLE GEMINI API                             │
│      https://generativelanguage.googleapis.com              │
│                                                              │
│  - Receives request with API key                            │
│  - Processes AI request                                     │
│  - Returns response                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                     │
                     │ Response flows back
                     ▼
              Backend receives response
                     │
                     ▼
              Frontend receives response
                     │
                     ▼
              User sees AI message in chat 🎉
```

---

## How It Worked Before (Insecure)

```
┌─────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                       │
│                                                              │
│  - User had to input their own API key                      │
│  - API key stored in browser (localStorage)                 │
│  - ❌ API key visible in browser dev tools                  │
│  - ❌ Anyone could view source and see the key              │
│  - Called Gemini API directly from browser                  │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Direct call with API key exposed
                     ▼
              Google Gemini API
```

---

## Key Components

### Frontend (GitHub Pages)
- **Location:** `https://victoriarg.com`
- **Files:**
  - `saponifyai.html` - Chat UI
  - `soap-chat.js` - Chat logic
  - `ai-config.js` - Backend URL configuration
- **What it does:**
  - Displays chat interface
  - Sends user messages to backend
  - Shows AI responses

### Backend (Render.com)
- **Location:** `https://saponify-ai-backend.onrender.com`
- **Files:**
  - `server/server.js` - Express server
  - `server/.env` - Environment variables (secret)
- **What it does:**
  - Receives chat requests from frontend
  - Securely adds API key
  - Forwards to Gemini API
  - Returns responses to frontend

### Gemini API (Google)
- **Location:** Google's servers
- **What it does:**
  - Processes AI chat requests
  - Generates intelligent responses
  - Returns results

---

## Data Flow Example

**User asks: "What is saponification?"**

```
1. User types in browser
   └─> saponifyai.html (frontend)

2. JavaScript sends request
   └─> POST https://saponify-ai-backend.onrender.com/api/chat
       Body: { contents: [{ role: "user", parts: [{ text: "What is saponification?" }] }] }

3. Backend receives request
   └─> server.js
       - Gets GEMINI_API_KEY from environment
       - Adds API key to request

4. Backend forwards to Gemini
   └─> POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
       With API key in URL: ?key=AIza...

5. Gemini processes and responds
   └─> Returns AI-generated answer about saponification

6. Backend receives Gemini response
   └─> Sends back to frontend

7. Frontend displays response
   └─> User sees AI answer in chat!
```

---

## Security Comparison

### Before (Insecure):
```
Browser → Gemini API
   ↑
API key visible here ❌
```

### After (Secure):
```
Browser → Backend → Gemini API
              ↑
        API key hidden here ✅
```

---

## Deployment Locations

### Local Development:
- **Frontend:** File system (`file:///...saponifyai.html`)
- **Backend:** `http://localhost:3000`

### Production:
- **Frontend:** `https://victoriarg.com` (GitHub Pages)
- **Backend:** `https://saponify-ai-backend.onrender.com` (Render)

---

## Why This Architecture?

✅ **Security:** API key never exposed to users
✅ **Cost:** Completely free (Gemini + Render free tiers)
✅ **Simple:** No complex infrastructure needed
✅ **Scalable:** Can handle thousands of users
✅ **Maintainable:** Clear separation of concerns

---

## Free Tier Limits

### Gemini 2.5 Flash:
- 15 requests per minute
- 1 million tokens per day
- **Cost:** $0

### Render.com:
- 750 hours per month
- Spins down after 15 min of inactivity
- **Cost:** $0

### GitHub Pages:
- Unlimited requests
- 100 GB bandwidth/month
- **Cost:** $0

**Total monthly cost:** $0! 🎉

---

This architecture gives you enterprise-level security at zero cost!
