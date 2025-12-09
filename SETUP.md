# Saponify AI Setup Guide

This guide will help you set up the Saponify AI chat feature with a secure backend that protects your API key.

## Overview

The Saponify AI chat uses a **two-tier architecture**:
- **Frontend**: HTML/CSS/JavaScript (hosted on GitHub Pages)
- **Backend**: Node.js/Express proxy server (hosted separately)

This setup keeps your Gemini API key secure on the backend, preventing exposure in client-side code.

---

## Quick Start

### Step 1: Set Up the Backend Server

1. **Navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

4. **Add your Gemini API key to `.env`:**
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   PORT=3000
   ```

   Get your FREE Gemini API key at: https://aistudio.google.com/app/apikey

5. **Start the server:**
   ```bash
   npm start
   ```

   The server will run on `http://localhost:3000`

### Step 2: Test Locally

1. Open [saponifyai.html](saponifyai.html) in your browser
2. The chat should connect to your local backend automatically
3. Try asking a question like "What is saponification?"

---

## Deployment to Production

For your live website, you need to deploy the backend to a cloud platform. Here are the recommended options:

### Option A: Render.com (Recommended - Free & Easy)

1. **Push your code to GitHub** (if not already done)

2. **Sign up at [render.com](https://render.com)**

3. **Create a new Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name:** `saponify-ai-backend` (or your choice)
     - **Root Directory:** `server`
     - **Environment:** `Node`
     - **Build Command:** `npm install`
     - **Start Command:** `npm start`

4. **Add environment variable:**
   - In the Render dashboard, go to "Environment"
   - Add: `GEMINI_API_KEY` = `your_api_key_here`
   - Add: `GEMINI_MODEL` = `gemini-2.5-flash`

5. **Deploy!** Render will give you a URL like:
   ```
   https://saponify-ai-backend.onrender.com
   ```

6. **Update the frontend:**
   - Open [ai-config.js](ai-config.js)
   - Change line 9:
     ```javascript
     this.backendUrl = 'https://saponify-ai-backend.onrender.com';
     ```
   - Commit and push to GitHub

7. **Configure CORS (important!):**
   - Open `server/server.js`
   - Update the CORS configuration to only allow your domain:
     ```javascript
     app.use(cors({
       origin: 'https://victoriarg.com'
     }));
     ```

### Option B: Railway.app (Alternative - Also Free)

1. Sign up at [railway.app](https://railway.app)
2. Create new project from GitHub
3. Set root directory to `server/`
4. Add environment variable `GEMINI_API_KEY`
5. Deploy and get your URL
6. Update `backendUrl` in [ai-config.js](ai-config.js)

### Option C: Vercel (Serverless)

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root
3. Follow prompts to deploy
4. Add `GEMINI_API_KEY` in Vercel dashboard
5. Update `backendUrl` in [ai-config.js](ai-config.js)

---

## Configuration

### Backend Configuration (server/.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | `AIza...` |
| `GEMINI_MODEL` | Model to use | `gemini-2.5-flash` |
| `PORT` | Server port (optional) | `3000` |

### Frontend Configuration (ai-config.js)

Update `this.backendUrl` on line 9:
- **Local development:** `http://localhost:3000`
- **Production:** Your deployed backend URL (e.g., `https://your-app.onrender.com`)

---

## Security Best Practices

✅ **DO:**
- Keep `.env` file in `.gitignore`
- Use environment variables for sensitive data
- Configure CORS to only allow your domain in production
- Regularly rotate your API keys

❌ **DON'T:**
- Commit `.env` file to Git
- Hardcode API keys in source code
- Allow CORS from all origins (`*`) in production
- Share your API keys publicly

---

## Troubleshooting

### Chat shows "AI: Offline (Local Mode)"
- Backend server is not running or not accessible
- Check that backend URL in `ai-config.js` is correct
- Verify backend server is running: visit `https://your-backend-url/health`

### CORS Errors in Browser Console
- Update CORS configuration in `server/server.js`
- Ensure your domain is allowed in CORS settings

### "Backend API error" messages
- Check backend server logs
- Verify `GEMINI_API_KEY` is set correctly in environment variables
- Ensure API key is valid and has proper permissions

### Backend won't start locally
- Make sure `.env` file exists in `server/` directory
- Verify Node.js version is 18+: `node --version`
- Check that all dependencies are installed: `npm install`

---

## Testing

### Test Backend Health
```bash
curl http://localhost:3000/health
```

Expected response:
```json
{"status":"ok","message":"Saponify AI proxy server is running"}
```

### Test Chat Endpoint
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400}
  }'
```

---

## Costs & Limits

### Gemini 2.5 Flash (FREE Tier)
- **Requests:** 15 requests per minute
- **Tokens:** 1 million tokens per day
- **Cost:** $0 (completely free!)

This is more than enough for a personal portfolio website.

### Hosting Costs
- **Render.com Free Tier:** $0/month (sleeps after inactivity)
- **Railway.app Free Tier:** $0/month with limited hours
- **Vercel Free Tier:** $0/month (serverless)

All options have generous free tiers suitable for portfolio sites.

---

## Next Steps

1. ✅ Set up local backend and test
2. ✅ Deploy backend to Render/Railway/Vercel
3. ✅ Update frontend with production backend URL
4. ✅ Configure CORS for your domain
5. ✅ Test on live site
6. ✅ Monitor usage in Gemini API dashboard

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review server logs for errors
3. Verify all configuration values
4. Test backend health endpoint

For Gemini API issues, visit: https://ai.google.dev/gemini-api/docs

---

**Last Updated:** 2025-12-08
