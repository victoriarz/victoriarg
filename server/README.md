# Saponify AI Backend Proxy Server

This is a secure backend server that proxies requests to the Google Gemini API, keeping your API key safe and hidden from client-side code.

## Setup Instructions

### 1. Install Dependencies

```bash
cd server
npm install
```

### 2. Configure Environment Variables

Create a `.env` file in the `server/` directory:

```bash
cp .env.example .env
```

Then edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.5-flash
PORT=3000
```

### 3. Run the Server

**Development mode (with auto-restart):**
```bash
npm run dev
```

**Production mode:**
```bash
npm start
```

The server will start on `http://localhost:3000` by default.

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server status

### Chat Endpoint
- **POST** `/api/chat`
- Proxies requests to Gemini API
- Request body:
  ```json
  {
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Your message here"}]
      }
    ],
    "generationConfig": {
      "temperature": 0.7,
      "maxOutputTokens": 400
    }
  }
  ```

## Deployment Options

### Option 1: Deploy to Render.com (Free)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Create a new "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `cd server && npm install`
   - **Start Command:** `cd server && npm start`
   - **Environment Variables:** Add `GEMINI_API_KEY` with your API key
6. Deploy!

### Option 2: Deploy to Railway.app (Free)

1. Go to [railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Set root directory to `server/`
4. Add environment variable `GEMINI_API_KEY`
5. Deploy!

### Option 3: Deploy to Vercel (Serverless)

Vercel is great for serverless functions. You'll need to restructure slightly:

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project directory
3. Add `GEMINI_API_KEY` as an environment variable in Vercel dashboard

### Option 4: Run Locally (Development)

Just run `npm run dev` and keep your computer running. Use ngrok for public access:

```bash
npx ngrok http 3000
```

## Security Notes

- ✅ API key is stored in environment variables, never in code
- ✅ `.env` file is in `.gitignore` to prevent accidental commits
- ✅ CORS is enabled for your frontend domain
- ⚠️ For production, configure CORS to only allow your specific domain
- ⚠️ Consider adding rate limiting to prevent abuse

## Updating Frontend

After deploying your backend, update the frontend to use your backend URL instead of calling Gemini directly. The deployed URL will look like:

- Render: `https://your-app-name.onrender.com`
- Railway: `https://your-app-name.up.railway.app`
- Vercel: `https://your-app-name.vercel.app`

Update the `BACKEND_URL` in `ai-config.js` to point to your deployed backend.

## Troubleshooting

**Server won't start:**
- Check that `.env` file exists and has `GEMINI_API_KEY`
- Ensure Node.js version is 18 or higher: `node --version`

**CORS errors:**
- Update CORS configuration in `server.js` to allow your domain

**API errors:**
- Verify your Gemini API key is valid
- Check API key has proper permissions
- Review server logs for detailed error messages
