# Production Deployment Guide - Step by Step

This guide will walk you through deploying your Saponify AI backend to production.

---

## Option 1: Render.com (Recommended - Easiest)

### Step 1: Prepare Your Code for GitHub

First, let's commit and push your code to GitHub:

```bash
# Make sure you're in the victoriarg directory
cd /Users/garrettgriffith/Downloads/vicg/victoriarg

# Check what files will be committed
git status

# Add all the new backend files
git add server/ ai-config.js soap-chat.js saponifyai.html .gitignore SETUP.md start-backend.sh

# Commit the changes
git commit -m "Add secure backend proxy server for Saponify AI chat

- Create Node.js/Express backend to securely handle Gemini API calls
- Update frontend to use backend proxy instead of exposing API key
- Remove API key settings modal from UI
- Add deployment documentation and setup guides"

# Push to GitHub
git push origin main
```

### Step 2: Sign Up for Render

1. Go to **https://render.com**
2. Click **"Get Started"** (top right)
3. Sign up with one of these options:
   - **GitHub** (recommended - easiest integration)
   - **GitLab**
   - **Email**

4. If using GitHub, authorize Render to access your repositories

### Step 3: Create a New Web Service

1. Once logged in, click the **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see "Create a new Web Service" page

### Step 4: Connect Your Repository

**If you signed up with GitHub:**
1. You'll see a list of your repositories
2. Find **"victoriarg"** in the list
3. Click **"Connect"** button next to it

**If you didn't sign up with GitHub:**
1. Click **"Connect account"** under GitHub
2. Authorize Render to access GitHub
3. Find and connect **"victoriarg"** repository

### Step 5: Configure the Web Service

Fill in these fields:

| Field | Value | Notes |
|-------|-------|-------|
| **Name** | `saponify-ai-backend` | Or any name you prefer |
| **Region** | `Oregon (US West)` | Choose closest to your users |
| **Branch** | `main` | Your main branch |
| **Root Directory** | `server` | **IMPORTANT:** This tells Render where your backend code is |
| **Runtime** | `Node` | Should auto-detect |
| **Build Command** | `npm install` | Should auto-fill |
| **Start Command** | `npm start` | Should auto-fill |
| **Instance Type** | `Free` | Select the free tier |

Click **"Advanced"** to expand more options (optional for now)

### Step 6: Add Environment Variables

**CRITICAL STEP - Don't skip this!**

Scroll down to the **"Environment Variables"** section:

1. Click **"Add Environment Variable"** button
2. Add the first variable:
   - **Key:** `GEMINI_API_KEY`
   - **Value:** `AIzaSyCwOqh0cqvUXzcJxbnvFNyLNJD3kI39_0k`

3. Click **"Add Environment Variable"** again
4. Add the second variable:
   - **Key:** `GEMINI_MODEL`
   - **Value:** `gemini-2.5-flash`

### Step 7: Create the Web Service

1. Scroll to the bottom
2. Click the blue **"Create Web Service"** button
3. Wait for deployment (usually 2-5 minutes)

You'll see a deployment log showing:
```
==> Downloading cache...
==> Installing dependencies...
==> Building...
==> Starting service...
```

### Step 8: Get Your Backend URL

Once deployed successfully:

1. You'll see **"Live"** with a green dot at the top
2. Your URL will be displayed at the top, something like:
   ```
   https://saponify-ai-backend.onrender.com
   ```
3. **Copy this URL** - you'll need it for the next step

### Step 9: Test Your Backend

Click on your backend URL or visit it in a browser:
```
https://saponify-ai-backend.onrender.com/health
```

You should see:
```json
{"status":"ok","message":"Saponify AI proxy server is running"}
```

If you see this, **your backend is live!** 🎉

### Step 10: Update Your Frontend

Now we need to tell your frontend to use the production backend:

1. Open `ai-config.js` in your code editor
2. Find line 9 (the `backendUrl` line)
3. Change it from:
   ```javascript
   this.backendUrl = 'http://localhost:3000';
   ```

   To (use YOUR actual Render URL):
   ```javascript
   this.backendUrl = 'https://saponify-ai-backend.onrender.com';
   ```

4. Save the file

### Step 11: Commit and Push Frontend Update

```bash
# Commit the change
git add ai-config.js
git commit -m "Update frontend to use production backend on Render"

# Push to GitHub
git push origin main
```

### Step 12: Test Your Live Site

1. Go to **https://victoriarg.com/saponifyai.html**
2. The status badge should show: **"🤖 AI: Gemini (Active)"**
3. Try asking a question in the chat
4. You should get AI responses! 🎉

---

## Important: Free Tier Limitations

**Render Free Tier:**
- Your backend will "spin down" after 15 minutes of inactivity
- First request after spin down takes 30-60 seconds to wake up
- Subsequent requests are fast
- **This is normal and expected on the free tier**

**To improve this:**
- Upgrade to paid tier ($7/month) for always-on
- Or use a service like UptimeRobot to ping your backend every 10 minutes

---

## Troubleshooting Render Deployment

### "Build Failed" Error

**Check the logs in Render dashboard:**
1. Click on your service
2. Look at the **"Logs"** tab
3. Find the error message

**Common fixes:**
- Ensure `Root Directory` is set to `server`
- Check that `package.json` exists in `server/` folder
- Verify Node version requirements

### "Deploy succeeded but health check fails"

1. Check environment variables are set correctly:
   - Go to "Environment" tab in Render
   - Verify `GEMINI_API_KEY` exists and has the right value

2. Check the logs for errors:
   - Look for "GEMINI_API_KEY not found" messages

### CORS Errors in Browser Console

Your backend needs to allow requests from your domain:

1. Edit `server/server.js`
2. Find the line: `app.use(cors());`
3. Change it to:
   ```javascript
   app.use(cors({
     origin: 'https://victoriarg.com'
   }));
   ```
4. Commit and push - Render will auto-deploy

---

## Option 2: Railway.app (Alternative)

Railway is similar to Render but with a different interface.

### Step 1: Sign Up

1. Go to **https://railway.app**
2. Click **"Start a New Project"**
3. Sign in with **GitHub**

### Step 2: Create New Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose **"victoriarg"** repository

### Step 3: Configure Project

1. Railway will auto-detect it's a Node.js project
2. Click **"Variables"** tab
3. Click **"+ New Variable"**
4. Add:
   - `GEMINI_API_KEY` = `AIzaSyCwOqh0cqvUXzcJxbnvFNyLNJD3kI39_0k`
   - `GEMINI_MODEL` = `gemini-2.5-flash`
   - `ROOT` = `server` (tells Railway where the app is)

### Step 4: Configure Build Settings

1. Click **"Settings"** tab
2. Set **"Root Directory"** to `server`
3. Set **"Start Command"** to `npm start`

### Step 5: Deploy

1. Click **"Deploy"**
2. Wait for build to complete
3. You'll get a URL like: `https://saponify-ai-backend.up.railway.app`

### Step 6: Update Frontend

Same as Render - update `ai-config.js` line 9 with your Railway URL.

---

## Option 3: Vercel (Serverless)

Vercel uses serverless functions. Requires slight code restructuring.

### Quick Setup

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Login:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   cd server
   vercel
   ```

4. Follow prompts:
   - **Set up and deploy?** Yes
   - **Which scope?** Your account
   - **Link to existing project?** No
   - **Project name?** saponify-ai-backend
   - **Directory?** `./`

5. Add environment variable:
   ```bash
   vercel env add GEMINI_API_KEY
   ```
   Paste your API key when prompted

6. Deploy to production:
   ```bash
   vercel --prod
   ```

7. Get your URL (shown in terminal) and update `ai-config.js`

---

## Recommended: Render.com

**Why Render?**
- ✅ Easiest for beginners
- ✅ Free tier with no time limit
- ✅ Auto-deploys on git push
- ✅ Simple dashboard
- ✅ Built-in logging

**Why not Railway?**
- Free tier has limited hours per month
- More complex pricing

**Why not Vercel?**
- Requires code restructuring for serverless
- More complex for beginners

---

## After Deployment Checklist

- [ ] Backend deployed and shows green "Live" status
- [ ] Health endpoint returns success: `/health`
- [ ] Environment variables set correctly
- [ ] Frontend `ai-config.js` updated with production URL
- [ ] Changes committed and pushed to GitHub
- [ ] Live site tested at victoriarg.com/saponifyai.html
- [ ] AI chat shows "Gemini (Active)" status
- [ ] Test chat sends/receives messages successfully

---

## Need Help?

**Render Support:**
- Documentation: https://render.com/docs
- Community: https://community.render.com

**Railway Support:**
- Documentation: https://docs.railway.app
- Discord: https://discord.gg/railway

**Vercel Support:**
- Documentation: https://vercel.com/docs
- Discord: https://vercel.com/discord

**Check Your Deployment:**
- Visit: `https://your-backend-url/health`
- Should return: `{"status":"ok","message":"Saponify AI proxy server is running"}`

---

Good luck with your deployment! 🚀
