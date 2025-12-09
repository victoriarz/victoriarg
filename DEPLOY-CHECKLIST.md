# 🚀 Deployment Checklist - Quick Reference

Follow these steps in order. Check off each one as you complete it.

---

## Part 1: Push Code to GitHub

```bash
cd /Users/garrettgriffith/Downloads/vicg/victoriarg
git status
git add server/ ai-config.js soap-chat.js saponifyai.html .gitignore SETUP.md
git commit -m "Add secure backend proxy server for Saponify AI"
git push origin main
```

- [ ] Code pushed to GitHub successfully

---

## Part 2: Deploy to Render.com

### A. Sign Up
- [ ] Go to https://render.com
- [ ] Click "Get Started"
- [ ] Sign up with GitHub

### B. Create Web Service
- [ ] Click "New +" button (top right)
- [ ] Select "Web Service"
- [ ] Connect "victoriarg" repository

### C. Configure Service

Fill in these exact values:

| Setting | Value |
|---------|-------|
| Name | `saponify-ai-backend` |
| Root Directory | `server` ⚠️ IMPORTANT |
| Build Command | `npm install` |
| Start Command | `npm start` |
| Instance Type | Free |

- [ ] All settings configured correctly

### D. Add Environment Variables

Click "Add Environment Variable" and add these:

1. **First variable:**
   - Key: `GEMINI_API_KEY`
   - Value: `AIzaSyCwOqh0cqvUXzcJxbnvFNyLNJD3kI39_0k`

2. **Second variable:**
   - Key: `GEMINI_MODEL`
   - Value: `gemini-2.5-flash`

- [ ] Both environment variables added

### E. Deploy
- [ ] Click "Create Web Service"
- [ ] Wait for deployment to complete (2-5 minutes)
- [ ] Status shows "Live" with green dot

### F. Copy Your URL
- [ ] Copy your Render URL (e.g., `https://saponify-ai-backend.onrender.com`)

### G. Test Backend
- [ ] Visit: `https://YOUR-URL/health`
- [ ] See success message: `{"status":"ok",...}`

---

## Part 3: Update Frontend

### A. Edit ai-config.js
- [ ] Open `ai-config.js` in your editor
- [ ] Find line 9: `this.backendUrl = 'http://localhost:3000';`
- [ ] Change to: `this.backendUrl = 'https://YOUR-RENDER-URL';`
- [ ] Replace YOUR-RENDER-URL with the actual URL from Render
- [ ] Save file

### B. Push Update
```bash
git add ai-config.js
git commit -m "Update frontend to use production backend"
git push origin main
```

- [ ] Frontend update pushed to GitHub
- [ ] GitHub Pages auto-deployed (wait 1-2 minutes)

---

## Part 4: Test Live Site

- [ ] Go to https://victoriarg.com/saponifyai.html
- [ ] Status badge shows: "🤖 AI: Gemini (Active)"
- [ ] Type a test message: "What is saponification?"
- [ ] Receive AI response (may take 30-60 sec first time)
- [ ] Chat works successfully! 🎉

---

## Troubleshooting

### If backend deploy fails:
1. Check Render logs tab for errors
2. Verify Root Directory is set to `server`
3. Verify environment variables are set

### If chat shows "Offline (Local Mode)":
1. Check backend URL in `ai-config.js` is correct
2. Verify backend /health endpoint works
3. Check browser console for CORS errors

### If first message is slow:
- Normal on free tier (backend "spins down" after 15 min)
- First request wakes it up (30-60 seconds)
- Subsequent requests are fast

---

## Quick Links

- **Render Dashboard:** https://dashboard.render.com
- **Your Site:** https://victoriarg.com/saponifyai.html
- **Full Guide:** See DEPLOYMENT-GUIDE.md
- **Backend Health:** `https://your-backend-url/health`

---

## Summary

**What you're doing:**
1. ✅ Push backend code to GitHub
2. ✅ Deploy backend to Render (free hosting)
3. ✅ Get backend URL from Render
4. ✅ Update frontend to use that URL
5. ✅ Test live site

**Total time:** ~15-20 minutes

**Cost:** $0 (completely free!)

---

Good luck! 🍀
