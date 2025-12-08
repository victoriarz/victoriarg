# Saponify AI - FREE LLM Integration Setup Guide

## Overview

The Saponify AI chat assistant now features **Google Gemini Flash 1.5** integration, providing intelligent conversational responses about soap making - **completely FREE!**

## Features

✅ **100% FREE**: Google Gemini has a generous free tier (15 req/min, 1M tokens/day)
✅ **Smart Reasoning**: Advanced AI understanding of soap making questions
✅ **Conversation Memory**: Maintains context across messages (last 20 messages)
✅ **Automatic Fallback**: Falls back to local knowledge base if API fails
✅ **Client-Side Integration**: Works with static GitHub Pages hosting
✅ **Simple Setup**: Just one API key needed!

## Quick Start

### 1. Get Your FREE Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

**That's it!** The free tier includes:
- 15 requests per minute
- 1 million tokens per day
- More than enough for a personal portfolio!

### 2. Configure the Chat

1. Visit the Saponify AI page: [victoriarg.com/saponifyai.html](https://victoriarg.com/saponifyai.html)
2. Click the **"⚙️ AI Settings"** button
3. Paste your Gemini API key
4. Click **"Save Settings"**

### 3. Start Chatting!

Once configured, the status badge will show: **🤖 AI: Gemini (FREE)**

The AI can now:
- Answer complex soap making questions naturally
- Calculate custom soap recipes with detailed explanations
- Provide troubleshooting advice with context
- Remember your conversation for better responses
- Automatically fall back to local knowledge if API is unavailable

## How It Works

### Architecture

```
User Message
    ↓
Try Google Gemini Flash 1.5 API
    ↓ (if fails)
Use Local Knowledge Base (keyword matching)
```

### AI Integration

- **Model**: Gemini Flash 1.5 (optimized for speed and efficiency)
- **Temperature**: 0.7 (balanced creativity and accuracy)
- **Max Tokens**: 800 per response
- **Context**: Last 20 messages (10 conversation exchanges)

### System Prompt

Gemini receives a comprehensive system prompt including:
- Role as a knowledgeable soap making assistant
- SAP (saponification) values for 20+ common oils
- Instructions for safe recipe calculations
- Lye safety guidelines
- Conversational response guidelines

## File Structure

```
victoriarg/
├── saponifyai.html          # Main page with chat UI and settings modal
├── soap-chat.js             # Chat logic with Gemini integration
├── ai-config.js             # API configuration and key management
├── style.css                # Styles including modal and AI controls
└── SAPONIFY_AI_SETUP.md     # This documentation
```

## API Details

### Google Gemini Integration
- **Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent`
- **Model**: `gemini-1.5-flash`
- **Temperature**: 0.7 (balanced creativity)
- **Max Tokens**: 800 per response
- **Context Window**: Includes conversation history

### Free Tier Limits
- **15 requests per minute** - More than enough for real-time chat
- **1 million tokens per day** - Thousands of conversations
- **Rate limiting**: Automatic, handled gracefully

## Security Considerations

⚠️ **Important**: This is a **demo implementation** with client-side API key storage.

### Current Setup (Demo)
- API key stored in browser `localStorage`
- Key visible in browser DevTools (F12)
- Key sent directly from client to Google Gemini

### For Production Use
Consider implementing:
1. **Backend Proxy**: Store key on a secure server
2. **Environment Variables**: Never commit keys to Git
3. **Rate Limiting**: Prevent API abuse
4. **Key Rotation**: Regularly update API keys
5. **User Authentication**: Limit who can use the API

## Troubleshooting

### "Sorry, I encountered an error" Message

**Possible causes:**
1. Invalid API key
2. API rate limit exceeded (15 req/min)
3. Network connectivity issues
4. Google AI service outage

**Solutions:**
- Verify your API key in settings
- Wait a moment if rate limited (rare)
- Check internet connection
- Clear key to use local knowledge base
- Check status: [status.cloud.google.com](https://status.cloud.google.com)

### Chat Showing "🤖 AI: Local Mode"

This means no API key is configured. The chat is using the built-in keyword-based responses.

**To enable AI mode:**
- Click "⚙️ AI Settings"
- Add your Gemini API key
- Click "Save Settings"

### Response Shows "Local Knowledge Base" Badge

This means Gemini API failed and the system fell back to keyword matching. This is normal for network issues and ensures the chat always works!

## Cost Analysis

### Google Gemini (Flash 1.5)
- **Pricing**: 100% FREE
- **Free Tier**:
  - 15 requests per minute
  - 1 million tokens per day
  - No credit card required
- **For portfolio use**: Completely FREE! ✨

The free tier is more than sufficient for:
- Personal portfolio demonstrations
- Testing and development
- Low-to-moderate traffic sites
- Educational projects

## Customization

### Change Model

Edit `ai-config.js`:

```javascript
this.geminiModel = 'gemini-1.5-pro'; // Use Pro instead of Flash (slower, more capable)
```

### Adjust Response Length

Edit `soap-chat.js`:

```javascript
// In callGemini function
maxOutputTokens: 1200 // Increase from 800 for longer responses
```

### Modify System Prompt

Edit the `systemPrompt` property in `ai-config.js` to customize the AI's personality, expertise, and response style.

### Adjust Context Window

Edit `soap-chat.js`:

```javascript
// In sendMessage function
if (conversationHistory.length > 30) {  // Increase from 20
    conversationHistory = conversationHistory.slice(-30);
}
```

## Why Gemini Flash 1.5?

### Advantages
✅ **100% Free** - No costs for portfolio use
✅ **Fast** - Flash model optimized for speed
✅ **Smart** - Excellent reasoning for complex questions
✅ **Generous Limits** - 15 req/min is plenty
✅ **No Credit Card** - Easy to get started
✅ **Multimodal** - Can handle images (future feature)

### Perfect For
- Personal portfolios
- Proof of concepts
- Educational projects
- Demonstrations
- Low-traffic apps

## Example Conversations

### Recipe Calculation
**User**: "I want to make a 500g batch of soap with olive oil and coconut oil"

**Gemini**: *Provides detailed recipe calculation with exact lye and water amounts, plus safety instructions*

### Troubleshooting
**User**: "My soap has white powder on it, is it ruined?"

**Gemini**: *Explains soda ash, confirms it's cosmetic only, provides solutions*

### Complex Questions
**User**: "What's the difference between cold process and hot process, and which is better for beginners?"

**Gemini**: *Detailed comparison with pros/cons and beginner recommendation*

## Credits

**Built with:**
- Google Gemini Flash 1.5 API (FREE!)
- Vanilla JavaScript (no frameworks)
- HTML5 & CSS3
- Hosted on GitHub Pages

**Part of the Victoria Ruiz Griffith portfolio**
[victoriarg.com](https://victoriarg.com)

## License

This is a portfolio demonstration project. API key and usage are the responsibility of the user.

---

**Last Updated**: 2025-12-08

**Free Tier Status**: ✅ Active and Generous!
