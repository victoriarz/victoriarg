// Saponify AI Backend Proxy Server
// This server securely handles Gemini API calls without exposing the API key to clients

const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', message: 'Saponify AI proxy server is running' });
});

// Proxy endpoint for Gemini API
app.post('/api/chat', async (req, res) => {
    try {
        const { contents, generationConfig } = req.body;

        // Validate request
        if (!contents || !Array.isArray(contents)) {
            return res.status(400).json({
                error: 'Invalid request: contents array is required'
            });
        }

        // Get API key from environment variable
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            console.error('GEMINI_API_KEY not found in environment variables');
            return res.status(500).json({
                error: 'Server configuration error: API key not configured'
            });
        }

        const model = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

        // Make request to Gemini API
        const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

        const response = await fetch(geminiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents,
                generationConfig: generationConfig || {
                    temperature: 0.7,
                    maxOutputTokens: 2048  // Increased default for full recipe outputs
                }
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Gemini API error:', errorData);
            return res.status(response.status).json({
                error: `Gemini API error: ${errorData.error?.message || response.statusText}`
            });
        }

        const data = await response.json();
        res.json(data);

    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Saponify AI proxy server running on port ${PORT}`);
    console.log(`🔑 API key configured: ${process.env.GEMINI_API_KEY ? 'Yes' : 'No'}`);
    console.log(`🤖 Model: ${process.env.GEMINI_MODEL || 'gemini-2.5-flash'}`);
});
