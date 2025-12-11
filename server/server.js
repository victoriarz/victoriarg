// Saponify AI Backend Proxy Server
// This server securely handles Gemini API calls without exposing the API key to clients

const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Security headers
app.use(helmet());

// CORS configuration - restrict to production domain
const corsOptions = {
    origin: process.env.NODE_ENV === 'production'
        ? ['https://victoriarg.com', 'https://www.victoriarg.com']
        : '*',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'X-Request-ID'],
    credentials: false
};
app.use(cors(corsOptions));

// Body parser
app.use(express.json({ limit: '10kb' })); // Limit payload size

// Rate limiter - 30 requests per minute per IP
const apiLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 30, // 30 requests per minute
    message: {
        error: 'Too many requests from this IP, please try again in a minute.',
        retryAfter: 60
    },
    standardHeaders: true,
    legacyHeaders: false,
    // Skip rate limiting for health checks
    skip: (req) => req.path === '/health'
});

// Apply rate limiting to API routes
app.use('/api/', apiLimiter);

// Request logging middleware
app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.path} - IP: ${req.ip}`);
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', message: 'Saponify AI proxy server is running' });
});

// Proxy endpoint for Gemini API
app.post('/api/chat', async (req, res) => {
    const startTime = Date.now();

    try {
        const { contents, generationConfig } = req.body;

        // Validate request
        if (!contents || !Array.isArray(contents)) {
            return res.status(400).json({
                error: 'Invalid request: contents array is required'
            });
        }

        // Validate contents array length (prevent abuse)
        if (contents.length > 50) {
            return res.status(400).json({
                error: 'Conversation too long. Maximum 50 messages allowed.'
            });
        }

        // Validate each message in contents
        for (const msg of contents) {
            if (!msg.role || !msg.parts || !Array.isArray(msg.parts)) {
                return res.status(400).json({
                    error: 'Invalid message format in contents array'
                });
            }

            // Check message length (prevent token abuse)
            const textLength = msg.parts.reduce((sum, part) =>
                sum + (part.text?.length || 0), 0);

            if (textLength > 10000) {
                return res.status(400).json({
                    error: 'Message too long. Maximum 10,000 characters per message.'
                });
            }
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

        // Make request to Gemini API with timeout
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 30000); // 30 second timeout

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
            }),
            signal: controller.signal
        });

        clearTimeout(timeout);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Gemini API error:', errorData);

            // Handle specific error cases
            if (response.status === 429) {
                return res.status(429).json({
                    error: 'API rate limit exceeded. Please try again in a moment.',
                    retryAfter: 60
                });
            }

            return res.status(response.status).json({
                error: `Gemini API error: ${errorData.error?.message || response.statusText}`
            });
        }

        const data = await response.json();

        // Log response time
        const responseTime = Date.now() - startTime;
        console.log(`Request completed in ${responseTime}ms`);

        res.json(data);

    } catch (error) {
        // Handle timeout
        if (error.name === 'AbortError') {
            console.error('Request timeout');
            return res.status(504).json({
                error: 'Request timeout. The AI took too long to respond. Please try again.',
                timeout: true
            });
        }

        console.error('Server error:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: process.env.NODE_ENV === 'production'
                ? 'An unexpected error occurred'
                : error.message
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Saponify AI proxy server running on port ${PORT}`);
    console.log(`🔑 API key configured: ${process.env.GEMINI_API_KEY ? 'Yes' : 'No'}`);
    console.log(`🤖 Model: ${process.env.GEMINI_MODEL || 'gemini-2.5-flash'}`);
});
