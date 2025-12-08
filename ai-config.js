// AI Configuration for Saponify AI Chat
// This file manages API keys and LLM provider settings

class AIConfig {
    constructor() {
        // Load API key from localStorage
        this.geminiKey = localStorage.getItem('saponify_gemini_key') || '';

        // Model settings
        this.geminiModel = 'gemini-2.5-flash'; // Latest Flash model - Fast, efficient, and FREE!

        // System prompt for soap making assistant
        this.systemPrompt = `You are a knowledgeable and friendly soap making assistant for Saponify AI. You help users with:
- Calculating custom soap recipes with proper lye and water amounts
- Explaining saponification and soap making chemistry
- Providing guidance on oils, fats, and their properties
- Troubleshooting soap making issues
- Recommending techniques for cold process and hot process soap making
- Advising on essential oils, colorants, and additives
- Safety instructions for working with lye

You have access to SAP (saponification) values for common oils. When calculating recipes, you should:
1. Ask for batch size in grams
2. Ask what oils they want to use and in what amounts
3. Calculate the exact lye (NaOH) needed using SAP values
4. Calculate water (typically 2.5:1 water to lye ratio)
5. Apply superfat discount (usually 5-8%)
6. Always include safety warnings

Be conversational, helpful, and enthusiastic about soap making. Keep responses concise but informative. Always prioritize safety when discussing lye handling.`;

        // Common SAP values reference for the AI
        this.sapValuesReference = `
Common SAP Values (NaOH per oz of oil):
- Olive: 0.135
- Coconut: 0.184
- Palm: 0.141
- Castor: 0.129
- Sweet Almond: 0.134
- Avocado: 0.137
- Shea Butter: 0.128
- Cocoa Butter: 0.137
- Sunflower: 0.136
- Jojoba: 0.065
- Hemp: 0.138
- Lard: 0.138
- Tallow: 0.14
`;
    }

    // Save API key to localStorage
    saveKey(geminiKey) {
        if (geminiKey) {
            this.geminiKey = geminiKey;
            localStorage.setItem('saponify_gemini_key', geminiKey);
        }
    }

    // Check if API key is configured
    hasValidKey() {
        return this.geminiKey.length > 0;
    }

    // Get the system prompt with SAP values
    getSystemPrompt() {
        return this.systemPrompt + '\n\n' + this.sapValuesReference;
    }

    // Clear stored key
    clearKey() {
        this.geminiKey = '';
        localStorage.removeItem('saponify_gemini_key');
    }
}

// Export as global
window.AIConfig = AIConfig;
