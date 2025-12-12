// Saponify AI - Soap Making Chat Assistant with LLM Integration

// Initialize AI configuration
let aiConfig;
let conversationHistory = [];
let soapCalculator = null;  // Will be initialized when SoapCalculator loads
let recipeValidator = null;  // Will be initialized when RecipeValidator loads

// NOTE: SAP values are now managed by SoapCalculator class in soap-calculator.js
// This duplicate data structure has been removed to avoid inconsistencies

// Conversation state for recipe building
let recipeState = {
    active: false,
    batchSizeGrams: 0,
    oils: [],
    superfat: 5
};

// Store last calculated recipe for quick wins
let lastCalculatedRecipe = null;
let lastCalculatedRecipeTime = null;

// Rate limiting state
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL_MS = 5000; // Minimum 5 seconds between API requests (increased for stability)
let requestQueue = [];
let isProcessingQueue = false;
let consecutiveRateLimitErrors = 0;
const MAX_RETRIES = 2;

// Knowledge base for soap making
const soapKnowledge = {
    'saponification': {
        keywords: ['saponification', 'process', 'chemical reaction', 'how does soap form'],
        response: "Saponification is the chemical reaction that occurs when fats or oils react with lye (sodium hydroxide for bar soap or potassium hydroxide for liquid soap). This process creates soap and glycerin. The lye breaks down the triglycerides in oils into fatty acid salts (soap) and glycerol. It's an exothermic reaction, meaning it produces heat. Safety is crucial - always add lye to water, never water to lye!"
    },
    'cold_process': {
        keywords: ['cold process', 'cp', 'cold soap', 'cp vs hp', 'difference between'],
        response: "Cold Process (CP) soap making doesn't require external heat after mixing. You mix oils and lye at around 100-110°F, pour into molds, and let it cure for 4-6 weeks. CP gives you more design freedom and smoother bars. Hot Process (HP) uses heat to speed saponification - the soap is ready to use in 1-2 weeks but has a rustic texture. CP is great for swirls and designs, while HP is faster if you need soap quickly!"
    },
    'first_batch': {
        keywords: ['first batch', 'beginner', 'start', 'new to', 'never made'],
        response: "For your first batch, start with a simple recipe: 16 oz olive oil, 16 oz coconut oil, 16 oz palm oil (or sustainable alternative), 6.7 oz distilled water, and 6.8 oz lye. You'll need: safety goggles, gloves, stick blender, heat-safe containers, thermometer, and silicone mold. Mix lye with water outside, let cool to 100-110°F. Melt oils to same temp, add lye solution, blend to trace, pour in mold. Cure 4-6 weeks. Safety first - always use protection!"
    },
    'oils': {
        keywords: ['oils', 'fats', 'butter', 'which oil', 'best oil'],
        response: "Different oils create different soap properties! Hard oils (coconut, palm): create hard bars and lather. Soft oils (olive, avocado, sweet almond): moisturizing and gentle. Luxury oils (castor, jojoba, shea butter): add special properties. A balanced recipe typically uses 60-70% hard oils, 25-35% soft oils, and 5% castor for bubbles. Coconut oil creates great lather but can dry skin over 30%. Olive oil (Castile soap) is super gentle but takes longer to cure!"
    },
    'essential_oils': {
        keywords: ['essential oil', 'fragrance', 'scent', 'smell', 'aroma'],
        response: "Essential oils add natural fragrance! Use 0.5-1 oz per pound of soap. Popular choices: Lavender (calming, anchors well), Peppermint (refreshing), Tea Tree (antimicrobial), Lemongrass (citrusy), Eucalyptus (spa-like). Some fade quickly (citrus), so pair them with anchoring oils like patchouli or cedarwood. Always check if an EO is skin-safe. Fragrance oils are synthetic alternatives that often last longer. Add at trace before pouring!"
    },
    'lye': {
        keywords: ['lye', 'sodium hydroxide', 'caustic', 'dangerous'],
        response: "Lye (sodium hydroxide/NaOH) is essential for soap making - there's no soap without it! It's caustic, so always use safety gear: goggles, gloves, long sleeves. Mix in well-ventilated area. ALWAYS add lye to water (not water to lye) to prevent explosive reaction. The lye calculator helps determine exact amounts needed. After saponification, no lye remains in properly made soap - it's all converted to soap molecules. Respect it, use it safely, and you'll make beautiful soap!"
    },
    'troubleshooting': {
        keywords: ['problem', 'wrong', 'failed', 'mistake', 'help', 'issue'],
        response: "Common issues: 1) Soap won't trace - check your scale, blend longer, or use stick blender. 2) Soap is crumbly - too much lye or not enough water. 3) Soap is soft/greasy - not enough lye or excess oils. 4) White powder (soda ash) - cosmetic only, wash off. 5) Orange spots - dreaded orange spot (DOS) from rancid oils. 6) Separation - oils and lye didn't mix properly, might be salvageable with rebatch. Always use a lye calculator and measure precisely!"
    },
    'recipe': {
        keywords: ['recipe', 'formula', 'how to make', 'lavender soap', 'oatmeal'],
        response: "Here's a versatile beginner recipe: 35% Olive Oil, 30% Coconut Oil, 25% Palm Oil (or Shea Butter), 10% Castor Oil. For a 2 lb batch: 11.2 oz olive oil, 9.6 oz coconut oil, 8 oz palm oil, 3.2 oz castor oil, 4.4 oz lye, 10.5 oz water. Add 1 oz lavender essential oil at trace. This creates a hard, bubbly bar with good moisturizing properties. Always run any recipe through a lye calculator before making!"
    },
    'curing': {
        keywords: ['cure', 'curing', 'how long', 'wait', 'ready'],
        response: "Curing is essential! While soap is technically ready after saponification (24-48 hours), curing for 4-6 weeks makes it better. During curing, excess water evaporates, making bars harder and longer-lasting. The soap also becomes milder as the pH drops. Store soap on a rack with good air circulation, turning occasionally. Test pH with a zap test (tongue touch - should not tingle) or pH strips (should be 9-10). Castile soap benefits from 6-12 month cure!"
    },
    'colorants': {
        keywords: ['color', 'colorant', 'pigment', 'dye', 'natural color'],
        response: "Natural colorants: Clays (white, pink, green), Activated charcoal (black/gray), Turmeric (yellow/gold), Paprika (peach/coral), Spirulina (green), Cocoa powder (brown), Alkanet root (purple). Micas and oxides give vibrant, stable colors. Add at trace and mix well. Natural colorants may fade or morph - test small batches first. Titanium dioxide creates white and brightens colors. For swirls, divide your batter and color each portion separately!"
    }
};

// LLM API Integration Functions

// Call backend proxy which handles Gemini API
async function callGemini(messages) {
    // Convert messages to Gemini format
    const contents = [];

    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg.role === 'system') {
            // Gemini doesn't have system role, prepend to first user message
            continue;
        }
        contents.push({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }]
        });
    }

    // Prepend system message to first user message
    if (contents.length > 0 && messages[0].role === 'system') {
        contents[0].parts[0].text = messages[0].content + '\n\n' + contents[0].parts[0].text;
    }

    // Call backend proxy instead of Gemini directly
    console.log('🌐 Calling backend API:', `${aiConfig.getBackendUrl()}/api/chat`);

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout (increased for Render free tier)

    try {
        const response = await fetch(
            `${aiConfig.getBackendUrl()}/api/chat`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: contents,
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 8192  // Increased to prevent cut-off responses (Gemini Flash supports up to 8192)
                    }
                }),
                signal: controller.signal
            }
        );

        clearTimeout(timeoutId);

        console.log('📡 Backend response status:', response.status, response.statusText);

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ Backend error:', errorData);
            throw new Error(`Backend API error: ${response.status} - ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.log('✅ Backend response received:', data);

        // Handle Gemini 2.5 response format (may not have parts in some responses)
        if (data.candidates && data.candidates[0]) {
            const candidate = data.candidates[0];

            // Try to get text from parts array - concatenate ALL parts to avoid cut-off
            if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                // Concatenate all text parts in case the response is split
                const fullText = candidate.content.parts
                    .map(part => part.text || '')
                    .join('');

                if (fullText) {
                    return fullText;
                }
            }

            // Fallback: check if there's text directly in content
            if (candidate.content && candidate.content.text) {
                return candidate.content.text;
            }

            // If no text found, throw error
            throw new Error('No text content in Gemini response');
        }

        throw new Error('Invalid response format from Gemini API');
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - backend took too long to respond');
        }
        throw error;
    }
}

// Main LLM call via backend proxy
async function getLLMResponse(userMessage) {
    // Build conversation history
    const messages = [
        { role: 'system', content: aiConfig.getSystemPrompt() },
        ...conversationHistory,
        { role: 'user', content: userMessage }
    ];

    // Call backend proxy (which securely handles the API key)
    const response = await callGemini(messages);
    return { response, provider: 'Gemini 2.5 Flash' };
}

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const suggestionButtons = document.querySelectorAll('.suggestion-btn');

// Render markdown to HTML with XSS protection
function renderMarkdown(text) {
    try {
        let html;

        if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
            // Use marked with inline options (v11+ compatible)
            html = marked.parse(text, {
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
        } else {
            // Fallback: simple replacements if marked is not available
            html = text
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
        }

        // Sanitize HTML to prevent XSS attacks
        if (typeof DOMPurify !== 'undefined') {
            return DOMPurify.sanitize(html, {
                ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4',
                              'code', 'pre', 'blockquote', 'a', 'table', 'thead', 'tbody', 'tr',
                              'th', 'td', 'hr', 'span', 'div'],
                ALLOWED_ATTR: ['href', 'target', 'rel', 'class'],
                ALLOW_DATA_ATTR: false
            });
        }

        // If DOMPurify not available, log warning and return html
        console.warn('⚠️ DOMPurify not loaded - HTML content may not be safe');
        return html;
    } catch (error) {
        console.error('Error rendering markdown:', error);
        // Return text with basic HTML escaping as fallback
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }
}

// Add user message to chat with animation
function addMessage(message, isBot = false, category = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isBot) {
        const icon = '<span class="bot-icon">🧼</span>';
        let categoryBadge = '';
        if (category) {
            categoryBadge = `<span class="topic-badge ${category}">${getCategoryLabel(category)}</span>`;
        }
        // Render markdown for bot messages
        const renderedMessage = renderMarkdown(message);
        contentDiv.innerHTML = `${icon}<div class="bot-text markdown-content">${categoryBadge}${renderedMessage}</div>`;
    } else {
        // User messages don't need markdown
        contentDiv.textContent = message;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Trigger fade-in animation
    setTimeout(() => {
        messageDiv.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    }, 10);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Get category label for badge
function getCategoryLabel(category) {
    const labels = {
        'saponification': 'Chemistry',
        'cold_process': 'Technique',
        'first_batch': 'Getting Started',
        'oils': 'Ingredients',
        'essential_oils': 'Fragrance',
        'lye': 'Safety',
        'troubleshooting': 'Help',
        'recipe': 'Recipe',
        'curing': 'Process',
        'colorants': 'Design'
    };
    return labels[category] || 'Info';
}

// Show typing indicator
function showTypingIndicator(message = 'AI is thinking') {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <span class="bot-icon">🧼</span>
        <div class="typing-container">
            <span class="typing-text" id="typingText">${message}</span>
            <div class="typing-dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        </div>
    `;

    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update typing indicator message
function updateTypingIndicator(message) {
    const typingText = document.getElementById('typingText');
    if (typingText) {
        typingText.textContent = message;
    }
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Calculate recipe using SoapCalculator class
function calculateRecipeWithCalculator(oils, superfatPercent = 5) {
    if (!soapCalculator) {
        throw new Error('SoapCalculator not initialized');
    }

    // Create new calculator instance for this recipe
    const calc = new SoapCalculator();
    calc.setSuperfat(superfatPercent);

    // Add each oil to the calculator
    for (const oil of oils) {
        try {
            calc.addOil(oil.name, oil.grams, 'grams');
        } catch (error) {
            throw new Error(`Failed to add ${oil.name}: ${error.message}`);
        }
    }

    // Calculate complete recipe with properties
    return calc.calculate();
}

// Find oil in database (uses SoapCalculator's database)
function findOilInDatabase(oilName) {
    if (!soapCalculator) {
        return null;
    }

    const oilData = soapCalculator.findOil(oilName);
    return oilData;
}

// Find matching response based on keywords
function findResponse(userInput) {
    const input = userInput.toLowerCase();

    // Check for save recipe commands
    if (input.includes('save') && (input.includes('recipe') || input.includes('this'))) {
        if (!lastCalculatedRecipe) {
            return {
                response: "I don't have a recipe to save right now. Please calculate a recipe first by saying something like 'create a beginner soap recipe for 500g batch', then I can help you save it!",
                category: 'recipe'
            };
        }
        return {
            response: "Great! To save this recipe, click the **💾 Save Recipe** button that appears below the recipe. This will open a form where you can:\n\n1. Give your recipe a memorable name (e.g., 'Lavender Dream Soap')\n2. Add optional notes about scent, color, or how it turned out\n3. Click 'Save Recipe' to store it\n\nYour saved recipes will be stored in your browser and can be accessed anytime by clicking the **📚 My Recipes** button at the top of the chat!",
            category: 'recipe'
        };
    }

    // Check for view/open recipes commands
    if ((input.includes('view') || input.includes('show') || input.includes('open') || input.includes('see')) &&
        (input.includes('recipe') || input.includes('saved') || input.includes('my recipes'))) {
        return {
            response: "To view your saved recipes, click the **📚 My Recipes** button at the top of the chat! From there you can:\n\n- Browse all your saved recipes\n- Search recipes by name or notes\n- Load a recipe to view it again\n- Delete recipes you no longer need\n\nYou can save up to 50 recipes, and they're stored locally in your browser.",
            category: 'recipe'
        };
    }

    // Check for recipe calculation keywords or recommendation requests
    if (input.includes('calculate') || input.includes('recipe calculator') ||
        input.includes('make recipe') || input.includes('create recipe') ||
        input.includes('help me calculate') || input.includes('recommend') ||
        input.includes('suggestion') || input.includes('your best')) {

        // If user is asking for a recommendation and already specified batch size
        const gramsMatch = input.match(/(\d+)\s*(grams?|g\b)/i);
        if ((input.includes('recommend') || input.includes('your best') || input.includes('suggestion')) && gramsMatch) {
            const batchSize = parseInt(gramsMatch[1]);

            // Provide a beginner-friendly recipe recommendation
            return {
                response: `Perfect! For your ${batchSize}g batch, I recommend this balanced recipe:\n\n` +
                    `**Oils & Fats:**\n` +
                    `- Olive Oil: ${Math.round(batchSize * 0.35)}g (35%) - conditioning\n` +
                    `- Coconut Oil: ${Math.round(batchSize * 0.30)}g (30%) - hardness & lather\n` +
                    `- Shea Butter: ${Math.round(batchSize * 0.25)}g (25%) - hardness & conditioning\n` +
                    `- Castor Oil: ${Math.round(batchSize * 0.10)}g (10%) - lather boost\n\n` +
                    `Would you like me to calculate the exact lye and water amounts? Type "calculate" or "yes" to proceed with 5% superfat!`,
                category: 'recipe'
            };
        }

        recipeState.active = true;
        recipeState.oils = [];
        recipeState.batchSizeGrams = 0;
        return {
            response: "Great! I'll help you calculate a safe soap recipe. First, how many grams of soap do you want to make? (For example: 500 grams, 1000 grams, etc.)",
            category: 'recipe'
        };
    }

    // Handle recipe building conversation
    if (recipeState.active) {
        // Check for batch size
        const gramsMatch = input.match(/(\d+)\s*(grams?|g\b)/i);
        if (gramsMatch && recipeState.batchSizeGrams === 0) {
            recipeState.batchSizeGrams = parseInt(gramsMatch[1]);

            // Get available oils from SoapCalculator database
            const availableOilsList = soapCalculator
                ? soapCalculator.getAvailableOils().slice(0, 10).map(oil => oil.name).join(', ')
                : 'olive oil, coconut oil, palm oil, castor oil, sweet almond oil, shea butter';

            return {
                response: `Perfect! You want to make ${recipeState.batchSizeGrams}g of soap. Now, tell me what oils you have and how many grams of each. For example: "300g olive oil, 150g coconut oil, 50g castor oil". Available oils include: ${availableOilsList}, and more. What oils do you have?`,
                category: 'recipe'
            };
        }

        // Check for oil inputs
        if (recipeState.batchSizeGrams > 0 && recipeState.oils.length === 0) {
            const oilMatches = input.match(/(\d+)\s*g?\s*([\w\s]+?)(?:,|and|$)/gi);
            if (oilMatches) {
                let totalGrams = 0;
                let unknownOils = [];

                for (const match of oilMatches) {
                    const parts = match.match(/(\d+)\s*g?\s*([\w\s]+)/i);
                    if (parts) {
                        const grams = parseInt(parts[1]);
                        const oilName = parts[2].trim().replace(/,/g, '');
                        const oilData = findOilInDatabase(oilName);

                        if (oilData) {
                            recipeState.oils.push({
                                name: oilData.name,
                                grams: grams
                            });
                            totalGrams += grams;
                        } else {
                            unknownOils.push(oilName);
                        }
                    }
                }

                if (unknownOils.length > 0) {
                    const availableOils = soapCalculator
                        ? soapCalculator.getAvailableOils().map(oil => oil.name).join(', ')
                        : 'olive oil, coconut oil, palm oil, etc.';

                    return {
                        response: `I don't have data for: ${unknownOils.join(', ')}. Available oils include: ${availableOils}. Please try again with oils from this list.`,
                        category: 'recipe'
                    };
                }

                if (totalGrams !== recipeState.batchSizeGrams) {
                    return {
                        response: `Note: Your oils total ${totalGrams}g, but you wanted ${recipeState.batchSizeGrams}g. That's okay! I'll calculate based on ${totalGrams}g. Would you like to change the superfat percentage? (Default is 5%, recommended range is 5-8%. Type "calculate" to proceed with 5% superfat, or specify like "7% superfat")`,
                        category: 'recipe'
                    };
                }

                return {
                    response: `Perfect! Your oils total ${totalGrams}g. Would you like to change the superfat percentage? (Default is 5%, recommended range is 5-8%. Type "calculate" to proceed with 5% superfat, or specify like "7% superfat")`,
                    category: 'recipe'
                };
            }
        }

        // Check for superfat adjustment or final calculation
        if (recipeState.oils.length > 0) {
            const superfatMatch = input.match(/(\d+)%?\s*superfat/i);
            if (superfatMatch) {
                recipeState.superfat = parseInt(superfatMatch[1]);
            }

            if (input.includes('calculate') || input.includes('yes') || superfatMatch) {
                try {
                    // Use the proper SoapCalculator class for accurate calculation
                    const recipe = calculateRecipeWithCalculator(
                        recipeState.oils,
                        recipeState.superfat
                    );

                    // Validate recipe for safety
                    let validationMessage = '';
                    if (recipeValidator) {
                        const validation = recipeValidator.validateRecipe(recipe);

                        if (!validation.valid) {
                            validationMessage = recipeValidator.formatValidationMessage(validation);
                            validationMessage += '\n---\n\n';
                        } else if (validation.warnings.length > 0) {
                            validationMessage = recipeValidator.formatValidationMessage(validation);
                            validationMessage += '\n---\n\n';
                        }
                    }

                    // Format the complete recipe with properties
                    const recipeOutput = formatCalculatedRecipe(recipe);
                    const response = validationMessage + recipeOutput;

                    // Reset state
                    recipeState = {
                        active: false,
                        batchSizeGrams: 0,
                        oils: [],
                        superfat: 5
                    };

                    return { response: response, category: 'recipe' };
                } catch (error) {
                    console.error('Recipe calculation error:', error);
                    return {
                        response: `Sorry, I encountered an error calculating your recipe: ${error.message}. Please try again or ask for help.`,
                        category: 'recipe'
                    };
                }
            }
        }
    }

    // Check each knowledge category
    for (const [category, data] of Object.entries(soapKnowledge)) {
        if (data.keywords.some(keyword => input.includes(keyword.toLowerCase()))) {
            return { response: data.response, category: category };
        }
    }

    // Default response if no match found
    return {
        response: "That's a great question about soap making! I can help you with:\n• Calculating a custom soap recipe (say 'calculate recipe')\n• Saponification process\n• Oils and ingredients\n• Cold/hot process techniques\n• Troubleshooting\n• Essential oils and colorants\n\nTry asking: 'Calculate a recipe' or 'What is saponification?'",
        category: null
    };
}

// Rate-limited API request function with exponential backoff retry
async function makeRateLimitedRequest(userMessage, retryCount = 0) {
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;

    // Calculate wait time with exponential backoff if there were recent rate limit errors
    let baseWaitTime = MIN_REQUEST_INTERVAL_MS;
    if (consecutiveRateLimitErrors > 0) {
        // Exponential backoff: 5s, 10s, 20s, 40s...
        baseWaitTime = MIN_REQUEST_INTERVAL_MS * Math.pow(2, consecutiveRateLimitErrors);
    }

    // If not enough time has passed since last request, wait
    if (timeSinceLastRequest < baseWaitTime) {
        const waitTime = baseWaitTime - timeSinceLastRequest;
        const waitSeconds = Math.round(waitTime/1000);
        console.log(`⏱️ Rate limiting: waiting ${waitSeconds}s before next request`);

        // Update typing indicator to show wait time
        updateTypingIndicator(`Rate limiting... waiting ${waitSeconds}s`);

        await new Promise(resolve => setTimeout(resolve, waitTime));

        // Reset typing indicator
        updateTypingIndicator('AI is thinking');
    }

    // Update last request time
    lastRequestTime = Date.now();

    try {
        // Make the actual API call
        const result = await getLLMResponse(userMessage);

        // Success! Reset consecutive error counter
        consecutiveRateLimitErrors = 0;

        return result;
    } catch (error) {
        // Check if it's a retryable error
        const isRateLimitError = error.message.includes('429') || error.message.includes('rate limit') || error.message.includes('Rate Limit');
        const isTimeoutError = error.message.includes('timeout') || error.message.includes('Request timeout');
        const isNetworkError = error.message.includes('Failed to fetch') || error.message.includes('NetworkError');

        // Retry on rate limit, timeout, or network errors (but not on other errors like auth failures)
        const shouldRetry = (isRateLimitError || isTimeoutError || isNetworkError) && retryCount < MAX_RETRIES;

        if (shouldRetry) {
            if (isRateLimitError) {
                consecutiveRateLimitErrors++;
            }

            const retryDelay = 10000 * Math.pow(2, retryCount); // 10s, 20s, 40s
            const retrySeconds = Math.round(retryDelay/1000);

            let retryReason = 'Error';
            if (isRateLimitError) retryReason = 'Rate limit hit';
            else if (isTimeoutError) retryReason = 'Request timeout';
            else if (isNetworkError) retryReason = 'Network error';

            console.log(`⚠️ ${retryReason}. Retry ${retryCount + 1}/${MAX_RETRIES} after ${retrySeconds}s...`);

            // Update typing indicator to show retry status
            updateTypingIndicator(`${retryReason}... retrying in ${retrySeconds}s (${retryCount + 1}/${MAX_RETRIES})`);

            // Wait with exponential backoff
            await new Promise(resolve => setTimeout(resolve, retryDelay));

            // Reset typing indicator
            updateTypingIndicator('AI is thinking');

            // Retry the request
            return makeRateLimitedRequest(userMessage, retryCount + 1);
        }

        // Not a retryable error or max retries reached - throw the error
        throw error;
    }
}

// Track if backend has been used in this session
let backendHasBeenUsed = false;

// Handle sending messages
async function sendMessage() {
    const userMessage = chatInput.value.trim();

    if (userMessage === '') return;

    // Add user message
    addMessage(userMessage, false);

    // Clear input
    chatInput.value = '';

    // Disable input while processing
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Show typing indicator
    showTypingIndicator();

    let aiResult;
    let messageDisplayed = false;

    try {
        // Wake up backend on first use (for Render free tier)
        if (!backendHasBeenUsed && aiConfig) {
            updateTypingIndicator('Waking up AI server...');
            const isAwake = await aiConfig.wakeUpBackend();
            if (isAwake) {
                backendHasBeenUsed = true;
                updateTypingIndicator('AI is thinking');
            } else {
                console.warn('Backend wake-up failed, proceeding anyway...');
            }
        }

        // Use rate-limited request to prevent API throttling
        aiResult = await makeRateLimitedRequest(userMessage);

        removeTypingIndicator();

        // Add response with provider badge (markdown will be rendered in addMessage)
        addMessage(aiResult.response, true);
        messageDisplayed = true;

        // Update conversation history (keep last 10 messages for context)
        conversationHistory.push({ role: 'user', content: userMessage });
        conversationHistory.push({ role: 'assistant', content: aiResult.response });

        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }
    } catch (error) {
        console.error('Error in sendMessage:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack,
            messageDisplayed: messageDisplayed
        });
        removeTypingIndicator();

        // If the message was already displayed successfully, don't show error to user
        if (messageDisplayed) {
            console.error('Error occurred after message was displayed - not showing error to user');
            return; // Exit early - message was already shown successfully
        }

        // Provide specific error messages based on error type
        let errorMessage = '';
        let useFallback = true;

        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMessage = '**Connection Error**: Unable to reach the AI server. Using local knowledge base instead.';
        } else if (error.message.includes('429') || error.message.includes('rate limit') || error.message.includes('Rate Limit')) {
            errorMessage = '**Rate Limit Exceeded**: The AI service has hit its request limit (this is normal for free-tier backend). The chat automatically retried but still hit limits. Please wait 30-60 seconds before trying again. Tip: Use the "Start Over" button to reset rate limiting if issues persist.';
            useFallback = true; // Fall back to local knowledge so user still gets an answer
        } else if (error.message.includes('timeout')) {
            errorMessage = '**Request Timeout**: The AI took too long to respond. Please try asking again with a simpler question.';
        } else if (error.message.includes('401') || error.message.includes('403')) {
            errorMessage = '**Authentication Error**: API key issue. Please contact support.';
            useFallback = true;
        } else {
            errorMessage = '**Unexpected Error**: The AI backend encountered an issue. Using local calculator instead...';
            console.error('Full error details:', error);
        }

        if (useFallback) {
            // Fallback to local knowledge base
            const result = findResponse(userMessage);

            // Build a helpful fallback message
            let fallbackMessage = errorMessage;

            // If this looks like a recipe-related question, give more context
            if (result.category === 'recipe') {
                fallbackMessage += '\n\n' + result.response;
            } else if (recipeState.active) {
                // If we're in the middle of a recipe conversation, explain what happened
                fallbackMessage += '\n\n**Recipe conversation interrupted.** The local calculator can still help! Please re-state your full request. For example: "Give me a honey and oat soap recipe for 500g with olive oil, coconut oil, shea butter, and castor oil."';
            } else {
                fallbackMessage += '\n\n' + result.response;
            }

            addMessage(fallbackMessage, true, result.category);
        } else {
            addMessage(errorMessage, true);
        }
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

// Keyboard shortcuts
chatInput.addEventListener('keydown', (e) => {
    // Enter to send (without shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
    // Escape to clear input
    else if (e.key === 'Escape') {
        chatInput.value = '';
        chatInput.blur();
    }
    // Ctrl/Cmd + K to start over
    else if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        startOver();
    }
});

// Handle suggestion buttons
suggestionButtons.forEach(button => {
    button.addEventListener('click', () => {
        const question = button.getAttribute('data-question');
        chatInput.value = question;
        sendMessage();
    });
});

// Generate unique recipe ID
function generateRecipeId() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Removed confusing chars (I, O, 0, 1)
    let id = 'SAP-';
    for (let i = 0; i < 4; i++) {
        id += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return id;
}

// Format timestamp as "X ago"
function formatTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);

    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    const days = Math.floor(hours / 24);
    return `${days} day${days === 1 ? '' : 's'} ago`;
}

// Format recipe for clipboard (plain text)
function formatRecipeForClipboard(result, recipeId, timestamp) {
    let output = `🧼 SOAP RECIPE - ${recipeId}\n`;
    output += `Calculated: ${new Date(timestamp).toLocaleString()}\n`;
    output += `Generated by Saponify AI (https://victoriarg.com/saponifyai.html)\n\n`;

    output += `OILS & FATS (${result.totalBatchSize.grams}g total):\n`;
    result.oils.forEach(oil => {
        output += `  • ${oil.name}: ${oil.grams}g (${oil.ounces}oz) - ${oil.percent}%\n`;
    });

    output += `\nLYE SOLUTION:\n`;
    output += `  • ${result.lye.type}: ${result.lye.grams}g (${result.lye.ounces}oz)\n`;
    output += `  • Water: ${result.water.grams}g (${result.water.ounces}oz)\n`;
    output += `  • Superfat: ${result.superfat}%\n`;

    if (result.fragrance) {
        output += `\nFRAGRANCE/ESSENTIAL OILS:\n`;
        output += `  • ${result.fragrance.grams}g (${result.fragrance.ounces}oz)\n`;
    }

    output += `\nSOAP PROPERTIES:\n`;
    const props = result.properties;
    output += `  • Hardness: ${props.hardness.value} (${props.hardness.range})\n`;
    output += `  • Cleansing: ${props.cleansing.value} (${props.cleansing.range})\n`;
    output += `  • Conditioning: ${props.conditioning.value} (${props.conditioning.range})\n`;
    output += `  • Bubbly Lather: ${props.bubbly.value} (${props.bubbly.range})\n`;
    output += `  • Creamy Lather: ${props.creamy.value} (${props.creamy.range})\n`;
    output += `  • Iodine: ${props.iodine.value} (${props.iodine.range})\n`;
    output += `  • INS: ${props.ins.value} (${props.ins.range})\n`;

    output += `\n⚠️ SAFETY REMINDERS:\n`;
    output += `  • Always add lye to water (never water to lye)\n`;
    output += `  • Wear safety goggles and gloves at all times\n`;
    output += `  • Work in a well-ventilated area\n`;
    output += `  • Double-check all measurements before mixing\n`;
    output += `  • Cure for 4-6 weeks before use\n`;

    return output;
}

// Copy recipe to clipboard
async function copyRecipeToClipboard() {
    if (!lastCalculatedRecipe) {
        alert('No recipe to copy. Please calculate a recipe first.');
        return;
    }

    const recipeText = formatRecipeForClipboard(
        lastCalculatedRecipe.recipe,
        lastCalculatedRecipe.id,
        lastCalculatedRecipe.timestamp
    );

    try {
        await navigator.clipboard.writeText(recipeText);
        showCopyFeedback('Recipe copied to clipboard! ✓');
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = recipeText;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showCopyFeedback('Recipe copied to clipboard! ✓');
        } catch (err2) {
            showCopyFeedback('Failed to copy recipe. Please select and copy manually.');
        }
        document.body.removeChild(textArea);
    }
}

// Show copy feedback
function showCopyFeedback(message) {
    const feedback = document.createElement('div');
    feedback.className = 'copy-feedback';
    feedback.textContent = message;
    document.body.appendChild(feedback);

    setTimeout(() => {
        feedback.classList.add('show');
    }, 10);

    setTimeout(() => {
        feedback.classList.remove('show');
        setTimeout(() => feedback.remove(), 300);
    }, 2000);
}

// Print recipe
function printRecipe() {
    if (!lastCalculatedRecipe) {
        alert('No recipe to print. Please calculate a recipe first.');
        return;
    }

    const printWindow = window.open('', '', 'width=800,height=600');
    const recipe = lastCalculatedRecipe.recipe;
    const props = recipe.properties;

    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Soap Recipe - ${lastCalculatedRecipe.id}</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
                h1 { color: #3d2e1f; border-bottom: 3px solid #7fa563; padding-bottom: 10px; }
                h2 { color: #6b5d52; margin-top: 30px; }
                .header { display: flex; justify-content: space-between; margin-bottom: 20px; }
                .recipe-id { color: #7fa563; font-weight: bold; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
                th { background-color: #f5ede3; }
                .warning { background-color: #fff3cd; border-left: 4px solid #ff9800; padding: 15px; margin: 20px 0; }
                ul { line-height: 1.8; }
                @media print { .no-print { display: none; } }
            </style>
        </head>
        <body>
            <div class="header">
                <div>
                    <h1>🧼 Soap Recipe</h1>
                    <p class="recipe-id">Recipe ID: ${lastCalculatedRecipe.id}</p>
                    <p>Calculated: ${new Date(lastCalculatedRecipe.timestamp).toLocaleString()}</p>
                </div>
            </div>

            <h2>Oils & Fats</h2>
            <table>
                <tr><th>Oil/Butter</th><th>Grams</th><th>Ounces</th><th>Percentage</th></tr>
                ${recipe.oils.map(oil => `
                    <tr>
                        <td>${oil.name}</td>
                        <td>${oil.grams}g</td>
                        <td>${oil.ounces}oz</td>
                        <td>${oil.percent}%</td>
                    </tr>
                `).join('')}
                <tr style="font-weight: bold;">
                    <td>Total</td>
                    <td>${recipe.totalBatchSize.grams}g</td>
                    <td>${recipe.totalBatchSize.ounces}oz</td>
                    <td>100%</td>
                </tr>
            </table>

            <h2>Lye Solution</h2>
            <ul>
                <li><strong>${recipe.lye.type}:</strong> ${recipe.lye.grams}g (${recipe.lye.ounces}oz)</li>
                <li><strong>Water:</strong> ${recipe.water.grams}g (${recipe.water.ounces}oz)</li>
                <li><strong>Superfat:</strong> ${recipe.superfat}%</li>
            </ul>

            <h2>Soap Properties</h2>
            <table>
                <tr><th>Property</th><th>Value</th><th>Typical Range</th></tr>
                <tr><td>Hardness</td><td>${props.hardness.value}</td><td>${props.hardness.range}</td></tr>
                <tr><td>Cleansing</td><td>${props.cleansing.value}</td><td>${props.cleansing.range}</td></tr>
                <tr><td>Conditioning</td><td>${props.conditioning.value}</td><td>${props.conditioning.range}</td></tr>
                <tr><td>Bubbly Lather</td><td>${props.bubbly.value}</td><td>${props.bubbly.range}</td></tr>
                <tr><td>Creamy Lather</td><td>${props.creamy.value}</td><td>${props.creamy.range}</td></tr>
                <tr><td>Iodine</td><td>${props.iodine.value}</td><td>${props.iodine.range}</td></tr>
                <tr><td>INS</td><td>${props.ins.value}</td><td>${props.ins.range}</td></tr>
            </table>

            <div class="warning">
                <h3>⚠️ SAFETY REMINDERS</h3>
                <ul>
                    <li><strong>Always add lye to water</strong>, never water to lye (risk of explosive boiling)</li>
                    <li>Wear <strong>safety goggles and gloves</strong> at all times</li>
                    <li>Work in a <strong>well-ventilated area</strong></li>
                    <li><strong>Double-check all measurements</strong> before mixing</li>
                    <li>Cure for <strong>4-6 weeks</strong> before use</li>
                    <li>Keep lye and raw soap away from children and pets</li>
                </ul>
            </div>

            <p style="text-align: center; color: #888; margin-top: 40px;">
                Generated by Saponify AI - https://victoriarg.com/saponifyai.html
            </p>
        </body>
        </html>
    `);

    printWindow.document.close();
    printWindow.focus();

    setTimeout(() => {
        printWindow.print();
    }, 250);
}

// Scale recipe by multiplier
function scaleRecipe(multiplier) {
    if (!lastCalculatedRecipe) {
        alert('No recipe to scale. Please calculate a recipe first.');
        return;
    }

    // Get original recipe
    const originalRecipe = lastCalculatedRecipe.recipe;

    // Scale all oil amounts
    const scaledOils = originalRecipe.oils.map(oil => ({
        name: oil.name,
        grams: Math.round(oil.grams * multiplier * 10) / 10,
        percent: oil.percent // Percentages stay the same
    }));

    // Recalculate using SoapCalculator with scaled amounts
    const calc = new SoapCalculator();
    calc.setSuperfat(originalRecipe.superfat);

    // Set water settings if available
    if (originalRecipe.water.concentration) {
        calc.setLyeConcentration(originalRecipe.water.concentration);
    } else if (originalRecipe.water.ratio) {
        calc.setWaterRatio(originalRecipe.water.ratio);
    }

    // Add scaled oils
    scaledOils.forEach(oil => {
        calc.addOil(oil.name, oil.grams, 'grams');
    });

    // Calculate new recipe
    const scaledRecipe = calc.calculate();

    // Validate the scaled recipe
    let validationMessage = '';
    if (recipeValidator) {
        const validation = recipeValidator.validateRecipe(scaledRecipe);

        if (!validation.valid) {
            validationMessage = recipeValidator.formatValidationMessage(validation);
            validationMessage += '\n---\n\n';
        } else if (validation.warnings.length > 0) {
            validationMessage = recipeValidator.formatValidationMessage(validation);
            validationMessage += '\n---\n\n';
        }
    }

    // Format and display the scaled recipe
    const scaledOutput = formatCalculatedRecipe(scaledRecipe);
    const scaleInfo = `\n**📐 Scaled Recipe** (×${multiplier} from original)\n\n`;
    const finalOutput = scaleInfo + validationMessage + scaledOutput;

    // Add as bot message
    addMessage(finalOutput, true);

    // Scroll to show new recipe
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Show feedback
    showCopyFeedback(`Recipe scaled to ${Math.round(scaledRecipe.totalBatchSize.grams)}g! ✓`);
}

// Start over - reset conversation
function startOver() {
    if (confirm('Are you sure you want to start a new conversation? This will clear the chat history.')) {
        // Clear conversation history
        conversationHistory = [];

        // Reset recipe state
        recipeState = {
            active: false,
            batchSizeGrams: 0,
            oils: [],
            superfat: 5
        };

        // Clear last recipe
        lastCalculatedRecipe = null;
        lastCalculatedRecipeTime = null;

        // Reset rate limiting state to allow fresh requests
        lastRequestTime = 0;
        consecutiveRateLimitErrors = 0;

        // Clear chat messages (except initial bot message)
        const chatMessages = document.getElementById('chatMessages');
        const firstMessage = chatMessages.querySelector('.bot-message');
        chatMessages.innerHTML = '';
        if (firstMessage) {
            chatMessages.appendChild(firstMessage.cloneNode(true));
        }

        showCopyFeedback('Conversation reset! Rate limits cleared. ✓');
    }
}

// Format recipe results from SoapCalculator for display
function formatCalculatedRecipe(result) {
    // Store recipe for quick wins
    const recipeId = generateRecipeId();
    const timestamp = Date.now();
    lastCalculatedRecipe = {
        recipe: result,
        id: recipeId,
        timestamp: timestamp
    };
    lastCalculatedRecipeTime = timestamp;

    let output = `## 🧼 Your Custom Soap Recipe\n\n`;
    output += `**Recipe ID:** \`${recipeId}\` | **Calculated:** ${formatTimeAgo(timestamp)}\n\n`;

    // Oils section
    output += `### Oils & Fats (${result.totalBatchSize.grams}g total)\n`;
    result.oils.forEach(oil => {
        output += `- **${oil.name}**: \`${oil.grams}g\` (\`${oil.ounces}oz\`) - ${oil.percent}%\n`;
    });

    // Lye section
    output += `\n### Lye Solution\n`;
    output += `- **${result.lye.type}**: \`${result.lye.grams}g\` (\`${result.lye.ounces}oz\`)\n`;
    output += `- **Water**: \`${result.water.grams}g\` (\`${result.water.ounces}oz\`)`;
    if (result.water.concentration) {
        output += ` (${result.water.concentration}% lye concentration)`;
    } else if (result.water.ratio) {
        output += ` (${result.water.ratio}:1 water:lye ratio)`;
    }
    output += `\n- **Superfat**: ${result.superfat}%\n`;

    // Fragrance if applicable
    if (result.fragrance) {
        output += `\n### Fragrance/Essential Oils\n`;
        output += `- \`${result.fragrance.grams}g\` (\`${result.fragrance.ounces}oz\`)\n`;
    }

    // Soap properties
    output += `\n### Soap Properties\n`;
    const props = result.properties;
    output += `| Property | Value | Typical Range | Status |\n`;
    output += `|----------|-------|---------------|--------|\n`;
    output += `| **Hardness** | ${props.hardness.value} | ${props.hardness.range} | ${getStatusEmoji(props.hardness.status)} |\n`;
    output += `| **Cleansing** | ${props.cleansing.value} | ${props.cleansing.range} | ${getStatusEmoji(props.cleansing.status)} |\n`;
    output += `| **Conditioning** | ${props.conditioning.value} | ${props.conditioning.range} | ${getStatusEmoji(props.conditioning.status)} |\n`;
    output += `| **Bubbly Lather** | ${props.bubbly.value} | ${props.bubbly.range} | ${getStatusEmoji(props.bubbly.status)} |\n`;
    output += `| **Creamy Lather** | ${props.creamy.value} | ${props.creamy.range} | ${getStatusEmoji(props.creamy.status)} |\n`;
    output += `| **Iodine** | ${props.iodine.value} | ${props.iodine.range} | ${getStatusEmoji(props.iodine.status)} |\n`;
    output += `| **INS** | ${props.ins.value} | ${props.ins.range} | ${getStatusEmoji(props.ins.status)} |\n`;

    // Quick action buttons
    output += `\n<div class="recipe-actions">\n`;
    output += `<button onclick="copyRecipeToClipboard()" class="recipe-action-btn">📋 Copy Recipe</button>\n`;
    output += `<button onclick="openSaveRecipeModal()" class="recipe-action-btn">💾 Save Recipe</button>\n`;
    output += `<button onclick="printRecipe()" class="recipe-action-btn">🖨️ Print Recipe</button>\n`;
    output += `</div>\n\n`;

    // Scaling buttons
    output += `<div class="recipe-scaling">\n`;
    output += `<span class="scaling-label">Scale Recipe:</span>\n`;
    output += `<button onclick="scaleRecipe(0.5)" class="scaling-btn" title="Half the recipe size">×0.5 (Half)</button>\n`;
    output += `<button onclick="scaleRecipe(1.5)" class="scaling-btn" title="Make 50% more">×1.5</button>\n`;
    output += `<button onclick="scaleRecipe(2)" class="scaling-btn" title="Double the recipe size">×2 (Double)</button>\n`;
    output += `<button onclick="scaleRecipe(3)" class="scaling-btn" title="Triple the recipe size">×3 (Triple)</button>\n`;
    output += `</div>\n\n`;

    // Safety warnings
    output += `\n> ### ⚠️ SAFETY REMINDERS\n`;
    output += `> - **Always add lye to water**, never water to lye (risk of explosive boiling)\n`;
    output += `> - Wear **safety goggles and gloves** at all times\n`;
    output += `> - Work in a **well-ventilated area**\n`;
    output += `> - **Double-check all measurements** before mixing\n`;
    output += `> - Cure for **4-6 weeks** before use\n`;
    output += `> - Keep lye and raw soap away from children and pets\n`;

    return output;
}

// Get status emoji for property ranges
function getStatusEmoji(status) {
    switch(status) {
        case 'good': return '✅ Good';
        case 'low': return '⬇️ Low';
        case 'high': return '⬆️ High';
        default: return '—';
    }
}

// Initialize AI configuration on page load
document.addEventListener('DOMContentLoaded', async () => {
    aiConfig = new AIConfig();

    // Check if markdown libraries are loaded
    console.log('📚 Library status:');
    console.log('  marked:', typeof marked !== 'undefined' ? '✅ Loaded' : '❌ Not loaded');
    console.log('  DOMPurify:', typeof DOMPurify !== 'undefined' ? '✅ Loaded' : '❌ Not loaded');

    // Initialize SoapCalculator if available
    if (typeof SoapCalculator !== 'undefined') {
        soapCalculator = new SoapCalculator();
        console.log('✅ SoapCalculator initialized successfully');
    } else {
        console.warn('⚠️ SoapCalculator not loaded - recipe calculations may be limited');
    }

    // Initialize RecipeValidator if available
    if (typeof RecipeValidator !== 'undefined') {
        recipeValidator = new RecipeValidator();
        console.log('✅ RecipeValidator initialized successfully');
    } else {
        console.warn('⚠️ RecipeValidator not loaded - recipe validation disabled');
    }

    // Check backend health
    const isHealthy = await aiConfig.checkBackendHealth();
    updateAIStatus(isHealthy);
});

// Update AI status badge
function updateAIStatus(isBackendHealthy) {
    const statusBadge = document.getElementById('aiStatusBadge');
    if (statusBadge) {
        if (isBackendHealthy) {
            statusBadge.textContent = '🤖 AI: Gemini (Active)';
            statusBadge.className = 'ai-status-badge active';
        } else {
            statusBadge.textContent = '🤖 AI: Offline (Local Mode)';
            statusBadge.className = 'ai-status-badge';
        }
    }
}
