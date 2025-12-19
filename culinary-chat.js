// Culinary Graph - Recipe & Ingredient Search Chat Assistant
// RAG-Enhanced Architecture: Local knowledge graph data enhances LLM responses

// Initialize AI configuration and conversation history
let culinaryAiConfig;
let conversationHistory = [];
let dietaryRestrictions = []; // User's dietary restrictions and allergies

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const suggestionButtons = document.querySelectorAll('.suggestion-btn');
const restrictionsInput = document.getElementById('restrictionsInput');
const addRestrictionBtn = document.getElementById('addRestrictionBtn');
const restrictionsChips = document.getElementById('restrictionsChips');
const restrictionsBanner = document.getElementById('restrictionsBanner');
const restrictionsToggle = document.getElementById('restrictionsToggle');
const restrictionsCount = document.getElementById('restrictionsCount');

// Cooking-themed loading messages
const cookingLoadingMessages = [
    // Kitchen actions
    'Chopping ingredients...',
    'Whisking up a response...',
    'Simmering ideas...',
    'Marinating on that...',
    'Stirring the pot...',
    'Folding in the details...',
    // Cooking progress
    'Preheating...',
    'Bringing to a boil...',
    'Reducing the sauce...',
    'Letting it rest...',
    'Plating up...',
    // Chef personality
    'Chef is in the kitchen...',
    'Tasting for seasoning...',
    'Adding a pinch of wisdom...',
    'Almost ready to serve...',
    'Gathering fresh ideas...',
];

// Get random cooking loading message
function getRandomLoadingMessage() {
    return cookingLoadingMessages[Math.floor(Math.random() * cookingLoadingMessages.length)];
}

// Configure marked for better markdown rendering
if (typeof marked !== 'undefined') {
    // Check if we're using marked v4+ or older version
    if (typeof marked.setOptions === 'function') {
        marked.setOptions({
            breaks: true, // Convert \n to <br>
            gfm: true, // GitHub Flavored Markdown
            headerIds: false,
            mangle: false,
            sanitize: false // Allow HTML in markdown
        });
    } else if (typeof marked.use === 'function') {
        // For marked v4+
        marked.use({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
    }
}

// Render markdown to HTML with enhanced formatting
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        try {
            // Use marked.parse if available, otherwise fall back to marked()
            const parseFunc = typeof marked.parse === 'function' ? marked.parse : marked;
            return parseFunc(text);
        } catch (error) {
            console.error('Markdown parsing error:', error);
            return fallbackMarkdown(text);
        }
    }
    return fallbackMarkdown(text);
}

// Fallback markdown renderer for simple formatting
function fallbackMarkdown(text) {
    return text
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold and italic
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Code blocks
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Links
        .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>')
        // Line breaks
        .replace(/\n/g, '<br>');
}

// ============================================
// DIETARY RESTRICTIONS MANAGEMENT
// ============================================

// Add a dietary restriction
function addRestriction() {
    const restriction = restrictionsInput.value.trim().toLowerCase();

    if (restriction === '') return;

    // Check if already exists
    if (dietaryRestrictions.includes(restriction)) {
        restrictionsInput.value = '';
        return;
    }

    // Add to array
    dietaryRestrictions.push(restriction);

    // Save to localStorage
    saveDietaryRestrictions();

    // Update UI
    renderRestrictionChips();

    // Clear input
    restrictionsInput.value = '';

    // Show confirmation message in chat
    addMessage(`Added restriction: **${restriction}**. I'll exclude this from all recommendations.`, true);
}

// Remove a dietary restriction
function removeRestriction(restriction) {
    dietaryRestrictions = dietaryRestrictions.filter(r => r !== restriction);
    saveDietaryRestrictions();
    renderRestrictionChips();

    addMessage(`Removed restriction: **${restriction}**. I can now include this in recommendations.`, true);
}

// Render restriction chips and update count
function renderRestrictionChips() {
    // Update count display
    if (restrictionsCount) {
        if (dietaryRestrictions.length > 0) {
            restrictionsCount.textContent = `(${dietaryRestrictions.length})`;
        } else {
            restrictionsCount.textContent = '';
        }
    }

    if (dietaryRestrictions.length === 0) {
        restrictionsChips.innerHTML = '';
        return;
    }

    const chipsHTML = dietaryRestrictions.map(restriction => `
        <div class="restriction-chip">
            <span class="chip-text">${restriction}</span>
            <button class="chip-remove" onclick="removeRestriction('${restriction}')" title="Remove restriction">×</button>
        </div>
    `).join('');

    restrictionsChips.innerHTML = chipsHTML;
}

// Save restrictions to localStorage
function saveDietaryRestrictions() {
    try {
        localStorage.setItem('culinaryRestrictions', JSON.stringify(dietaryRestrictions));
    } catch (error) {
        console.error('Failed to save dietary restrictions:', error);
    }
}

// Load restrictions from localStorage
function loadDietaryRestrictions() {
    try {
        const saved = localStorage.getItem('culinaryRestrictions');
        if (saved) {
            dietaryRestrictions = JSON.parse(saved);
            renderRestrictionChips();
        } else {
            renderRestrictionChips();
        }
    } catch (error) {
        console.error('Failed to load dietary restrictions:', error);
        renderRestrictionChips();
    }
}

// Get restrictions as a formatted string for AI prompts
function getRestrictionsPrompt() {
    if (dietaryRestrictions.length === 0) {
        return '';
    }

    return `\n\n**IMPORTANT DIETARY RESTRICTIONS**: The user has the following allergies/restrictions - DO NOT recommend any ingredients containing: ${dietaryRestrictions.join(', ')}. Always exclude these from all suggestions and warn if a recipe contains them.`;
}

// Make removeRestriction available globally for onclick handlers
window.removeRestriction = removeRestriction;

// Add message to chat with animation
function addMessage(message, isBot = false, category = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isBot) {
        const icon = '<span class="bot-icon">🥗</span>';
        let categoryBadge = '';
        if (category) {
            categoryBadge = `<span class="topic-badge ${category}">${getCategoryLabel(category)}</span>`;
        }
        // Render markdown for bot messages
        const renderedMessage = renderMarkdown(message);
        contentDiv.innerHTML = `${icon}<div class="bot-text markdown-content">${categoryBadge}${renderedMessage}</div>`;
    } else {
        // User messages don't need markdown, but need user-text wrapper for styling
        const escapedMessage = message.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        contentDiv.innerHTML = `<div class="user-text">${escapedMessage}</div>`;
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
        'substitution': 'Substitution',
        'pairing': 'Flavor Pairing',
        'ingredient': 'Ingredient Search',
        'recipe': 'Recipe Idea',
        'dietary': 'Dietary Info',
        'cuisine': 'Cuisine',
        'category': 'Category'
    };
    return labels[category] || 'Info';
}

// Typing indicator message cycling interval
let typingMessageInterval = null;

// Show typing indicator with cooking-themed message
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';

    const initialMessage = getRandomLoadingMessage();
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <span class="bot-icon">🥗</span>
        <div class="typing-container">
            <span class="typing-text" id="typingText">${initialMessage}</span>
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

    // Cycle through messages every 2.5 seconds for longer waits
    typingMessageInterval = setInterval(() => {
        const typingText = document.getElementById('typingText');
        if (typingText) {
            typingText.textContent = getRandomLoadingMessage();
        }
    }, 2500);
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
    // Clear the message cycling interval
    if (typingMessageInterval) {
        clearInterval(typingMessageInterval);
        typingMessageInterval = null;
    }
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Find ingredient by name (fuzzy matching)
function findIngredient(searchTerm) {
    if (typeof culinaryGraphData === 'undefined') return null;

    const normalized = searchTerm.toLowerCase().trim();

    // Exact match first
    let match = culinaryGraphData.nodes.find(node =>
        node.label.toLowerCase() === normalized || node.id === normalized
    );

    if (match) return match;

    // Partial match
    return culinaryGraphData.nodes.find(node =>
        node.label.toLowerCase().includes(normalized) ||
        normalized.includes(node.label.toLowerCase())
    );
}

// ============================================
// QUERY CLASSIFICATION FOR FOCUSED RAG
// ============================================

// Classify user query to optimize context retrieval
function classifyQuery(userMessage) {
    const input = userMessage.toLowerCase();

    // Substitution queries
    if (input.includes('substitute') || input.includes('replace') || input.includes('instead of')) {
        return { type: 'substitution' };
    }

    // Pairing queries
    if (input.includes('pair') || input.includes('goes well') || input.includes('complement')) {
        return { type: 'pairing' };
    }

    // Recipe/combination queries
    if (input.includes('recipe') || input.includes('used with') || input.includes('combine')) {
        return { type: 'recipe' };
    }

    // Dietary filtering
    if (input.includes('vegan')) {
        return { type: 'dietary', dietaryType: 'vegan' };
    }
    if (input.includes('vegetarian')) {
        return { type: 'dietary', dietaryType: 'vegetarian' };
    }
    if (input.includes('gluten-free') || input.includes('gluten free')) {
        return { type: 'dietary', dietaryType: 'gluten-free' };
    }

    // Cuisine filtering
    if (input.includes('italian')) {
        return { type: 'cuisine', cuisineType: 'italian' };
    }
    if (input.includes('asian')) {
        return { type: 'cuisine', cuisineType: 'asian' };
    }
    if (input.includes('mexican')) {
        return { type: 'cuisine', cuisineType: 'mexican' };
    }
    if (input.includes('mediterranean')) {
        return { type: 'cuisine', cuisineType: 'mediterranean' };
    }

    // Category search
    if (input.includes('protein')) {
        return { type: 'category', categoryType: 'protein' };
    }
    if (input.includes('dairy')) {
        return { type: 'category', categoryType: 'dairy' };
    }
    if (input.includes('grain')) {
        return { type: 'category', categoryType: 'grain' };
    }
    if (input.includes('vegetable')) {
        return { type: 'category', categoryType: 'vegetable' };
    }
    if (input.includes('herb')) {
        return { type: 'category', categoryType: 'herb' };
    }
    if (input.includes('spice')) {
        return { type: 'category', categoryType: 'spice' };
    }

    // General query
    return { type: 'general' };
}

// Get substitutions for an ingredient (used by fallback only)
function getSubstitutions(ingredientId) {
    if (typeof culinaryGraphData === 'undefined') return [];

    const edges = culinaryGraphData.edges.filter(edge =>
        edge.source === ingredientId && edge.type === 'substitutes'
    );

    return edges.map(edge => {
        const substitute = culinaryGraphData.nodes.find(node => node.id === edge.target);
        return {
            ingredient: substitute,
            ratio: edge.ratio,
            context: edge.context
        };
    });
}

// Get detailed substitution rules (used by fallback only)
function getDetailedSubstitutions(ingredientId) {
    if (typeof substitutionRules === 'undefined') return [];
    if (typeof culinaryGraphData === 'undefined') return [];

    if (substitutionRules[ingredientId]) {
        return substitutionRules[ingredientId].map(rule => {
            const substitute = culinaryGraphData.nodes.find(node => node.id === rule.substitute);
            return {
                ingredient: substitute,
                ratio: rule.ratio,
                notes: rule.notes,
                confidence: rule.confidence,
                dietary: rule.dietary
            };
        });
    }
    return [];
}

// Get pairings for an ingredient (used by fallback only)
function getPairings(ingredientId) {
    if (typeof culinaryGraphData === 'undefined') return [];

    const edges = culinaryGraphData.edges.filter(edge =>
        (edge.source === ingredientId || edge.target === ingredientId) &&
        edge.type === 'pairs-with'
    );

    return edges.map(edge => {
        const pairedId = edge.source === ingredientId ? edge.target : edge.source;
        const paired = culinaryGraphData.nodes.find(node => node.id === pairedId);
        return {
            ingredient: paired,
            strength: edge.strength
        };
    });
}

// Get recipe combinations (used by fallback only)
function getRecipeCombinations(ingredientId) {
    if (typeof culinaryGraphData === 'undefined') return [];

    const edges = culinaryGraphData.edges.filter(edge =>
        (edge.source === ingredientId || edge.target === ingredientId) &&
        edge.type === 'used-with'
    );

    return edges.map(edge => {
        const combinedId = edge.source === ingredientId ? edge.target : edge.source;
        const combined = culinaryGraphData.nodes.find(node => node.id === combinedId);
        return {
            ingredient: combined,
            context: edge.context
        };
    });
}

// ============================================
// LOCAL FALLBACK RESPONSE (Only used when LLM is unavailable)
// ============================================

// Generate a basic response from local knowledge graph when LLM fails
function getLocalFallbackResponse(userMessage) {
    if (typeof culinaryGraphData === 'undefined') {
        return {
            response: "I'm having trouble accessing the knowledge base. Please try again in a moment.",
            category: null
        };
    }

    const queryType = classifyQuery(userMessage);
    const input = userMessage.toLowerCase();

    // Try to extract an ingredient from the query
    let ingredient = null;
    const patterns = [
        /substitute for ([a-z\s]+)/i,
        /replace ([a-z\s]+)/i,
        /instead of ([a-z\s]+)/i,
        /pair(?:s)? (?:well )?with ([a-z\s]+)/i,
        /goes well with ([a-z\s]+)/i,
        /what (?:goes|pairs) with ([a-z\s]+)/i,
        /recipe(?:s)? with ([a-z\s]+)/i
    ];

    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            ingredient = findIngredient(match[1].replace(/\?/g, '').trim());
            break;
        }
    }

    // If no match from patterns, try direct lookup
    if (!ingredient) {
        ingredient = findIngredient(userMessage);
    }

    // Generate response based on query type and found ingredient
    switch (queryType.type) {
        case 'substitution':
            if (ingredient) {
                const subs = getDetailedSubstitutions(ingredient.id);
                const basicSubs = getSubstitutions(ingredient.id);
                if (subs.length > 0 || basicSubs.length > 0) {
                    let response = `## Substitutes for ${ingredient.label}\n\n`;
                    const allSubs = subs.length > 0 ? subs : basicSubs.map(s => ({ ingredient: s.ingredient, ratio: s.ratio }));
                    allSubs.slice(0, 4).forEach((sub, i) => {
                        if (sub.ingredient) {
                            response += `${i + 1}. **${sub.ingredient.label}** - ${sub.ratio || '1:1'}\n`;
                        }
                    });
                    return { response, category: 'substitution' };
                }
            }
            return {
                response: "I couldn't find substitution data for that ingredient. Try common ingredients like butter, eggs, milk, or flour.",
                category: 'substitution'
            };

        case 'pairing':
            if (ingredient) {
                const pairs = getPairings(ingredient.id);
                if (pairs.length > 0) {
                    let response = `## Pairs well with ${ingredient.label}\n\n`;
                    pairs.slice(0, 5).forEach(pair => {
                        if (pair.ingredient) {
                            response += `- **${pair.ingredient.label}** (${pair.strength || 'good'})\n`;
                        }
                    });
                    return { response, category: 'pairing' };
                }
            }
            return {
                response: "I couldn't find pairing data for that ingredient. Try garlic, tomato, basil, or olive oil.",
                category: 'pairing'
            };

        case 'recipe':
            if (ingredient) {
                const combos = getRecipeCombinations(ingredient.id);
                if (combos.length > 0) {
                    let response = `## Common combinations with ${ingredient.label}\n\n`;
                    combos.slice(0, 5).forEach(combo => {
                        if (combo.ingredient) {
                            response += `- **${combo.ingredient.label}**${combo.context ? ` (${combo.context})` : ''}\n`;
                        }
                    });
                    return { response, category: 'recipe' };
                }
            }
            return {
                response: "I couldn't find recipe combination data for that ingredient.",
                category: 'recipe'
            };

        case 'dietary':
            const dietaryFiltered = culinaryGraphData.nodes.filter(n =>
                n.dietary && n.dietary.includes(queryType.dietaryType)
            );
            if (dietaryFiltered.length > 0) {
                let response = `## ${queryType.dietaryType.charAt(0).toUpperCase() + queryType.dietaryType.slice(1)} Ingredients\n\n`;
                dietaryFiltered.slice(0, 10).forEach(item => {
                    response += `- **${item.label}** (${item.category})\n`;
                });
                if (dietaryFiltered.length > 10) {
                    response += `\n...and ${dietaryFiltered.length - 10} more`;
                }
                return { response, category: 'dietary' };
            }
            return {
                response: `No ${queryType.dietaryType} ingredients found in the database.`,
                category: 'dietary'
            };

        case 'cuisine':
            const cuisineFiltered = culinaryGraphData.nodes.filter(n =>
                n.cuisine && n.cuisine.includes(queryType.cuisineType)
            );
            if (cuisineFiltered.length > 0) {
                let response = `## ${queryType.cuisineType.charAt(0).toUpperCase() + queryType.cuisineType.slice(1)} Cuisine Ingredients\n\n`;
                cuisineFiltered.slice(0, 10).forEach(item => {
                    response += `- **${item.label}** (${item.category})\n`;
                });
                return { response, category: 'cuisine' };
            }
            return {
                response: `No ${queryType.cuisineType} cuisine ingredients found.`,
                category: 'cuisine'
            };

        case 'category':
            const categoryFiltered = culinaryGraphData.nodes.filter(n => n.category === queryType.categoryType);
            if (categoryFiltered.length > 0) {
                let response = `## ${queryType.categoryType.charAt(0).toUpperCase() + queryType.categoryType.slice(1)} Ingredients\n\n`;
                categoryFiltered.slice(0, 12).forEach(item => {
                    response += `- **${item.label}**\n`;
                });
                return { response, category: 'category' };
            }
            return {
                response: `No ${queryType.categoryType} ingredients found.`,
                category: 'category'
            };

        default:
            // General ingredient info
            if (ingredient) {
                let response = `## ${ingredient.label}\n\n`;
                response += `**Category**: ${ingredient.category}\n`;
                response += `**Cuisines**: ${ingredient.cuisine.join(', ')}\n`;
                if (ingredient.dietary && ingredient.dietary.length > 0) {
                    response += `**Dietary**: ${ingredient.dietary.join(', ')}\n`;
                }
                return { response, category: 'ingredient' };
            }

            // Help message
            return {
                response: `I can help you with:\n\n- **Substitutions**: "Substitute for eggs"\n- **Pairings**: "What pairs well with garlic?"\n- **Recipes**: "Recipes with chickpeas"\n- **Dietary**: "Show vegan proteins"\n- **Cuisine**: "Italian ingredients"\n\nTry one of these!`,
                category: null
            };
    }
}

// ============================================
// AI INTEGRATION FUNCTIONS
// ============================================

// Call backend proxy which handles Gemini API
// Returns { text, finishReason } to support auto-continuation
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
    const response = await fetch(
        `${culinaryAiConfig.getBackendUrl()}/api/chat`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: contents,
                generationConfig: {
                    temperature: 0.7,
                    maxOutputTokens: 2048  // Allow complete responses for recipes
                }
            })
        }
    );

    if (!response.ok) {
        let errorData = {};
        try {
            errorData = await response.json();
        } catch (e) {
            // Response body wasn't JSON
        }
        // Include status code in error message for retry logic to detect
        const errorMsg = `Backend API error: ${response.status} - ${errorData.error || response.statusText}`;
        const error = new Error(errorMsg);
        error.status = response.status;
        error.retryAfter = errorData.retryAfter || null;
        throw error;
    }

    const data = await response.json();

    // Handle Gemini 2.5 response format
    if (data.candidates && data.candidates[0]) {
        const candidate = data.candidates[0];
        const finishReason = candidate.finishReason || 'STOP';

        // Try to get text from parts array
        if (candidate.content && candidate.content.parts && candidate.content.parts[0]) {
            return { text: candidate.content.parts[0].text, finishReason };
        }

        // Fallback: check if there's text directly in content
        if (candidate.content && candidate.content.text) {
            return { text: candidate.content.text, finishReason };
        }

        // If no text found, throw error
        throw new Error('No text content in Gemini response');
    }

    throw new Error('Invalid response format from Gemini API');
}

// Sleep utility for retry delays
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Wrapper with exponential backoff for rate limits (429 errors)
async function callGeminiWithRetry(messages, maxRetries = 3) {
    let lastError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await callGemini(messages);
        } catch (error) {
            lastError = error;

            // Check if it's a rate limit error (429)
            if (error.message.includes('429') || error.status === 429) {
                // Use exponential backoff: 2s, 4s, 8s
                const baseDelay = 2000;
                const delay = baseDelay * Math.pow(2, attempt);

                console.log(`Rate limited. Retrying in ${delay/1000}s (attempt ${attempt + 1}/${maxRetries})`);

                // Update typing indicator to show retry status
                updateTypingIndicator(`Rate limited, retrying in ${Math.round(delay/1000)}s...`);

                await sleep(delay);
                continue;
            }

            // Non-rate-limit error, throw immediately
            throw error;
        }
    }

    // All retries exhausted
    throw lastError;
}

// Main LLM call via backend proxy with focused GraphRAG enhancement
// Supports auto-continuation if response hits token limit
async function getLLMResponse(userMessage) {
    // Classify the query for focused context extraction
    const queryType = classifyQuery(userMessage);
    console.log('Query classified as:', queryType.type);

    // Get focused RAG context based on query type
    let ragContext = '';
    if (culinaryAiConfig && typeof culinaryAiConfig.enhanceQueryWithGraphContext === 'function') {
        const ragResult = culinaryAiConfig.enhanceQueryWithGraphContext(userMessage, queryType);
        ragContext = ragResult.ragContext || '';
        if (ragContext) {
            console.log('✨ GraphRAG: Focused context added for query type:', queryType.type);
        }
    }

    // Add dietary restrictions to user message
    const userMessageWithRestrictions = userMessage + getRestrictionsPrompt();

    // Build system prompt with RAG context injected
    const systemPrompt = culinaryAiConfig.getContextualSystemPrompt(ragContext);

    // Build conversation history
    const messages = [
        { role: 'system', content: systemPrompt },
        ...conversationHistory,
        { role: 'user', content: userMessageWithRestrictions }
    ];

    // Call backend proxy with retry logic for rate limits
    let result = await callGeminiWithRetry(messages);
    let fullResponse = result.text;
    let continuationCount = 0;
    const maxContinuations = 3; // Prevent infinite loops

    // Auto-continue if response was truncated due to token limit
    while (result.finishReason === 'MAX_TOKENS' && continuationCount < maxContinuations) {
        continuationCount++;
        console.log(`Response truncated, auto-continuing (${continuationCount}/${maxContinuations})...`);

        // Add the partial response and request continuation
        const continuationMessages = [
            { role: 'system', content: systemPrompt },
            ...conversationHistory,
            { role: 'user', content: userMessageWithRestrictions },
            { role: 'assistant', content: fullResponse },
            { role: 'user', content: 'Please continue from where you left off.' }
        ];

        result = await callGeminiWithRetry(continuationMessages);
        fullResponse += '\n\n' + result.text;
    }

    if (continuationCount > 0) {
        console.log(`Response completed after ${continuationCount} continuation(s)`);
    }

    return { response: fullResponse, provider: 'Gemini 2.5 Flash' };
}

// Handle sending messages - ALWAYS uses LLM first, local fallback only on error
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

    try {
        // ALWAYS try LLM first - RAG context enhances the response
        const result = await getLLMResponse(userMessage);
        removeTypingIndicator();

        // Add response
        addMessage(result.response, true);

        // Update conversation history
        conversationHistory.push({ role: 'user', content: userMessage });
        conversationHistory.push({ role: 'assistant', content: result.response });

        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }
    } catch (error) {
        console.error('LLM response failed, using local fallback:', error);
        removeTypingIndicator();

        // Fallback to local knowledge base ONLY when LLM fails
        const fallbackResult = getLocalFallbackResponse(userMessage);
        const fallbackMessage = `${fallbackResult.response}\n\n> *AI is currently unavailable. This response is from the local knowledge graph.*`;
        addMessage(fallbackMessage, true, fallbackResult.category);

        // Still update conversation history
        conversationHistory.push({ role: 'user', content: userMessage });
        conversationHistory.push({ role: 'assistant', content: fallbackResult.response });

        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
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

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
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

// Handle restriction input
if (addRestrictionBtn) {
    addRestrictionBtn.addEventListener('click', addRestriction);
}

if (restrictionsInput) {
    restrictionsInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addRestriction();
        }
    });
}

// Handle restrictions toggle
if (restrictionsToggle && restrictionsBanner) {
    restrictionsToggle.addEventListener('click', () => {
        restrictionsBanner.classList.toggle('collapsed');
    });
}

// Update AI status badge
function updateAIStatus(isBackendHealthy) {
    const statusBadge = document.getElementById('culinaryAIStatusBadge');
    if (statusBadge) {
        if (isBackendHealthy) {
            statusBadge.textContent = 'AI: Gemini (Active)';
            statusBadge.className = 'ai-status-badge active';
        } else {
            statusBadge.textContent = 'AI: Fallback Mode';
            statusBadge.className = 'ai-status-badge';
        }
    }
}

// Initialize chat on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Culinary Chat Assistant initialized - RAG-Enhanced Architecture');

    // Load dietary restrictions from localStorage
    loadDietaryRestrictions();

    // Check if knowledge graph data is available
    if (typeof culinaryGraphData !== 'undefined') {
        console.log(`Loaded ${culinaryGraphData.nodes.length} ingredients and ${culinaryGraphData.edges.length} relationships`);
    } else {
        console.warn('Warning: culinaryGraphData not loaded yet');
    }

    // Initialize AI configuration
    try {
        if (typeof CulinaryAIConfig !== 'undefined') {
            culinaryAiConfig = new CulinaryAIConfig();

            // Check backend health
            const isHealthy = await culinaryAiConfig.checkBackendHealth();
            updateAIStatus(isHealthy);

            if (isHealthy) {
                console.log('AI backend is healthy - Gemini 2.5 Flash active with RAG enhancement');
            } else {
                console.log('AI backend unavailable - will use local knowledge base as fallback');
            }
        } else {
            console.error('CulinaryAIConfig not loaded - AI features disabled');
            updateAIStatus(false);
        }
    } catch (error) {
        console.error('Error initializing AI:', error);
        updateAIStatus(false);
    }
});
