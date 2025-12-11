// Culinary Graph - Recipe & Ingredient Search Chat Assistant with AI Integration

// Initialize AI configuration and conversation history
let culinaryAiConfig;
let conversationHistory = [];
let useAI = true; // Toggle between AI and local knowledge base

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const suggestionButtons = document.querySelectorAll('.suggestion-btn');

// Configure marked for better markdown rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true, // Convert \n to <br>
        gfm: true, // GitHub Flavored Markdown
        headerIds: false,
        mangle: false
    });
}

// Render markdown to HTML
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    // Fallback: simple replacements if marked is not available
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

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

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <span class="bot-icon">🥗</span>
        <div class="typing-dots">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;

    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Find ingredient by name (fuzzy matching)
function findIngredient(searchTerm) {
    const normalized = searchTerm.toLowerCase().trim();

    // Exact match first
    let match = ingredientNodes.find(node =>
        node.label.toLowerCase() === normalized || node.id === normalized
    );

    if (match) return match;

    // Partial match
    return ingredientNodes.find(node =>
        node.label.toLowerCase().includes(normalized) ||
        normalized.includes(node.label.toLowerCase())
    );
}

// Get substitutions for an ingredient
function getSubstitutions(ingredientId) {
    const edges = ingredientEdges.filter(edge =>
        edge.source === ingredientId && edge.type === 'substitutes'
    );

    return edges.map(edge => {
        const substitute = ingredientNodes.find(node => node.id === edge.target);
        return {
            ingredient: substitute,
            ratio: edge.ratio,
            context: edge.context
        };
    });
}

// Get detailed substitution rules
function getDetailedSubstitutions(ingredientId) {
    if (substitutionRules[ingredientId]) {
        return substitutionRules[ingredientId].map(rule => {
            const substitute = ingredientNodes.find(node => node.id === rule.substitute);
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

// Get pairings for an ingredient
function getPairings(ingredientId) {
    const edges = ingredientEdges.filter(edge =>
        (edge.source === ingredientId || edge.target === ingredientId) &&
        edge.type === 'pairs-with'
    );

    return edges.map(edge => {
        const pairedId = edge.source === ingredientId ? edge.target : edge.source;
        const paired = ingredientNodes.find(node => node.id === pairedId);
        return {
            ingredient: paired,
            strength: edge.strength
        };
    });
}

// Get recipe combinations (used-with relationships)
function getRecipeCombinations(ingredientId) {
    const edges = ingredientEdges.filter(edge =>
        (edge.source === ingredientId || edge.target === ingredientId) &&
        edge.type === 'used-with'
    );

    return edges.map(edge => {
        const combinedId = edge.source === ingredientId ? edge.target : edge.source;
        const combined = ingredientNodes.find(node => node.id === combinedId);
        return {
            ingredient: combined,
            context: edge.context
        };
    });
}

// Filter ingredients by criteria
function filterIngredients(criteria) {
    return ingredientNodes.filter(node => {
        let matches = true;

        if (criteria.category) {
            matches = matches && node.category === criteria.category;
        }

        if (criteria.cuisine) {
            matches = matches && (
                node.cuisine.includes(criteria.cuisine) ||
                node.cuisine.includes('all')
            );
        }

        if (criteria.dietary) {
            matches = matches && node.dietary.includes(criteria.dietary);
        }

        return matches;
    });
}

// Process user query and generate response
function processQuery(userMessage) {
    const input = userMessage.toLowerCase();

    // Substitution queries
    if (input.includes('substitute') || input.includes('replace') || input.includes('instead of')) {
        return handleSubstitutionQuery(userMessage, input);
    }

    // Pairing queries
    if (input.includes('pair') || input.includes('goes well') || input.includes('complement')) {
        return handlePairingQuery(userMessage, input);
    }

    // Recipe/combination queries
    if (input.includes('recipe') || input.includes('used with') || input.includes('combine')) {
        return handleRecipeQuery(userMessage, input);
    }

    // Dietary filtering
    if (input.includes('vegan') || input.includes('vegetarian') || input.includes('gluten-free')) {
        return handleDietaryQuery(userMessage, input);
    }

    // Cuisine filtering
    if (input.includes('italian') || input.includes('asian') || input.includes('mexican') || input.includes('mediterranean')) {
        return handleCuisineQuery(userMessage, input);
    }

    // Category search
    if (input.includes('protein') || input.includes('dairy') || input.includes('grain') ||
        input.includes('vegetable') || input.includes('herb') || input.includes('spice')) {
        return handleCategoryQuery(userMessage, input);
    }

    // General ingredient info
    const ingredient = findIngredient(userMessage);
    if (ingredient) {
        return handleIngredientInfo(ingredient);
    }

    // Default help response
    return {
        response: `I can help you with:\n\n**🔍 Search Ingredients**\n- "Find vegan proteins"\n- "Show me Italian ingredients"\n- "What dairy alternatives are available?"\n\n**🔄 Find Substitutions**\n- "Substitute for eggs"\n- "Replace butter with what?"\n- "What can I use instead of milk?"\n\n**👥 Discover Pairings**\n- "What pairs well with garlic?"\n- "What goes with tomatoes?"\n\n**🍳 Recipe Ideas**\n- "Recipes with chickpeas"\n- "What's used with rice?"\n\nTry asking one of these questions!`,
        category: null
    };
}

// Handle substitution queries
function handleSubstitutionQuery(userMessage, input) {
    // Extract ingredient name
    const patterns = [
        /substitute for ([a-z\s]+)/i,
        /replace ([a-z\s]+)/i,
        /instead of ([a-z\s]+)/i,
        /substitute ([a-z\s]+)/i
    ];

    let ingredientName = null;
    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            ingredientName = match[1].replace(/\?/g, '').trim();
            break;
        }
    }

    if (!ingredientName) {
        return {
            response: "I'd love to help with substitutions! Please specify which ingredient you'd like to substitute. For example: 'Substitute for eggs' or 'What can I use instead of butter?'",
            category: 'substitution'
        };
    }

    const ingredient = findIngredient(ingredientName);

    if (!ingredient) {
        return {
            response: `I couldn't find "${ingredientName}" in our database of 236+ ingredients. Try searching for common ingredients like: butter, eggs, milk, flour, sugar, oil, cheese, or any vegetables, proteins, or grains.`,
            category: 'substitution'
        };
    }

    // Get detailed substitution rules first
    const detailedSubs = getDetailedSubstitutions(ingredient.id);
    const basicSubs = getSubstitutions(ingredient.id);

    if (detailedSubs.length === 0 && basicSubs.length === 0) {
        return {
            response: `I don't have specific substitution rules for **${ingredient.label}** yet. However, consider ingredients with similar properties in the **${ingredient.category}** category. Check the graph visualization below to explore similar ingredients!`,
            category: 'substitution'
        };
    }

    let response = `## Substitutes for ${ingredient.label}\n\n`;

    // Show detailed substitutions first
    if (detailedSubs.length > 0) {
        detailedSubs.forEach((sub, index) => {
            const confidence = sub.confidence === 'high' ? '✅' : '⚠️';
            response += `${index + 1}. **${sub.ingredient.label}** ${confidence}\n`;
            response += `   - **Ratio**: ${sub.ratio}\n`;
            response += `   - **Notes**: ${sub.notes}\n`;
            if (sub.dietary && sub.dietary.length > 0) {
                response += `   - **Dietary**: ${sub.dietary.join(', ')}\n`;
            }
            response += '\n';
        });
    } else {
        // Show basic substitutions
        basicSubs.forEach((sub, index) => {
            response += `${index + 1}. **${sub.ingredient.label}**\n`;
            if (sub.ratio) response += `   - **Ratio**: ${sub.ratio}\n`;
            if (sub.context) response += `   - **Best for**: ${sub.context}\n`;
            response += '\n';
        });
    }

    response += `\n💡 *Click on **${ingredient.label}** in the graph below to see visual connections!*`;

    return { response, category: 'substitution' };
}

// Handle pairing queries
function handlePairingQuery(userMessage, input) {
    // Extract ingredient name
    const patterns = [
        /pair(?:s)? (?:well )?with ([a-z\s]+)/i,
        /goes well with ([a-z\s]+)/i,
        /complement(?:s)? ([a-z\s]+)/i,
        /what (?:goes|pairs) with ([a-z\s]+)/i
    ];

    let ingredientName = null;
    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            ingredientName = match[1].replace(/\?/g, '').trim();
            break;
        }
    }

    if (!ingredientName) {
        return {
            response: "I can help you discover flavor pairings! Tell me which ingredient you're curious about. For example: 'What pairs well with garlic?' or 'What goes with tomatoes?'",
            category: 'pairing'
        };
    }

    const ingredient = findIngredient(ingredientName);

    if (!ingredient) {
        return {
            response: `I couldn't find "${ingredientName}" in our database. Try searching for herbs, spices, proteins, or vegetables like: garlic, basil, tomato, chicken, or olive oil.`,
            category: 'pairing'
        };
    }

    const pairings = getPairings(ingredient.id);

    if (pairings.length === 0) {
        return {
            response: `I don't have specific pairing data for **${ingredient.label}** yet. Explore the graph visualization below to see what ingredients are commonly used together!`,
            category: 'pairing'
        };
    }

    let response = `## Ingredients that Pair Well with ${ingredient.label}\n\n`;

    const strongPairs = pairings.filter(p => p.strength === 'strong');
    const mediumPairs = pairings.filter(p => p.strength === 'medium');

    if (strongPairs.length > 0) {
        response += `### 🌟 Strong Pairings\n`;
        strongPairs.forEach(pair => {
            response += `- **${pair.ingredient.label}**\n`;
        });
        response += '\n';
    }

    if (mediumPairs.length > 0) {
        response += `### ⭐ Good Pairings\n`;
        mediumPairs.forEach(pair => {
            response += `- **${pair.ingredient.label}**\n`;
        });
        response += '\n';
    }

    response += `\n💡 *Explore the graph to discover more connections!*`;

    return { response, category: 'pairing' };
}

// Handle recipe combination queries
function handleRecipeQuery(userMessage, input) {
    // Extract ingredient name
    const patterns = [
        /recipe(?:s)? with ([a-z\s]+)/i,
        /used with ([a-z\s]+)/i,
        /combine(?:s)? (?:with )?([a-z\s]+)/i
    ];

    let ingredientName = null;
    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            ingredientName = match[1].replace(/\?/g, '').trim();
            break;
        }
    }

    if (!ingredientName) {
        return {
            response: "I can suggest recipe combinations! Tell me which ingredient you want to use. For example: 'Recipes with chickpeas' or 'What's used with rice?'",
            category: 'recipe'
        };
    }

    const ingredient = findIngredient(ingredientName);

    if (!ingredient) {
        return {
            response: `I couldn't find "${ingredientName}". Try common ingredients like: rice, chicken, beans, pasta, or any vegetables.`,
            category: 'recipe'
        };
    }

    const combinations = getRecipeCombinations(ingredient.id);
    const pairings = getPairings(ingredient.id);

    if (combinations.length === 0 && pairings.length === 0) {
        return {
            response: `I don't have specific recipe combinations for **${ingredient.label}** yet. Explore the graph below to see visual connections with other ingredients!`,
            category: 'recipe'
        };
    }

    let response = `## Recipe Ideas with ${ingredient.label}\n\n`;

    if (combinations.length > 0) {
        response += `### Common Combinations\n`;
        combinations.forEach(combo => {
            response += `- **${combo.ingredient.label}**`;
            if (combo.context) response += ` _(${combo.context})_`;
            response += '\n';
        });
        response += '\n';
    }

    if (pairings.length > 0) {
        const strongPairs = pairings.filter(p => p.strength === 'strong').slice(0, 5);
        if (strongPairs.length > 0) {
            response += `### Flavor Pairings to Try\n`;
            strongPairs.forEach(pair => {
                response += `- **${pair.ingredient.label}**\n`;
            });
        }
    }

    response += `\n💡 *Mix and match these ingredients for delicious results!*`;

    return { response, category: 'recipe' };
}

// Handle dietary filtering queries
function handleDietaryQuery(userMessage, input) {
    let dietary = null;
    let category = null;

    if (input.includes('vegan')) dietary = 'vegan';
    else if (input.includes('vegetarian')) dietary = 'vegetarian';
    else if (input.includes('gluten-free') || input.includes('gluten free')) dietary = 'gluten-free';

    // Check for category
    if (input.includes('protein')) category = 'protein';
    else if (input.includes('dairy')) category = 'dairy';
    else if (input.includes('grain')) category = 'grain';
    else if (input.includes('oil')) category = 'oil';

    const results = filterIngredients({ dietary, category });

    if (results.length === 0) {
        return {
            response: `I couldn't find any ingredients matching those criteria. Try: "vegan proteins", "vegetarian options", or "gluten-free grains".`,
            category: 'dietary'
        };
    }

    let response = `## ${dietary.charAt(0).toUpperCase() + dietary.slice(1)} ${category ? category.charAt(0).toUpperCase() + category.slice(1) + 's' : 'Ingredients'}\n\n`;
    response += `Found **${results.length}** ingredients:\n\n`;

    // Group by category
    const grouped = {};
    results.forEach(node => {
        if (!grouped[node.category]) grouped[node.category] = [];
        grouped[node.category].push(node.label);
    });

    for (const [cat, items] of Object.entries(grouped)) {
        response += `### ${cat.charAt(0).toUpperCase() + cat.slice(1)}\n`;
        items.slice(0, 10).forEach(item => {
            response += `- ${item}\n`;
        });
        if (items.length > 10) {
            response += `- *...and ${items.length - 10} more*\n`;
        }
        response += '\n';
    }

    return { response, category: 'dietary' };
}

// Handle cuisine filtering queries
function handleCuisineQuery(userMessage, input) {
    let cuisine = null;

    if (input.includes('italian')) cuisine = 'italian';
    else if (input.includes('asian')) cuisine = 'asian';
    else if (input.includes('mexican')) cuisine = 'mexican';
    else if (input.includes('mediterranean')) cuisine = 'mediterranean';

    const results = filterIngredients({ cuisine });

    if (results.length === 0) {
        return {
            response: `I couldn't find ingredients for that cuisine. Try: "Italian ingredients", "Asian ingredients", "Mexican ingredients", or "Mediterranean ingredients".`,
            category: 'cuisine'
        };
    }

    let response = `## ${cuisine.charAt(0).toUpperCase() + cuisine.slice(1)} Cuisine Ingredients\n\n`;
    response += `Found **${results.length}** ingredients commonly used in ${cuisine} cooking:\n\n`;

    // Group by category
    const grouped = {};
    results.forEach(node => {
        if (!grouped[node.category]) grouped[node.category] = [];
        grouped[node.category].push(node.label);
    });

    for (const [cat, items] of Object.entries(grouped)) {
        response += `### ${cat.charAt(0).toUpperCase() + cat.slice(1)}\n`;
        items.slice(0, 8).forEach(item => {
            response += `- ${item}\n`;
        });
        if (items.length > 8) {
            response += `- *...and ${items.length - 8} more*\n`;
        }
        response += '\n';
    }

    return { response, category: 'cuisine' };
}

// Handle category search queries
function handleCategoryQuery(userMessage, input) {
    let category = null;

    if (input.includes('protein')) category = 'protein';
    else if (input.includes('dairy')) category = 'dairy';
    else if (input.includes('grain')) category = 'grain';
    else if (input.includes('vegetable')) category = 'vegetable';
    else if (input.includes('herb')) category = 'herb';
    else if (input.includes('spice')) category = 'spice';

    const results = filterIngredients({ category });

    if (results.length === 0) {
        return {
            response: `No ingredients found in that category. Try: proteins, dairy, grains, vegetables, herbs, or spices.`,
            category: 'category'
        };
    }

    let response = `## ${category.charAt(0).toUpperCase() + category.slice(1)} Ingredients\n\n`;
    response += `Our database includes **${results.length}** ${category} options:\n\n`;

    results.slice(0, 20).forEach(item => {
        response += `- **${item.label}**`;
        if (item.dietary && item.dietary.length > 0) {
            response += ` _(${item.dietary.join(', ')})_`;
        }
        response += '\n';
    });

    if (results.length > 20) {
        response += `\n*...and ${results.length - 20} more! Explore the graph to see them all.*`;
    }

    return { response, category: 'category' };
}

// Handle general ingredient info
function handleIngredientInfo(ingredient) {
    let response = `## ${ingredient.label}\n\n`;
    response += `**Category**: ${ingredient.category}\n`;
    response += `**Cuisines**: ${ingredient.cuisine.join(', ')}\n`;
    response += `**Dietary**: ${ingredient.dietary.join(', ')}\n\n`;

    const subs = getSubstitutions(ingredient.id);
    const pairs = getPairings(ingredient.id);
    const combos = getRecipeCombinations(ingredient.id);

    if (subs.length > 0) {
        response += `**Substitutes**: ${subs.slice(0, 3).map(s => s.ingredient.label).join(', ')}\n`;
    }

    if (pairs.length > 0) {
        response += `**Pairs well with**: ${pairs.slice(0, 3).map(p => p.ingredient.label).join(', ')}\n`;
    }

    if (combos.length > 0) {
        response += `**Used with**: ${combos.slice(0, 3).map(c => c.ingredient.label).join(', ')}\n`;
    }

    response += `\n💡 *Ask me about substitutions, pairings, or recipes with ${ingredient.label}!*`;

    return { response, category: 'ingredient' };
}

// ============================================
// AI INTEGRATION FUNCTIONS
// ============================================

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
                    maxOutputTokens: 800  // Concise but comprehensive responses
                }
            })
        }
    );

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Backend API error: ${response.status} - ${errorData.error || response.statusText}`);
    }

    const data = await response.json();

    // Handle Gemini 2.5 response format
    if (data.candidates && data.candidates[0]) {
        const candidate = data.candidates[0];

        // Try to get text from parts array
        if (candidate.content && candidate.content.parts && candidate.content.parts[0]) {
            return candidate.content.parts[0].text;
        }

        // Fallback: check if there's text directly in content
        if (candidate.content && candidate.content.text) {
            return candidate.content.text;
        }

        // If no text found, throw error
        throw new Error('No text content in Gemini response');
    }

    throw new Error('Invalid response format from Gemini API');
}

// Main LLM call via backend proxy
async function getLLMResponse(userMessage) {
    // Build conversation history
    const messages = [
        { role: 'system', content: culinaryAiConfig.getSystemPrompt() },
        ...conversationHistory,
        { role: 'user', content: userMessage }
    ];

    // Call backend proxy (which securely handles the API key)
    const response = await callGemini(messages);
    return { response, provider: 'Gemini 2.5 Flash' };
}

// Enhance local knowledge base response with context
function enhanceWithContext(userMessage, localResult) {
    // Add ingredient count and database info to help AI
    let context = '';
    if (typeof ingredientNodes !== 'undefined' && typeof ingredientEdges !== 'undefined') {
        context = `\n\n[Knowledge Base Context: ${ingredientNodes.length} ingredients, ${ingredientEdges.length} relationships]`;
    }

    // If we have local results, provide them as context
    if (localResult && localResult.response) {
        return {
            userMessage: userMessage + context,
            localData: localResult.response
        };
    }

    return {
        userMessage: userMessage + context,
        localData: null
    };
}

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

    try {
        // Try AI first if enabled
        if (useAI) {
            try {
                const result = await getLLMResponse(userMessage);
                removeTypingIndicator();

                // Add response with provider badge
                const responseWithBadge = `<span class="ai-provider-badge">${result.provider}</span>${result.response}`;
                addMessage(responseWithBadge, true);

                // Update conversation history
                conversationHistory.push({ role: 'user', content: userMessage });
                conversationHistory.push({ role: 'assistant', content: result.response });

                if (conversationHistory.length > 20) {
                    conversationHistory = conversationHistory.slice(-20);
                }
            } catch (aiError) {
                console.error('AI response failed, falling back to local knowledge base:', aiError);
                // Fall back to local knowledge base
                const result = processQuery(userMessage);
                removeTypingIndicator();
                addMessage(`<span class="ai-provider-badge">Local Knowledge Base</span>${result.response}`, true, result.category);
            }
        } else {
            // Use local knowledge base only
            const result = processQuery(userMessage);
            removeTypingIndicator();
            addMessage(result.response, true, result.category);

            // Update conversation history for local mode too
            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: result.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        }
    } catch (error) {
        console.error('Error processing query:', error);
        removeTypingIndicator();
        addMessage("I'm sorry, I encountered an error processing your question. Please try rephrasing it or ask something else!", true);
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

// Update AI status badge
function updateAIStatus(isBackendHealthy) {
    const statusBadge = document.getElementById('culinaryAIStatusBadge');
    if (statusBadge) {
        if (isBackendHealthy) {
            statusBadge.textContent = '🤖 AI: Gemini (Active)';
            statusBadge.className = 'ai-status-badge active';
            useAI = true;
        } else {
            statusBadge.textContent = '🤖 AI: Offline (Local Mode)';
            statusBadge.className = 'ai-status-badge';
            useAI = false;
        }
    }
}

// Initialize chat on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Culinary Chat Assistant initialized');

    // Check if ingredientNodes is available
    if (typeof ingredientNodes !== 'undefined' && typeof ingredientEdges !== 'undefined') {
        console.log(`Loaded ${ingredientNodes.length} ingredients and ${ingredientEdges.length} relationships`);
    } else {
        console.warn('Warning: ingredientNodes or ingredientEdges not loaded yet');
    }

    // Initialize AI configuration
    try {
        if (typeof CulinaryAIConfig !== 'undefined') {
            culinaryAiConfig = new CulinaryAIConfig();

            // Check backend health
            const isHealthy = await culinaryAiConfig.checkBackendHealth();
            updateAIStatus(isHealthy);

            if (isHealthy) {
                console.log('AI backend is healthy - Gemini 2.5 Flash active');
            } else {
                console.log('AI backend unavailable - using local knowledge base');
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
