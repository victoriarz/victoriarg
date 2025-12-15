// Saponify AI - Soap Making Chat Assistant with LLM Integration
// Uses local knowledge bank first, then falls back to AI API to save tokens

// Initialize AI configuration
let aiConfig;
let conversationHistory = [];
let soapCalculator = null;  // Will be initialized when SoapCalculator loads
let recipeValidator = null;  // Will be initialized when RecipeValidator loads
let knowledgeBank = null;    // Will be loaded from soap-knowledge-bank.js

// NOTE: SAP values are now managed by SoapCalculator class in soap-calculator.js
// This duplicate data structure has been removed to avoid inconsistencies

// Conversation state for recipe building
let recipeState = {
    active: false,
    batchSizeGrams: 0,
    oils: [],
    superfat: 5
};

// Conversation context tracking for follow-up questions
let conversationContext = {
    lastTopic: null,           // 'recipe', 'troubleshooting', 'oils', 'safety', etc.
    lastOilMentioned: null,    // Track last oil discussed for follow-ups
    lastRecipeDiscussed: null, // Store recipe details for modifications
    questionCount: 0           // Track depth of conversation on topic
};

// Recipe draft auto-save key for localStorage
const RECIPE_DRAFT_KEY = 'saponify_recipe_draft';

// ===========================================
// RECIPE DRAFT PERSISTENCE
// Auto-saves recipe progress so users don't lose work
// ===========================================

/**
 * Save current recipe state to localStorage
 */
function saveRecipeDraft() {
    if (!recipeState.active) return;

    try {
        const draft = {
            ...recipeState,
            savedAt: Date.now()
        };
        localStorage.setItem(RECIPE_DRAFT_KEY, JSON.stringify(draft));
        console.log('📝 Recipe draft saved');
    } catch (e) {
        console.warn('Could not save recipe draft:', e);
    }
}

/**
 * Load recipe draft from localStorage
 * Returns true if a draft was loaded
 */
function loadRecipeDraft() {
    try {
        const saved = localStorage.getItem(RECIPE_DRAFT_KEY);
        if (!saved) return false;

        const draft = JSON.parse(saved);

        // Only restore if draft is less than 1 hour old
        const ageHours = (Date.now() - draft.savedAt) / (1000 * 60 * 60);
        if (ageHours > 1) {
            clearRecipeDraft();
            return false;
        }

        // Restore state
        recipeState = {
            active: draft.active,
            batchSizeGrams: draft.batchSizeGrams || 0,
            oils: draft.oils || [],
            superfat: draft.superfat || 5,
            pendingOils: draft.pendingOils || null
        };

        console.log('📝 Recipe draft restored');
        return true;
    } catch (e) {
        console.warn('Could not load recipe draft:', e);
        return false;
    }
}

/**
 * Clear saved recipe draft
 */
function clearRecipeDraft() {
    try {
        localStorage.removeItem(RECIPE_DRAFT_KEY);
    } catch (e) {
        // Ignore errors
    }
}

/**
 * Check if there's a saved draft and offer to resume
 */
function checkForSavedDraft() {
    try {
        const saved = localStorage.getItem(RECIPE_DRAFT_KEY);
        if (!saved) return null;

        const draft = JSON.parse(saved);
        const ageMinutes = Math.round((Date.now() - draft.savedAt) / (1000 * 60));

        if (ageMinutes > 60) {
            clearRecipeDraft();
            return null;
        }

        // Build description of saved progress
        let description = '';
        if (draft.batchSizeGrams > 0) {
            description += `${draft.batchSizeGrams}g batch`;
        }
        if (draft.oils && draft.oils.length > 0) {
            const oilNames = draft.oils.map(o => o.name).join(', ');
            description += description ? ` with ${oilNames}` : oilNames;
        }
        if (draft.pendingOils && draft.pendingOils.length > 0) {
            description += description ? ` (oils: ${draft.pendingOils.join(', ')})` : draft.pendingOils.join(', ');
        }

        return {
            ageMinutes,
            description: description || 'recipe in progress',
            draft
        };
    } catch (e) {
        return null;
    }
}

// ===========================================
// USER-FRIENDLY ERROR MESSAGES
// Provides helpful feedback when things go wrong
// ===========================================

/**
 * Generate user-friendly error message based on error type
 */
function generateUserFriendlyError(error, userMessage) {
    const errorMsg = error.message || '';

    // Network/API errors
    if (errorMsg.includes('network') || errorMsg.includes('fetch') || errorMsg.includes('Failed to fetch')) {
        return `**Connection Issue**\n\n` +
            `I couldn't reach the AI service. This might be a temporary network issue.\n\n` +
            `**In the meantime, I can still help with:**\n` +
            `- Calculating soap recipes (just tell me your oils!)\n` +
            `- Answering questions from my knowledge base\n` +
            `- Validating recipe safety\n\n` +
            `Try your question again, or ask about something specific like "calculate a recipe with olive oil and coconut oil".`;
    }

    // API key/auth errors
    if (errorMsg.includes('401') || errorMsg.includes('403') || errorMsg.includes('API key') || errorMsg.includes('unauthorized')) {
        return `**AI Service Unavailable**\n\n` +
            `The AI service isn't available right now, but no worries!\n\n` +
            `**I can still help you with:**\n` +
            `- Recipe calculations (my specialty!)\n` +
            `- Soap making knowledge\n` +
            `- Oil properties and recommendations\n\n` +
            `What would you like to know about soap making?`;
    }

    // Rate limit errors
    if (errorMsg.includes('429') || errorMsg.includes('rate limit') || errorMsg.includes('too many requests')) {
        return `**Please wait a moment**\n\n` +
            `I'm getting a lot of questions right now! Give me about 30 seconds and try again.\n\n` +
            `While you wait, here's a tip: I can calculate soap recipes instantly without needing the AI - just say "create a recipe"!`;
    }

    // Recipe calculation errors
    if (errorMsg.includes('oil') || errorMsg.includes('SAP') || errorMsg.includes('calculate')) {
        return `**Recipe Calculation Issue**\n\n` +
            `I had trouble with that calculation. This usually happens when:\n` +
            `- An oil name wasn't recognized\n` +
            `- The amounts don't add up correctly\n\n` +
            `**Try again with:**\n` +
            `- Standard oil names: "olive oil", "coconut oil", "shea butter"\n` +
            `- Clear amounts: "300g olive oil, 200g coconut oil"\n\n` +
            `Or just list your oils and I'll suggest amounts!`;
    }

    // Generic fallback with helpful suggestions
    const legacyResult = findResponse(userMessage);

    if (legacyResult.category !== null) {
        return legacyResult.response;
    }

    return `**Hmm, something went wrong**\n\n` +
        `I couldn't process that request, but I'm still here to help!\n\n` +
        `**Try asking about:**\n` +
        `- Creating a soap recipe\n` +
        `- Oil properties and recommendations\n` +
        `- Cold process vs hot process\n` +
        `- Troubleshooting soap problems\n` +
        `- Lye safety\n\n` +
        `What would you like to know?`;
}

/**
 * Generate specific error for recipe validation failures
 */
function generateRecipeError(issue, context) {
    const errors = {
        'unknown_oil': {
            title: 'Unknown Oil',
            message: `I don't recognize "${context.oilName}" in my database.`,
            suggestions: [
                'Check the spelling',
                'Try common names like "olive oil" instead of brand names',
                'Ask "what oils do you know about?" to see available options'
            ]
        },
        'invalid_amount': {
            title: 'Invalid Amount',
            message: `I couldn't understand the amount "${context.amount}".`,
            suggestions: [
                'Use grams: "300g olive oil"',
                'Use ounces: "10 oz coconut oil"',
                'Use percentages: "30% olive oil"'
            ]
        },
        'batch_too_small': {
            title: 'Batch Too Small',
            message: `${context.grams}g is quite small for a batch.`,
            suggestions: [
                'Minimum recommended: 200g',
                'Small test batch: 500g',
                'Standard batch: 1000g'
            ]
        },
        'batch_too_large': {
            title: 'Very Large Batch',
            message: `${context.grams}g is a large batch! Make sure you have the equipment.`,
            suggestions: [
                'Large batches need bigger containers',
                'Consider splitting into multiple batches',
                'Ensure you have a large enough mold'
            ]
        },
        'high_coconut': {
            title: 'High Coconut Oil',
            message: `${context.percent}% coconut oil may be drying for some skin types.`,
            suggestions: [
                'Consider reducing to 20-30%',
                'Increase superfat to 8-10%',
                'Or keep it for a "cleansing" bar'
            ]
        },
        'no_hard_oils': {
            title: 'No Hard Oils',
            message: 'Your recipe has no hard oils - bars may be soft.',
            suggestions: [
                'Add coconut oil (15-30%)',
                'Add palm oil or shea butter (20-30%)',
                'Or make a "bastille" style soft bar'
            ]
        }
    };

    const errorInfo = errors[issue] || {
        title: 'Recipe Issue',
        message: 'There might be an issue with your recipe.',
        suggestions: ['Try adjusting the amounts', 'Ask for a beginner recipe']
    };

    let response = `**${errorInfo.title}**\n\n${errorInfo.message}\n\n**Suggestions:**\n`;
    errorInfo.suggestions.forEach(s => {
        response += `- ${s}\n`;
    });

    return response;
}

/**
 * Update conversation context based on response category and content
 */
function updateConversationContext(category, userMessage) {
    // Track topic
    if (category) {
        if (conversationContext.lastTopic === category) {
            conversationContext.questionCount++;
        } else {
            conversationContext.lastTopic = category;
            conversationContext.questionCount = 1;
        }
    }

    // Extract and track oil mentions
    const mentionedOils = extractOilsFromText(userMessage);
    if (mentionedOils.length > 0) {
        conversationContext.lastOilMentioned = mentionedOils[mentionedOils.length - 1];
    }

    // Track recipe context
    if (lastCalculatedRecipe) {
        conversationContext.lastRecipeDiscussed = {
            timestamp: lastCalculatedRecipeTime,
            oils: lastCalculatedRecipe.recipe.oils.map(o => o.name),
            superfat: lastCalculatedRecipe.recipe.superfat
        };
    }
}

/**
 * Check for follow-up questions that reference previous context
 * Handles: "what about...", "and coconut?", "tell me more", etc.
 */
function checkFollowUpQuestion(input) {
    const followUpPatterns = {
        // "Tell me more" / "More details"
        moreInfo: /\b(tell me more|more (info|details|about)|elaborate|explain more|go on)\b/i,
        // "What about X?" - referencing a different oil/topic
        whatAbout: /\bwhat about\b/i,
        // "And X?" - adding to previous question
        andQuestion: /^and\s+(\w+)/i,
        // "How about X instead?"
        howAbout: /\bhow about\b/i,
        // "Is that good?" / "Is that safe?" - referencing previous recipe
        isQuestion: /\bis (that|it|this)\s+(good|safe|okay|ok|right)\b/i,
        // "Why?" / "Why not?" - asking for explanation of previous answer
        why: /^why\b(?!\s+do|\s+should|\s+would)/i,
        // "Can I use X instead?"
        canI: /\bcan i\s+(use|substitute|swap|replace)\b/i
    };

    // Check if there's context to reference
    if (!conversationContext.lastTopic && !lastCalculatedRecipe) {
        return null;
    }

    // "Tell me more" about current topic
    if (followUpPatterns.moreInfo.test(input) && conversationContext.lastTopic) {
        return handleMoreInfoRequest(conversationContext.lastTopic);
    }

    // "What about [oil]?" when discussing oils
    if (followUpPatterns.whatAbout.test(input)) {
        const oils = extractOilsFromText(input);
        if (oils.length > 0) {
            return getOilInfo(oils[0]);
        }
    }

    // "And coconut?" style questions
    const andMatch = input.match(followUpPatterns.andQuestion);
    if (andMatch && conversationContext.lastTopic === 'oils') {
        const oils = extractOilsFromText(input);
        if (oils.length > 0) {
            return getOilInfo(oils[0]);
        }
    }

    // "Is that safe?" after a recipe
    if (followUpPatterns.isQuestion.test(input) && lastCalculatedRecipe) {
        const recipe = lastCalculatedRecipe.recipe;
        if (recipeValidator) {
            const validation = recipeValidator.validateRecipe(recipe);
            if (validation.valid && validation.warnings.length === 0) {
                return {
                    response: `Yes! Your recipe looks good. It has balanced properties and is safe to make. Just remember to:\n\n` +
                        `- Wear safety gear (goggles, gloves)\n` +
                        `- Add lye to water, never water to lye\n` +
                        `- Work in a well-ventilated area\n` +
                        `- Cure for 4-6 weeks before use`,
                    category: 'safety'
                };
            } else {
                return {
                    response: recipeValidator.formatValidationMessage(validation) +
                        `\nWould you like me to help adjust the recipe?`,
                    category: 'recipe'
                };
            }
        }
    }

    // "Can I use X instead?" - oil substitution
    if (followUpPatterns.canI.test(input) && lastCalculatedRecipe) {
        const oils = extractOilsFromText(input);
        if (oils.length > 0) {
            return handleOilSwapRequest(input, lastCalculatedRecipe.recipe);
        }
    }

    return null;
}

/**
 * Handle "tell me more" requests based on topic
 */
function handleMoreInfoRequest(topic) {
    const moreInfoResponses = {
        'recipe': {
            response: `Here's more about soap recipe formulation:\n\n` +
                `**Key principles:**\n` +
                `- **Hard oils** (coconut, palm, butters): 30-50% for bar hardness\n` +
                `- **Soft oils** (olive, sweet almond): 40-60% for conditioning\n` +
                `- **Castor oil**: 5-10% to boost lather\n\n` +
                `**Superfat** (5-8%): Extra oils for moisturizing\n` +
                `**Lye concentration** (28-33%): Higher = faster trace, faster unmold\n\n` +
                `Would you like me to explain any of these in more detail?`,
            category: 'recipe'
        },
        'oils': {
            response: `More about soap-making oils:\n\n` +
                `**For hardness**: Coconut, Palm, Cocoa Butter, Tallow, Lard\n` +
                `**For conditioning**: Olive, Avocado, Sweet Almond, Shea Butter\n` +
                `**For lather**: Coconut (bubbly), Castor (boosts all lather)\n` +
                `**Budget-friendly**: Canola, Sunflower, Lard\n` +
                `**Luxury**: Jojoba, Argan, Hemp Seed\n\n` +
                `Each oil has unique fatty acids that affect the final soap. Ask about any specific oil!`,
            category: 'oils'
        },
        'safety': {
            response: `More on soap-making safety:\n\n` +
                `**Lye handling:**\n` +
                `- Always add lye TO water (snow falls on the lake)\n` +
                `- Mix in ventilated area - fumes are caustic\n` +
                `- Solution reaches 200°F+ initially\n\n` +
                `**PPE required:**\n` +
                `- Chemical splash goggles (not just glasses)\n` +
                `- Long rubber gloves\n` +
                `- Long sleeves, closed-toe shoes\n\n` +
                `**First aid:** Flush with water 15+ minutes for any contact. Eye contact = ER immediately.`,
            category: 'safety'
        },
        'troubleshooting': {
            response: `Common soap problems and solutions:\n\n` +
                `**Soap won't trace:** Blend longer, check temps (100-120°F), verify lye measurements\n` +
                `**Soft/sticky bars:** Too much soft oil, not enough lye, or needs more cure time\n` +
                `**Crumbly/brittle:** Too much lye or cooked too hot\n` +
                `**White powder (soda ash):** Cosmetic only - wash off or prevent by covering mold\n` +
                `**Orange spots (DOS):** Rancid oils - use fresh oils and add vitamin E\n\n` +
                `What specific issue are you experiencing?`,
            category: 'troubleshooting'
        }
    };

    return moreInfoResponses[topic] || null;
}

/**
 * Get detailed info about a specific oil
 */
function getOilInfo(oilName) {
    if (!soapCalculator) return null;

    const oilData = soapCalculator.findOil(oilName);
    if (!oilData) return null;

    const props = oilData.properties || {};
    const response = `**${oilData.name}** properties:\n\n` +
        `- **SAP Value (NaOH):** ${oilData.sapNaOH}\n` +
        `- **Hardness:** ${props.hardness || 'N/A'}\n` +
        `- **Cleansing:** ${props.cleansing || 'N/A'}\n` +
        `- **Conditioning:** ${props.conditioning || 'N/A'}\n` +
        `- **Bubbly lather:** ${props.bubbly || 'N/A'}\n` +
        `- **Creamy lather:** ${props.creamy || 'N/A'}\n\n` +
        `**Recommended %:** ${oilData.recommendedPercent || '5-30%'}\n\n` +
        `Want to add this to a recipe or learn about another oil?`;

    // Update context
    conversationContext.lastOilMentioned = oilName;
    conversationContext.lastTopic = 'oils';

    return { response, category: 'oils' };
}

// Store last calculated recipe for quick wins
let lastCalculatedRecipe = null;
let lastCalculatedRecipeTime = null;

// Track if we're using local knowledge or AI
let lastResponseSource = 'local';

// Removed rate limiting - no artificial delays between requests

// Legacy knowledge base for soap making (kept as fallback)
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
        response: "Lye (sodium hydroxide/NaOH) is essential for soap making - there's no soap without it! It's caustic, so always use safety gear: goggles, gloves, long sleeves. Mix in well-ventilated area. ALWAYS add lye to water (not water to lye) to prevent explosive reaction. The soap calculator helps determine exact amounts needed. After saponification, no lye remains in properly made soap - it's all converted to soap molecules. Respect it, use it safely, and you'll make beautiful soap!"
    },
    'troubleshooting': {
        keywords: ['problem', 'wrong', 'failed', 'mistake', 'help', 'issue'],
        response: "Common issues: 1) Soap won't trace - check your scale, blend longer, or use stick blender. 2) Soap is crumbly - too much lye or not enough water. 3) Soap is soft/greasy - not enough lye or excess oils. 4) White powder (soda ash) - cosmetic only, wash off. 5) Orange spots - dreaded orange spot (DOS) from rancid oils. 6) Separation - oils and lye didn't mix properly, might be salvageable with rebatch. Always use a soap calculator and measure precisely!"
    },
    'recipe': {
        keywords: ['beginner recipe', 'basic recipe', 'simple recipe', 'starter recipe', 'first recipe'],
        response: "Here's a versatile beginner recipe: 35% Olive Oil, 30% Coconut Oil, 25% Palm Oil (or Shea Butter), 10% Castor Oil. For a 2 lb batch: 11.2 oz olive oil, 9.6 oz coconut oil, 8 oz palm oil, 3.2 oz castor oil, 4.4 oz lye, 10.5 oz water. Add 1 oz lavender essential oil at trace. This creates a hard, bubbly bar with good moisturizing properties. Always run any recipe through a soap calculator before making!"
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

// ===========================================
// KNOWLEDGE BANK SEARCH SYSTEM
// Searches the comprehensive soap-knowledge-bank.js first to save API tokens
// ===========================================

/**
 * Detect if user is asking an explicit knowledge-seeking question
 * These should be answered even during active recipe conversations
 * Examples: "what is saponification?", "explain cold process", "tell me about safety"
 */
function isKnowledgeSeekingQuestion(input) {
    // Patterns that indicate user wants information, not recipe building
    const knowledgePatterns = [
        // "What is/are..." questions
        /\bwhat\s+(is|are|does|do)\b/i,
        // "How does... work" (not "how much" which is recipe-related)
        /\bhow\s+(does|do|is)\b.*\bwork\b/i,
        // "Explain..." or "Tell me about..."
        /\b(explain|tell me about|describe|what's the difference)\b/i,
        // "Why..." questions about concepts
        /\bwhy\s+(is|are|do|does|should)\b/i,
        // "Can you explain..." or "Could you tell me..."
        /\b(can|could)\s+you\s+(explain|tell|describe)\b/i,
        // Direct topic questions: "what about saponification", "about cold process"
        /\babout\s+(saponification|cold process|hot process|lye safety|curing|trace|superfat|dos|soda ash)\b/i,
        // "Quick question" or "just wondering"
        /\b(quick question|just wondering|curious about|want to know)\b/i,
        // "What's the deal with..." or "What are the..."
        /\bwhat('s| is) the\b/i,
        // Safety questions
        /\b(is it safe|safety|dangerous|precaution|first aid)\b/i,
        // Troubleshooting questions (not during recipe building about the current recipe)
        /\b(why did my|what went wrong|problem with|troubleshoot|help my soap)\b/i,
        // Definition requests
        /\bdefin(e|ition)\b/i,
        // Method comparisons
        /\b(difference between|compare|vs|versus)\b/i
    ];

    return knowledgePatterns.some(pattern => pattern.test(input));
}

/**
 * Check if the input contains explicit recipe creation action words
 * These override knowledge question detection when present
 * Examples: "calculate", "create a recipe", "make me a batch"
 */
function hasExplicitRecipeActionWord(input) {
    const recipeActionPatterns = [
        /\b(calculate|create|make|build|formulate)\s+(a\s+)?(recipe|batch|soap)\b/i,
        /\bgive me\s+(a\s+)?recipe\b/i,
        /\b(start|begin)\s+(a\s+)?(new\s+)?recipe\b/i,
        /\brecipe\s+calculator\b/i,
        /\bcalculate\s+(the\s+)?lye\b/i,
        /\b(\d+)\s*(g|grams?|oz)\s+(batch|recipe)\b/i
    ];

    return recipeActionPatterns.some(pattern => pattern.test(input));
}

/**
 * Search the comprehensive knowledge bank for relevant information
 * Returns a formatted response if found, null if no good match
 */
function searchKnowledgeBank(userInput) {
    // If knowledge bank not loaded, return null to trigger API call
    if (typeof SOAP_KNOWLEDGE_BANK === 'undefined') {
        console.log('📚 Knowledge bank not loaded, will use API');
        return null;
    }

    const input = userInput.toLowerCase();
    const kb = SOAP_KNOWLEDGE_BANK;

    // Check if this is an explicit knowledge-seeking question (even during recipe building)
    const isExplicitKnowledgeQuestion = isKnowledgeSeekingQuestion(input);

    // If recipe building is active, only allow explicit knowledge questions through
    // This prevents oil names like "coconut" from triggering info lookups instead of recipe building
    if (recipeState.active && !isExplicitKnowledgeQuestion) {
        console.log('📚 Recipe conversation active - skipping knowledge bank search (not an explicit knowledge question)');
        return null;
    }

    if (recipeState.active && isExplicitKnowledgeQuestion) {
        console.log('📚 Explicit knowledge question detected during recipe - allowing knowledge bank search');
    }

    // Define keyword mappings to knowledge bank sections
    const searchMappings = [
        // Saponification & Chemistry
        {
            keywords: ['saponification', 'chemical reaction', 'how soap made', 'how is soap made', 'soap chemistry', 'triglyceride', 'fatty acid'],
            section: 'saponification',
            category: 'Chemistry',
            format: (data) => formatSaponificationResponse(data)
        },
        // SAP Values & Lye Calculation
        {
            keywords: ['sap value', 'saponification value', 'lye calculator', 'how much lye', 'calculate lye', 'lye amount'],
            section: 'saponification.sapValues',
            category: 'Chemistry',
            format: (data) => formatSapValuesResponse(data)
        },
        // Cold Process
        {
            keywords: ['cold process', 'cp soap', 'cold soap', 'cp method', 'cold process steps'],
            section: 'methods.coldProcess',
            category: 'Technique',
            format: (data) => formatColdProcessResponse(data)
        },
        // Hot Process
        {
            keywords: ['hot process', 'hp soap', 'crockpot soap', 'hot process steps', 'hp method'],
            section: 'methods.hotProcess',
            category: 'Technique',
            format: (data) => formatHotProcessResponse(data)
        },
        // Melt and Pour
        {
            keywords: ['melt and pour', 'melt pour', 'm&p', 'mp soap', 'glycerin soap', 'soap base'],
            section: 'methods.meltAndPour',
            category: 'Technique',
            format: (data) => formatMeltAndPourResponse(data)
        },
        // Liquid Soap
        {
            keywords: ['liquid soap', 'potassium hydroxide', 'koh soap', 'castile liquid'],
            section: 'methods.liquidSoap',
            category: 'Technique',
            format: (data) => formatLiquidSoapResponse(data)
        },
        // Oils - General
        {
            keywords: ['which oil', 'best oil', 'oils for soap', 'soap oils', 'oil properties', 'hard oil', 'soft oil'],
            section: 'oils',
            category: 'Ingredients',
            format: (data) => formatOilsOverviewResponse(data)
        },
        // Specific Oils
        {
            keywords: ['coconut oil', 'olive oil', 'palm oil', 'castor oil', 'shea butter', 'cocoa butter', 'sweet almond', 'avocado oil', 'lard', 'tallow'],
            section: 'oils.commonOils',
            category: 'Ingredients',
            format: (data, input) => formatSpecificOilResponse(data, input)
        },
        // Recipe Formulation
        {
            keywords: ['beginner recipe', 'recipe ratio', 'formulation', 'recipe percentage', '33/33/33', '34/33/33', 'palm free', 'castile recipe'],
            section: 'formulation.classicRatios',
            category: 'Recipe',
            format: (data) => formatRecipeRatiosResponse(data)
        },
        // Superfat
        {
            keywords: ['superfat', 'lye discount', 'how much superfat', 'superfat percentage'],
            section: 'formulation.superfat',
            category: 'Recipe',
            format: (data) => formatSuperfatResponse(data)
        },
        // Water Calculations
        {
            keywords: ['water discount', 'water ratio', 'lye concentration', 'how much water'],
            section: 'formulation.waterCalculations',
            category: 'Recipe',
            format: (data) => formatWaterCalculationsResponse(data)
        },
        // Soap Properties
        {
            keywords: ['soap properties', 'hardness', 'cleansing', 'conditioning', 'bubbly', 'creamy lather', 'ins value', 'iodine'],
            section: 'formulation.soapProperties',
            category: 'Recipe',
            format: (data) => formatSoapPropertiesResponse(data)
        },
        // Natural Colorants
        {
            keywords: ['natural color', 'colorant', 'purple soap', 'blue soap', 'green soap', 'pink soap', 'yellow soap', 'clay color', 'indigo', 'spirulina', 'turmeric', 'madder'],
            section: 'additives.naturalColorants',
            category: 'Design',
            format: (data, input) => formatColorantsResponse(data, input)
        },
        // Essential Oils & Fragrance
        {
            keywords: ['essential oil', 'fragrance oil', 'scent', 'how much essential', 'eo usage', 'fragrance rate', 'smell', 'aroma'],
            section: 'additives.fragrance',
            category: 'Fragrance',
            format: (data) => formatFragranceResponse(data)
        },
        // Exfoliants
        {
            keywords: ['exfoliant', 'scrub', 'oatmeal soap', 'coffee grounds', 'poppy seeds'],
            section: 'additives.exfoliants',
            category: 'Additives',
            format: (data) => formatExfoliantsResponse(data)
        },
        // Botanicals
        {
            keywords: ['botanical', 'flower', 'herb', 'lavender buds', 'rose petals', 'calendula', 'dried flower'],
            section: 'additives.botanicals',
            category: 'Additives',
            format: (data) => formatBotanicalsResponse(data)
        },
        // Special Additives
        {
            keywords: ['sodium lactate', 'honey soap', 'milk soap', 'goat milk', 'sugar', 'salt', 'vitamin e', 'roe', 'rosemary extract'],
            section: 'additives.specialAdditives',
            category: 'Additives',
            format: (data, input) => formatSpecialAdditivesResponse(data, input)
        },
        // Safety
        {
            keywords: ['safety', 'lye safety', 'protective', 'goggles', 'gloves', 'burn', 'first aid', 'dangerous', 'caustic'],
            section: 'safety',
            category: 'Safety',
            format: (data) => formatSafetyResponse(data)
        },
        // Troubleshooting - General
        {
            keywords: ['problem', 'troubleshoot', 'wrong', 'failed', 'help', 'issue', 'fix'],
            section: 'troubleshooting',
            category: 'Help',
            format: (data) => formatTroubleshootingOverview(data)
        },
        // Specific Troubleshooting Issues
        {
            keywords: ['lye heavy', 'zap test', 'burning', 'too much lye'],
            section: 'troubleshooting.lyeHeavySoap',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Lye Heavy Soap')
        },
        {
            keywords: ['soft soap', 'won\'t harden', 'squishy', 'too soft'],
            section: 'troubleshooting.softSoap',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Soft Soap')
        },
        {
            keywords: ['seizing', 'soap on a stick', 'seized', 'solid in pot'],
            section: 'troubleshooting.seizing',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Seizing')
        },
        {
            keywords: ['acceleration', 'trace too fast', 'thickening fast'],
            section: 'troubleshooting.acceleration',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Acceleration')
        },
        {
            keywords: ['ricing', 'rice like', 'lumps', 'curdled'],
            section: 'troubleshooting.ricing',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Ricing')
        },
        {
            keywords: ['separation', 'oil pooling', 'separating'],
            section: 'troubleshooting.separation',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Separation')
        },
        {
            keywords: ['soda ash', 'white powder', 'ash on soap'],
            section: 'troubleshooting.sodaAsh',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Soda Ash')
        },
        {
            keywords: ['dos', 'orange spots', 'rancid', 'dreaded orange'],
            section: 'troubleshooting.dreadedOrangeSpots',
            category: 'Help',
            format: (data) => formatSpecificTroubleshooting(data, 'Dreaded Orange Spots (DOS)')
        },
        // Curing
        {
            keywords: ['cure', 'curing', 'how long cure', 'cure time', 'when ready', 'storage'],
            section: 'curingAndStorage',
            category: 'Process',
            format: (data) => formatCuringResponse(data)
        },
        // Design Techniques
        {
            keywords: ['swirl', 'design', 'layer', 'drop swirl', 'hanger swirl', 'taiwan swirl', 'in the pot'],
            section: 'designTechniques',
            category: 'Design',
            format: (data) => formatDesignResponse(data)
        },
        // Business & Regulations
        {
            keywords: ['sell soap', 'business', 'fda', 'regulation', 'label', 'labeling', 'legal', 'cosmetic'],
            section: 'businessRegulations',
            category: 'Business',
            format: (data) => formatBusinessResponse(data)
        },
        // Glossary
        {
            keywords: ['what is', 'what does', 'define', 'meaning of', 'term'],
            section: 'glossary',
            category: 'Info',
            format: (data, input) => formatGlossaryResponse(data, input)
        }
    ];

    // Search for matching keywords with RELEVANCE RANKING
    // Score each mapping based on number and quality of keyword matches
    const scoredMatches = [];

    for (const mapping of searchMappings) {
        let score = 0;
        let matchedKeywords = [];

        for (const kw of mapping.keywords) {
            const kwLower = kw.toLowerCase();
            if (input.includes(kwLower)) {
                // Base score for match
                let keywordScore = 1;

                // Bonus for longer keywords (more specific)
                if (kw.length > 10) keywordScore += 1;

                // Bonus for exact phrase match at word boundary
                const wordBoundaryRegex = new RegExp(`\\b${kwLower.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
                if (wordBoundaryRegex.test(input)) {
                    keywordScore += 2;
                }

                // Bonus if keyword appears at start of input
                if (input.startsWith(kwLower) || input.startsWith('what is ' + kwLower)) {
                    keywordScore += 1;
                }

                score += keywordScore;
                matchedKeywords.push(kw);
            }
        }

        if (score > 0) {
            scoredMatches.push({
                mapping,
                score,
                matchedKeywords
            });
        }
    }

    // Sort by score (highest first)
    scoredMatches.sort((a, b) => b.score - a.score);

    // Return best match
    if (scoredMatches.length > 0) {
        const best = scoredMatches[0];
        console.log(`📚 Knowledge bank match: "${best.matchedKeywords.join(', ')}" -> ${best.mapping.section} (score: ${best.score})`);

        // Log other candidates for debugging
        if (scoredMatches.length > 1) {
            console.log(`   Other candidates: ${scoredMatches.slice(1, 4).map(m => `${m.mapping.section}(${m.score})`).join(', ')}`);
        }

        // Navigate to the section in the knowledge bank
        const data = getNestedProperty(kb, best.mapping.section);
        if (data) {
            const response = best.mapping.format(data, input);
            if (response) {
                return { response, category: best.mapping.category, source: 'knowledge_bank' };
            }
        }
    }

    console.log('📚 No knowledge bank match found for:', input.substring(0, 50));
    return null;
}

/**
 * Helper to get nested property from object using dot notation
 */
function getNestedProperty(obj, path) {
    return path.split('.').reduce((current, key) => current && current[key], obj);
}

// ===========================================
// KNOWLEDGE BANK RESPONSE FORMATTERS
// ===========================================

function formatSaponificationResponse(data) {
    let response = `## What is Saponification?\n\n`;
    response += `${data.definition}\n\n`;
    response += `### The Chemical Reaction\n`;
    response += `**${data.chemicalReaction.equation}**\n\n`;
    response += `${data.chemicalReaction.explanation}\n\n`;
    response += `**By-products:** ${data.chemicalReaction.byproducts.join(', ')}\n\n`;
    response += `### Timeline\n`;
    for (const [stage, desc] of Object.entries(data.timeline)) {
        response += `- **${stage}:** ${desc}\n`;
    }
    return response;
}

function formatSapValuesResponse(data) {
    let response = `## SAP Values & Lye Calculation\n\n`;
    response += `**What is a SAP value?** ${data.definition}\n\n`;
    response += `### How to Calculate\n${data.calculation}\n\n`;
    response += `**Example:** ${data.example}\n\n`;
    response += `⚠️ **Important:** ${data.importance}`;
    return response;
}

function formatColdProcessResponse(data) {
    let response = `## Cold Process Soap Making\n\n`;
    response += `${data.description}\n\n`;
    response += `### Steps\n`;
    data.steps.forEach(step => {
        response += `${step}\n`;
    });
    response += `\n### What is Trace?\n${data.traceDefinition}\n\n`;
    response += `**Cure Time:** ${data.cureTime}\n\n`;
    response += `### Advantages\n`;
    data.advantages.forEach(adv => response += `- ${adv}\n`);
    response += `\n### Disadvantages\n`;
    data.disadvantages.forEach(dis => response += `- ${dis}\n`);
    return response;
}

function formatHotProcessResponse(data) {
    let response = `## Hot Process Soap Making\n\n`;
    response += `${data.description}\n\n`;
    response += `### Steps\n`;
    data.steps.forEach(step => response += `${step}\n`);
    response += `\n### Stages During Cooking\n`;
    data.stages.forEach(stage => response += `- ${stage}\n`);
    response += `\n**Cure Time:** ${data.cureTime}\n\n`;
    response += `⚠️ **Warning:** ${data.volcanoWarning}\n\n`;
    response += `### Advantages\n`;
    data.advantages.forEach(adv => response += `- ${adv}\n`);
    return response;
}

function formatMeltAndPourResponse(data) {
    let response = `## Melt and Pour Soap Making\n\n`;
    response += `${data.description}\n\n`;
    response += `### Steps\n`;
    data.steps.forEach(step => response += `${step}\n`);
    response += `\n### Available Base Types\n`;
    data.baseTypes.forEach(base => response += `- ${base}\n`);
    response += `\n**Ready to use:** ${data.cureTime}\n\n`;
    response += `### Tips\n`;
    data.tips.forEach(tip => response += `- ${tip}\n`);
    return response;
}

function formatLiquidSoapResponse(data) {
    let response = `## Liquid Soap Making\n\n`;
    response += `${data.description}\n\n`;
    response += `### Process\n`;
    data.process.forEach(step => response += `${step}\n`);
    response += `\n⚠️ **Note:** ${data.kohNote}`;
    return response;
}

function formatOilsOverviewResponse(data) {
    let response = `## Oils for Soap Making\n\n`;
    response += `### Hard Oils\n${data.categories.hardOils.definition}\n`;
    response += `**Examples:** ${data.categories.hardOils.examples.join(', ')}\n\n`;
    response += `### Soft Oils\n${data.categories.softOils.definition}\n`;
    response += `**Examples:** ${data.categories.softOils.examples.join(', ')}\n\n`;
    response += `### Popular Oils at a Glance\n`;
    const topOils = ['coconutOil', 'oliveOil', 'palmOil', 'castorOil', 'sheaButter'];
    topOils.forEach(oilKey => {
        const oil = data.commonOils[oilKey];
        if (oil) {
            response += `- **${oilKey.replace(/([A-Z])/g, ' $1').trim()}**: ${oil.recommendedPercentage} - ${oil.properties.lather} lather\n`;
        }
    });
    return response;
}

function formatSpecificOilResponse(data, input) {
    // Find which oil the user is asking about
    const oilMappings = {
        'coconut': 'coconutOil',
        'olive': 'oliveOil',
        'palm oil': 'palmOil',
        'castor': 'castorOil',
        'shea': 'sheaButter',
        'cocoa': 'cocoaButter',
        'sweet almond': 'sweetAlmondOil',
        'avocado': 'avocadoOil',
        'lard': 'lard',
        'tallow': 'tallow',
        'sunflower': 'sunflowerOil',
        'rice bran': 'riceBranOil'
    };

    for (const [keyword, oilKey] of Object.entries(oilMappings)) {
        if (input.includes(keyword)) {
            const oil = data[oilKey];
            if (oil) {
                let response = `## ${oilKey.replace(/([A-Z])/g, ' $1').trim()}\n\n`;
                response += `**SAP Value (NaOH):** ${oil.sapValueNaOH}\n\n`;
                response += `### Properties\n`;
                response += `| Property | Level |\n|----------|-------|\n`;
                for (const [prop, value] of Object.entries(oil.properties)) {
                    response += `| ${prop} | ${value} |\n`;
                }
                response += `\n**Recommended Percentage:** ${oil.recommendedPercentage}\n\n`;
                response += `### Notes\n${oil.notes}`;
                return response;
            }
        }
    }
    return null;
}

function formatRecipeRatiosResponse(data) {
    let response = `## Classic Soap Recipe Ratios\n\n`;
    for (const [key, recipe] of Object.entries(data)) {
        response += `### ${recipe.name}\n`;
        for (const [oil, percent] of Object.entries(recipe.recipe)) {
            response += `- ${oil.replace(/([A-Z])/g, ' $1').trim()}: ${percent}\n`;
        }
        response += `*${recipe.notes}*\n\n`;
    }
    return response;
}

function formatSuperfatResponse(data) {
    let response = `## Superfat (Lye Discount)\n\n`;
    response += `**Definition:** ${data.definition}\n\n`;
    response += `### Recommended Superfat Percentages\n`;
    for (const [use, percent] of Object.entries(data.recommendations)) {
        response += `- **${use}:** ${percent}\n`;
    }
    response += `\n### Calculation\n${data.calculation}`;
    return response;
}

function formatWaterCalculationsResponse(data) {
    let response = `## Water & Lye Ratios\n\n`;
    response += `### Common Ratios\n`;
    for (const [name, info] of Object.entries(data)) {
        if (typeof info === 'object' && info.ratio) {
            response += `**${name}:** ${info.ratio} (${info.percentage})\n- ${info.use}\n\n`;
        }
    }
    response += `**Safe Range:** ${data.safeRange || 'Water:lye ratios between 3:1 (25% lye) and 1:1 (50% lye) are workable.'}`;
    return response;
}

function formatSoapPropertiesResponse(data) {
    let response = `## Soap Properties Explained\n\n`;
    for (const [prop, info] of Object.entries(data)) {
        response += `### ${prop.charAt(0).toUpperCase() + prop.slice(1)}\n`;
        response += `${info.description}\n`;
        response += `**Ideal Range:** ${info.idealRange}\n`;
        response += `**Increased by:** ${info.increasedBy.join(', ')}\n\n`;
    }
    return response;
}

function formatColorantsResponse(data, input) {
    let response = `## Natural Soap Colorants\n\n`;

    // Check if user is asking about a specific color
    const colorMap = {
        'purple': 'purples',
        'blue': 'blues',
        'green': 'greens',
        'pink': 'pinks',
        'yellow': 'yellows',
        'brown': 'browns',
        'black': 'blackGray',
        'gray': 'blackGray',
        'grey': 'blackGray'
    };

    let specificColor = null;
    for (const [keyword, colorKey] of Object.entries(colorMap)) {
        if (input.includes(keyword)) {
            specificColor = colorKey;
            break;
        }
    }

    if (specificColor && data[specificColor]) {
        const colorData = data[specificColor];
        response += `### ${specificColor.charAt(0).toUpperCase() + specificColor.slice(1)} Colorants\n`;
        for (const [name, info] of Object.entries(colorData)) {
            response += `**${name}:**\n`;
            if (info.color) response += `- Color: ${info.color}\n`;
            if (info.method) response += `- Method: ${info.method}\n`;
            if (info.usage) response += `- Usage: ${info.usage}\n`;
            if (info.notes) response += `- Notes: ${info.notes}\n`;
            response += `\n`;
        }
    } else {
        // General overview
        response += `### By Color\n`;
        const colors = ['purples', 'blues', 'greens', 'pinks', 'yellows', 'browns'];
        colors.forEach(color => {
            if (data[color]) {
                const options = Object.keys(data[color]).map(k => k.replace(/([A-Z])/g, ' $1').trim()).join(', ');
                response += `- **${color}:** ${options}\n`;
            }
        });
        response += `\n### Clays\n`;
        if (data.clays) {
            data.clays.types.forEach(type => response += `- ${type}\n`);
            response += `\n**Usage:** ${data.clays.usage}`;
        }
    }
    return response;
}

function formatFragranceResponse(data) {
    let response = `## Essential Oils & Fragrance in Soap\n\n`;
    response += `### Essential Oils\n`;
    response += `**Usage Rate:** ${data.essentialOils.usageRate}\n\n`;
    response += `**Scent Categories:**\n`;
    for (const [note, info] of Object.entries(data.essentialOils.categories)) {
        response += `- **${note}:** ${info.description} - Examples: ${info.examples.join(', ')}\n`;
    }
    response += `\n**Tip:** ${data.essentialOils.anchoring}\n\n`;
    response += `### Fragrance Oils\n`;
    response += `**Usage Rate:** ${data.fragranceOils.usageRate}\n`;
    response += `**For Melt & Pour:** ${data.fragranceOils.meltAndPour}\n\n`;
    response += `⚠️ **Behaviors to Watch:**\n`;
    for (const [behavior, desc] of Object.entries(data.fragranceOils.behaviors)) {
        response += `- **${behavior}:** ${desc}\n`;
    }
    return response;
}

function formatExfoliantsResponse(data) {
    let response = `## Exfoliants for Soap\n\n`;
    response += `### By Intensity\n`;
    response += `**Gentle:** ${data.gentle.join(', ')}\n`;
    response += `**Medium:** ${data.medium.join(', ')}\n`;
    response += `**Heavy:** ${data.heavy.join(', ')}\n\n`;
    response += `**Usage:** ${data.usage}\n\n`;
    response += `**Tip:** ${data.notes}`;
    return response;
}

function formatBotanicalsResponse(data) {
    let response = `## Using Botanicals in Soap\n\n`;
    response += `### Color Stability\n`;
    response += `**Hold their color:** ${data.holdColor.join(', ')}\n`;
    response += `**Turn brown:** ${data.turnBrown.join(', ')}\n\n`;
    response += `### When to Add Botanicals\n`;
    for (const [method, desc] of Object.entries(data.whenToAdd)) {
        response += `- **${method}:** ${desc}\n`;
    }
    response += `\n⚠️ ${data.allergenWarning}`;
    return response;
}

function formatSpecialAdditivesResponse(data, input) {
    let response = `## Special Soap Additives\n\n`;

    // Check for specific additive
    const additiveMap = {
        'sodium lactate': 'sodiumLactate',
        'sugar': 'sugar',
        'salt': 'salt',
        'honey': 'honey',
        'milk': 'milks',
        'goat milk': 'milks',
        'yogurt': 'yogurt',
        'oatmeal': 'oatmeal',
        'vitamin e': 'vitaminE',
        'roe': 'ROE',
        'rosemary': 'ROE'
    };

    let foundSpecific = false;
    for (const [keyword, key] of Object.entries(additiveMap)) {
        if (input.includes(keyword)) {
            const additive = data[key];
            if (additive) {
                response += `### ${key.replace(/([A-Z])/g, ' $1').trim()}\n`;
                response += `**Use:** ${additive.use}\n`;
                if (additive.rate) response += `**Rate:** ${additive.rate}\n`;
                if (additive.addTo) response += `**Add to:** ${additive.addTo}\n`;
                if (additive.method) response += `**Method:** ${additive.method}\n`;
                if (additive.notes) response += `**Notes:** ${additive.notes}\n`;
                if (additive.types) response += `**Types:** ${additive.types.join(', ')}\n`;
                foundSpecific = true;
                break;
            }
        }
    }

    if (!foundSpecific) {
        // Show overview of all additives
        for (const [key, additive] of Object.entries(data)) {
            response += `**${key}:** ${additive.use} (${additive.rate || 'varies'})\n`;
        }
    }
    return response;
}

function formatSafetyResponse(data) {
    let response = `## Soap Making Safety\n\n`;
    response += `### Personal Protective Equipment\n`;
    for (const [area, info] of Object.entries(data.personalProtection)) {
        response += `**${area}:** ${info.equipment}\n- *${info.importance}*\n\n`;
    }
    response += `### Lye Handling\n`;
    response += `⚠️ **Golden Rule:** ${data.lyeHandling.goldenRule}\n`;
    response += `*${data.lyeHandling.reason}*\n\n`;
    response += `**Safe Materials:** ${data.lyeHandling.materials.safe.join(', ')}\n`;
    response += `**Avoid:** ${data.lyeHandling.materials.avoid.join(', ')}\n\n`;
    response += `### First Aid\n`;
    response += `**Skin:** ${data.firstAid.skinExposure[0]}\n`;
    response += `**Eyes:** ${data.firstAid.eyeExposure[0]} - Seek medical attention immediately!\n`;
    response += `**Poison Control:** ${data.firstAid.poisonControl}`;
    return response;
}

function formatTroubleshootingOverview(data) {
    let response = `## Soap Making Troubleshooting\n\n`;
    response += `Here are common issues and how to fix them:\n\n`;
    const issues = ['lyeHeavySoap', 'softSoap', 'seizing', 'acceleration', 'ricing', 'separation', 'sodaAsh', 'dreadedOrangeSpots'];
    issues.forEach(issue => {
        if (data[issue]) {
            const title = issue.replace(/([A-Z])/g, ' $1').trim();
            response += `### ${title}\n`;
            response += `**Symptoms:** ${data[issue].symptoms.slice(0, 2).join(', ')}\n`;
            response += `**Main causes:** ${data[issue].causes.slice(0, 2).join(', ')}\n\n`;
        }
    });
    response += `*Ask me about a specific issue for detailed solutions!*`;
    return response;
}

function formatSpecificTroubleshooting(data, title) {
    let response = `## ${title}\n\n`;
    response += `### Symptoms\n`;
    data.symptoms.forEach(s => response += `- ${s}\n`);
    response += `\n### Causes\n`;
    data.causes.forEach(c => response += `- ${c}\n`);
    if (data.solutions) {
        response += `\n### Solutions\n`;
        data.solutions.forEach(s => response += `- ${s}\n`);
    }
    if (data.prevention) {
        response += `\n### Prevention\n`;
        data.prevention.forEach(p => response += `- ${p}\n`);
    }
    if (data.note) {
        response += `\n*Note: ${data.note}*`;
    }
    return response;
}

function formatCuringResponse(data) {
    let response = `## Curing Soap\n\n`;
    response += `### Why Cure?\n`;
    for (const [reason, desc] of Object.entries(data.whyCure)) {
        response += `- **${reason}:** ${desc}\n`;
    }
    response += `\n### Cure Times\n`;
    for (const [method, time] of Object.entries(data.cureTime)) {
        response += `- **${method}:** ${time}\n`;
    }
    response += `\n### How to Cure\n`;
    response += `**Environment:** ${data.curingMethod.environment}\n`;
    response += `**Setup:** ${data.curingMethod.setup}\n`;
    response += `**Position:** ${data.curingMethod.position}\n\n`;
    response += `### How to Know When Done\n`;
    for (const [test, desc] of Object.entries(data.howToKnowWhenCured)) {
        response += `- **${test}:** ${desc}\n`;
    }
    return response;
}

function formatDesignResponse(data) {
    let response = `## Soap Design Techniques\n\n`;
    response += `### Swirl Techniques\n`;
    for (const [name, info] of Object.entries(data.swirls)) {
        response += `**${name}** (${info.difficulty})\n`;
        response += `${info.description}\n`;
        response += `*Tip: ${info.tips}*\n\n`;
    }
    response += `### Tips for Success\n`;
    for (const [tip, desc] of Object.entries(data.tipsForSuccess)) {
        response += `- **${tip}:** ${desc}\n`;
    }
    return response;
}

function formatBusinessResponse(data) {
    let response = `## Selling Soap: Regulations & Requirements\n\n`;
    response += `### Product Classification\n`;
    response += `**True Soap:** ${data.productClassification.trueSoap.definition}\n`;
    response += `- Regulated by: ${data.productClassification.trueSoap.regulatedBy}\n\n`;
    response += `**Cosmetic:** ${data.productClassification.cosmetic.definition}\n`;
    response += `- Regulated by: ${data.productClassification.cosmetic.regulatedBy}\n\n`;
    response += `### Labeling Requirements\n`;
    response += `**For True Soap:** ${data.labelingRequirements.trueSoap.required.join(', ')}\n`;
    response += `**For Cosmetics:** Full ingredient list required\n\n`;
    response += `### Claims to Avoid\n`;
    response += `These claims change your product's classification:\n`;
    response += `- Cosmetic claims: ${data.claimsToAvoid.cosmeticClaims.join(', ')}\n`;
    response += `- Drug claims: ${data.claimsToAvoid.drugClaims.join(', ')}\n`;
    return response;
}

function formatGlossaryResponse(data, input) {
    // Try to find a specific term
    for (const [term, definition] of Object.entries(data)) {
        if (input.includes(term.toLowerCase())) {
            return `## ${term.charAt(0).toUpperCase() + term.slice(1)}\n\n${definition}`;
        }
    }

    // Return full glossary if no specific term found
    let response = `## Soap Making Glossary\n\n`;
    for (const [term, definition] of Object.entries(data)) {
        response += `**${term}:** ${definition}\n\n`;
    }
    return response;
}

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

    // Call backend proxy which handles Gemini API
    const apiUrl = `${aiConfig.getBackendUrl()}/api/chat`;
    console.log('🌐 Calling backend API');
    console.log('📍 Backend URL:', apiUrl);
    console.log('📦 Request payload:', { contentsLength: contents.length });

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for backend

    try {
        const response = await fetch(
            apiUrl,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: contents,
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 8192
                    }
                }),
                signal: controller.signal
            }
        );

        clearTimeout(timeoutId);

        console.log('📡 Backend response status:', response.status, response.statusText);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Backend error:', errorText);
            throw new Error(`Backend API error: ${response.status}`);
        }

        const data = await response.json();
        console.log('✅ Backend response received');

        // Handle Gemini response format
        if (data.candidates && data.candidates[0]) {
            const candidate = data.candidates[0];

            // Get text from parts array
            if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                const fullText = candidate.content.parts
                    .map(part => part.text || '')
                    .join('');

                if (fullText) {
                    return fullText;
                }
            }

            throw new Error('No text content in Gemini response');
        }

        throw new Error('Invalid response format from backend');
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - Backend took too long to respond');
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

    // Call Gemini API directly
    const response = await callGemini(messages);
    return { response, provider: 'Gemini 2.0 Flash' };
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
                              'th', 'td', 'hr', 'span', 'div', 'button'],
                ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'onclick', 'title'],
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
        // User messages with proper structure
        const userText = document.createElement('div');
        userText.className = 'user-text';
        userText.textContent = message;
        contentDiv.innerHTML = userText.outerHTML;
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

// ===========================================
// RECIPE INTENT DETECTION SYSTEM
// Detects when user wants to create/calculate a recipe
// Must run BEFORE knowledge bank to prevent oil names triggering info lookups
// ===========================================

/**
 * Check if user wants to modify a recently calculated recipe
 * Handles: "adjust that recipe", "make it bigger", "change superfat", "swap olive for avocado"
 */
function checkRecipeModificationIntent(input) {
    // Only check if we have a recent recipe (within last 10 minutes)
    const recipeAge = lastCalculatedRecipe ? (Date.now() - lastCalculatedRecipeTime) : Infinity;
    const isRecentRecipe = recipeAge < 10 * 60 * 1000; // 10 minutes

    if (!isRecentRecipe || !lastCalculatedRecipe) {
        return null;
    }

    const recipe = lastCalculatedRecipe.recipe;

    // Patterns for recipe modification
    const modificationPatterns = {
        // Scale/resize requests
        scale: /\b(scale|resize|double|triple|halve|half|make it|change.*(size|batch)|bigger|smaller)\b/i,
        // Superfat changes
        superfat: /\b(change|adjust|increase|decrease|modify).*(superfat|super\s*fat|lye\s*discount)\b/i,
        // Oil swap requests
        swap: /\b(swap|replace|substitute|switch|change|use).*(oil|butter|instead)\b/i,
        // Add/remove oil
        addOil: /\b(add|include|put in)\b.*\b(oil|butter|castor|shea|coconut|olive|palm)\b/i,
        removeOil: /\b(remove|take out|without|no)\b.*\b(oil|butter|castor|shea|coconut|olive|palm)\b/i,
        // Generic modification
        adjust: /\b(adjust|modify|change|tweak)\b.*(recipe|that|it)\b/i,
        recalculate: /\b(recalculate|redo|again)\b/i
    };

    // Check for scale request
    if (modificationPatterns.scale.test(input)) {
        return handleScaleRequest(input, recipe);
    }

    // Check for superfat change
    if (modificationPatterns.superfat.test(input)) {
        const newSuperfat = parseSuperfat(input);
        if (newSuperfat !== null) {
            return recalculateWithNewSuperfat(recipe, newSuperfat);
        }
        return {
            response: `The current recipe has **${recipe.superfat}% superfat**. What percentage would you like instead? (e.g., "7% superfat")`,
            category: 'recipe'
        };
    }

    // Check for oil swap
    if (modificationPatterns.swap.test(input)) {
        return handleOilSwapRequest(input, recipe);
    }

    // Check for add oil
    if (modificationPatterns.addOil.test(input)) {
        return handleAddOilRequest(input, recipe);
    }

    // Check for remove oil
    if (modificationPatterns.removeOil.test(input)) {
        return handleRemoveOilRequest(input, recipe);
    }

    // Generic adjust/modify request
    if (modificationPatterns.adjust.test(input) || modificationPatterns.recalculate.test(input)) {
        return {
            response: `I can help you modify the recipe! What would you like to change?\n\n` +
                `- **Scale**: "double it", "make it 1000g", "half the batch"\n` +
                `- **Superfat**: "change to 7% superfat"\n` +
                `- **Swap oils**: "replace palm with shea butter"\n` +
                `- **Add oils**: "add 10% castor oil"\n` +
                `- **Remove oils**: "remove the palm oil"\n\n` +
                `Current recipe: ${recipe.totalBatchSize.grams}g with ${recipe.superfat}% superfat`,
            category: 'recipe'
        };
    }

    return null;
}

/**
 * Handle recipe scaling request
 */
function handleScaleRequest(input, recipe) {
    const currentGrams = recipe.totalBatchSize.grams;
    let newGrams = currentGrams;

    // Check for multipliers
    if (/\bdouble\b/i.test(input)) {
        newGrams = currentGrams * 2;
    } else if (/\btriple\b/i.test(input)) {
        newGrams = currentGrams * 3;
    } else if (/\b(halve|half)\b/i.test(input)) {
        newGrams = Math.round(currentGrams / 2);
    } else if (/\bbigger\b/i.test(input)) {
        newGrams = Math.round(currentGrams * 1.5);
    } else if (/\bsmaller\b/i.test(input)) {
        newGrams = Math.round(currentGrams * 0.75);
    } else {
        // Try to extract specific size
        const batchResult = parseBatchSize(input);
        if (batchResult) {
            newGrams = batchResult.grams;
        }
    }

    if (newGrams === currentGrams) {
        return {
            response: `What size would you like? The current recipe is **${currentGrams}g**.\n\nYou can say "double it", "half the batch", or specify a size like "1000g".`,
            category: 'recipe'
        };
    }

    // Scale the oils proportionally
    const scaleFactor = newGrams / currentGrams;
    const scaledOils = recipe.oils.map(oil => ({
        name: oil.name,
        grams: Math.round(oil.grams * scaleFactor)
    }));

    return calculateAndFormatRecipe(scaledOils, recipe.superfat);
}

/**
 * Recalculate recipe with new superfat
 */
function recalculateWithNewSuperfat(recipe, newSuperfat) {
    const oils = recipe.oils.map(oil => ({
        name: oil.name,
        grams: oil.grams
    }));

    return calculateAndFormatRecipe(oils, newSuperfat);
}

/**
 * Handle oil swap request
 */
function handleOilSwapRequest(input, recipe) {
    const mentionedOils = extractOilsFromText(input);

    if (mentionedOils.length < 2) {
        const currentOils = recipe.oils.map(o => o.name).join(', ');
        return {
            response: `Which oils do you want to swap? Current recipe has: **${currentOils}**\n\nSay something like "replace palm oil with shea butter" or "swap coconut for babassu".`,
            category: 'recipe'
        };
    }

    // Find which oil is being removed and which is being added
    const [firstOil, secondOil] = mentionedOils;
    const existingOil = recipe.oils.find(o => o.name === firstOil);
    const existingOil2 = recipe.oils.find(o => o.name === secondOil);

    let removeOil, addOil;
    if (existingOil && !existingOil2) {
        removeOil = firstOil;
        addOil = secondOil;
    } else if (existingOil2 && !existingOil) {
        removeOil = secondOil;
        addOil = firstOil;
    } else {
        return {
            response: `I'm not sure which oil to remove and which to add. Please be more specific, like "replace ${firstOil} with ${secondOil}".`,
            category: 'recipe'
        };
    }

    // Create new oils array with swap
    const swappedOils = recipe.oils.map(oil => {
        if (oil.name === removeOil) {
            return { name: addOil, grams: oil.grams };
        }
        return { name: oil.name, grams: oil.grams };
    });

    return calculateAndFormatRecipe(swappedOils, recipe.superfat);
}

/**
 * Handle add oil request
 */
function handleAddOilRequest(input, recipe) {
    const mentionedOils = extractOilsFromText(input);
    const newOils = mentionedOils.filter(o => !recipe.oils.find(ro => ro.name === o));

    if (newOils.length === 0) {
        return {
            response: `Which oil would you like to add? The recipe already contains: ${recipe.oils.map(o => o.name).join(', ')}`,
            category: 'recipe'
        };
    }

    // Try to extract percentage or grams for the new oil
    const newOilName = newOils[0];
    let newOilGrams = 0;

    // Check for percentage
    const percentMatch = input.match(/(\d+)\s*%/);
    if (percentMatch) {
        const percent = parseInt(percentMatch[1]);
        newOilGrams = Math.round(recipe.totalBatchSize.grams * (percent / 100));
    } else {
        // Check for grams
        const gramsMatch = input.match(/(\d+)\s*g/i);
        if (gramsMatch) {
            newOilGrams = parseInt(gramsMatch[1]);
        } else {
            // Default to 10% of batch
            newOilGrams = Math.round(recipe.totalBatchSize.grams * 0.1);
        }
    }

    // Add new oil, proportionally reduce others
    const totalOldGrams = recipe.oils.reduce((sum, o) => sum + o.grams, 0);
    const scaleFactor = (totalOldGrams - newOilGrams) / totalOldGrams;

    const adjustedOils = recipe.oils.map(oil => ({
        name: oil.name,
        grams: Math.round(oil.grams * scaleFactor)
    }));

    adjustedOils.push({ name: newOilName, grams: newOilGrams });

    return calculateAndFormatRecipe(adjustedOils, recipe.superfat);
}

/**
 * Handle remove oil request
 */
function handleRemoveOilRequest(input, recipe) {
    const mentionedOils = extractOilsFromText(input);
    const oilsToRemove = mentionedOils.filter(o => recipe.oils.find(ro => ro.name === o));

    if (oilsToRemove.length === 0) {
        return {
            response: `Which oil would you like to remove? Current recipe has: ${recipe.oils.map(o => o.name).join(', ')}`,
            category: 'recipe'
        };
    }

    if (oilsToRemove.length >= recipe.oils.length) {
        return {
            response: `I can't remove all the oils! You need at least one oil in your recipe.`,
            category: 'recipe'
        };
    }

    // Remove oil and scale others proportionally
    const remainingOils = recipe.oils.filter(o => !oilsToRemove.includes(o.name));
    const removedGrams = recipe.oils
        .filter(o => oilsToRemove.includes(o.name))
        .reduce((sum, o) => sum + o.grams, 0);

    const totalRemainingGrams = remainingOils.reduce((sum, o) => sum + o.grams, 0);
    const scaleFactor = (totalRemainingGrams + removedGrams) / totalRemainingGrams;

    const adjustedOils = remainingOils.map(oil => ({
        name: oil.name,
        grams: Math.round(oil.grams * scaleFactor)
    }));

    return calculateAndFormatRecipe(adjustedOils, recipe.superfat);
}

/**
 * Check if user message indicates recipe creation intent
 * Returns response object if recipe intent detected, null otherwise
 */
function checkRecipeIntent(userInput) {
    const input = userInput.toLowerCase();

    // If recipe building is already active, handle that conversation
    if (recipeState.active) {
        // But first check if user is asking a knowledge question mid-recipe
        if (isKnowledgeSeekingQuestion(input)) {
            console.log('🔍 Knowledge question detected during recipe - deferring to knowledge bank');
            return null; // Let knowledge bank handle it
        }
        return handleActiveRecipeConversation(input);
    }

    // Check for recipe modification requests FIRST (if we have a recent recipe)
    const modificationResult = checkRecipeModificationIntent(input);
    if (modificationResult) {
        return modificationResult;
    }

    // IMPORTANT: If this is clearly an information-seeking question, don't treat it as recipe intent
    // This prevents "what oils should I use for conditioning soap?" from starting a recipe flow
    // when the user just wants information
    if (isKnowledgeSeekingQuestion(input) && !hasExplicitRecipeActionWord(input)) {
        console.log('🔍 Knowledge question detected - deferring to knowledge bank');
        return null;
    }

    // Patterns that indicate user wants to CREATE a recipe (not just learn about something)
    const recipeIntentPatterns = [
        // Direct recipe requests
        /\b(calculate|create|make|build|design|formulate)\b.*\b(recipe|soap|batch)\b/i,
        /\b(recipe|soap|batch)\b.*\b(for|with|using)\b/i,
        /\brecipe calculator\b/i,
        /\bhelp me (calculate|make|create)\b/i,

        // "I want to make" patterns
        /\bi want to (make|create|try)\b.*\bsoap\b/i,
        /\bi('d| would) like to (make|create)\b.*\bsoap\b/i,
        /\bcan you (make|create|give) me a\b.*\b(recipe|soap)\b/i,
        /\bgive me a\b.*\brecipe\b/i,
        /\bwhat('s| is) a good recipe\b/i,

        // Batch size with oils mentioned
        /\b(\d+)\s*(g|grams?|oz|ounces?|lbs?|pounds?)\b.*\b(batch|soap|recipe)\b/i,
        /\b(batch|soap|recipe)\b.*\b(\d+)\s*(g|grams?|oz|ounces?)\b/i,

        // Oil combinations suggesting recipe building
        /\b(using|with)\b.*\b(olive|coconut|palm|shea|castor)\b.*\b(and|,)\b.*\b(olive|coconut|palm|shea|castor)\b/i,

        // Recommendation requests
        /\brecommend\b.*\b(recipe|soap|formula)\b/i,
        /\bsuggest\b.*\b(recipe|soap|formula)\b/i,
        /\byour best\b.*\brecipe\b/i,
        /\bbeginner\b.*\brecipe\b/i,
        /\bfirst\b.*\b(batch|soap|recipe)\b/i,

        // Direct calculation requests
        /\bcalculate\b.*\blye\b/i,
        /\bhow much lye\b.*\bfor\b/i,
        /\blye.*(amount|calculation)\b/i
    ];

    // Check if any pattern matches
    const hasRecipeIntent = recipeIntentPatterns.some(pattern => pattern.test(input));

    if (!hasRecipeIntent) {
        return null; // No recipe intent - let knowledge bank handle it
    }

    console.log('🧮 Recipe intent detected in:', input.substring(0, 50));

    // Try to extract batch size if provided (use parseBatchSize for "small", "medium", "large" support)
    const batchSizeResult = parseBatchSize(input);
    const batchSizeMatch = input.match(/(\d+)\s*(g|grams?|oz|ounces?|lbs?|pounds?)\b/i);

    // Try to extract oils if mentioned
    const mentionedOils = extractOilsFromText(input);

    // If user provided both batch size AND oils, try to start calculation directly
    if ((batchSizeResult || batchSizeMatch) && mentionedOils.length > 0) {
        // Use parseBatchSize result if available, otherwise fall back to regex
        const grams = batchSizeResult ? batchSizeResult.grams : null;
        return handleDirectRecipeRequest(batchSizeMatch, mentionedOils, input, grams);
    }

    // If user provided batch size but no oils, ask for oils
    if (batchSizeResult || batchSizeMatch) {
        let grams;
        if (batchSizeResult) {
            grams = batchSizeResult.grams;
        } else {
            const size = parseInt(batchSizeMatch[1]);
            const unit = batchSizeMatch[2].toLowerCase();
            grams = size;
            if (unit.startsWith('oz')) grams = Math.round(size * 28.35);
            else if (unit.startsWith('lb') || unit === 'pounds') grams = Math.round(size * 453.6);
        }

        recipeState.active = true;
        recipeState.batchSizeGrams = grams;
        recipeState.oils = [];
        recipeState.superfat = 5;
        saveRecipeDraft(); // Auto-save progress

        const availableOils = soapCalculator
            ? soapCalculator.getAvailableOils().slice(0, 8).map(o => o.name).join(', ')
            : 'olive oil, coconut oil, palm oil, castor oil, shea butter';

        return {
            response: `Great! I'll help you create a **${grams}g soap recipe**.\n\nNow, what oils would you like to use? Tell me the oils and amounts, for example:\n- "300g olive oil, 150g coconut oil, 50g castor oil"\n- Or just list oils and I'll suggest percentages: "olive oil, coconut oil, shea butter"\n\n**Available oils:** ${availableOils}`,
            category: 'recipe'
        };
    }

    // If user mentioned oils but no batch size, ask for batch size
    if (mentionedOils.length > 0) {
        recipeState.active = true;
        recipeState.batchSizeGrams = 0;
        recipeState.pendingOils = mentionedOils;
        recipeState.superfat = 5;
        saveRecipeDraft(); // Auto-save progress

        return {
            response: `I can create a recipe with **${mentionedOils.join(', ')}**! 🧼\n\nHow much soap do you want to make? (e.g., "500g", "1000 grams", "32 oz")`,
            category: 'recipe'
        };
    }

    // Generic recipe request - start the guided flow
    recipeState.active = true;
    recipeState.batchSizeGrams = 0;
    recipeState.oils = [];
    recipeState.superfat = 5;
    saveRecipeDraft(); // Auto-save progress

    return {
        response: `I'll help you create a safe, calculated soap recipe! 🧮\n\nFirst, how much soap do you want to make?\n- **Small batch:** 500g\n- **Medium batch:** 1000g\n- **Large batch:** 1500g\n\nJust tell me the batch size (e.g., "500g" or "1000 grams")`,
        category: 'recipe'
    };
}

/**
 * Extract oil names from user text
 */
function extractOilsFromText(text) {
    const input = text.toLowerCase();
    const foundOils = [];

    // List of recognizable oil names
    const oilPatterns = [
        { pattern: /\b(olive)\s*(oil)?\b/, name: 'Olive Oil' },
        { pattern: /\b(coconut)\s*(oil)?\b/, name: 'Coconut Oil' },
        { pattern: /\b(palm)\s*(oil)?\b(?!\s*kernel)/, name: 'Palm Oil' },
        { pattern: /\b(castor)\s*(oil)?\b/, name: 'Castor Oil' },
        { pattern: /\b(shea)\s*(butter)?\b/, name: 'Shea Butter' },
        { pattern: /\b(cocoa)\s*(butter)?\b/, name: 'Cocoa Butter' },
        { pattern: /\b(sweet\s*almond|almond)\s*(oil)?\b/, name: 'Sweet Almond Oil' },
        { pattern: /\b(avocado)\s*(oil)?\b/, name: 'Avocado Oil' },
        { pattern: /\b(sunflower)\s*(oil)?\b/, name: 'Sunflower Oil' },
        { pattern: /\b(rice\s*bran)\s*(oil)?\b/, name: 'Rice Bran Oil' },
        { pattern: /\b(grapeseed|grape\s*seed)\s*(oil)?\b/, name: 'Grapeseed Oil' },
        { pattern: /\b(jojoba)\s*(oil)?\b/, name: 'Jojoba Oil' },
        { pattern: /\b(hemp|hemp\s*seed)\s*(oil)?\b/, name: 'Hemp Seed Oil' },
        { pattern: /\b(canola)\s*(oil)?\b/, name: 'Canola Oil' },
        { pattern: /\b(lard)\b/, name: 'Lard' },
        { pattern: /\b(tallow)\b/, name: 'Tallow' },
        { pattern: /\b(mango)\s*(butter)?\b/, name: 'Mango Butter' },
        { pattern: /\b(babassu)\s*(oil)?\b/, name: 'Babassu Oil' },
        { pattern: /\b(apricot)\s*(kernel)?\s*(oil)?\b/, name: 'Apricot Kernel Oil' }
    ];

    for (const { pattern, name } of oilPatterns) {
        if (pattern.test(input) && !foundOils.includes(name)) {
            foundOils.push(name);
        }
    }

    return foundOils;
}

// ===========================================
// IMPROVED PARSING FUNCTIONS
// Handle more natural language input formats
// ===========================================

/**
 * Parse batch size from text with flexible format support
 * Handles: "500g", "500 grams", "1 pound", "16 oz", "about 500g", "half a kilo", etc.
 */
function parseBatchSize(text) {
    const input = text.toLowerCase();

    // Common word-based sizes (check longer phrases first to avoid partial matches)
    const wordSizes = {
        'small batch': 500,
        'medium batch': 1000,
        'large batch': 1500,
        'half a kilo': 500,
        'half kilo': 500,
        'one kilo': 1000,
        '1 kilo': 1000,
        'a kilo': 1000,
        'two kilos': 2000,
        '2 kilos': 2000,
        // Single-word shortcuts
        'small': 500,
        'medium': 1000,
        'large': 1500
    };

    for (const [phrase, grams] of Object.entries(wordSizes)) {
        if (input.includes(phrase)) {
            return { grams, original: phrase };
        }
    }

    // Multiple regex patterns for flexibility
    const patterns = [
        // Standard: "500g", "500 g", "500 grams"
        /(\d+(?:\.\d+)?)\s*(g|grams?)\b/i,
        // Ounces: "16 oz", "16 ounces"
        /(\d+(?:\.\d+)?)\s*(oz|ounces?)\b/i,
        // Pounds: "1 lb", "2 lbs", "1 pound", "2 pounds"
        /(\d+(?:\.\d+)?)\s*(lbs?|pounds?)\b/i,
        // Kilograms: "1 kg", "1.5 kg", "1 kilogram"
        /(\d+(?:\.\d+)?)\s*(kg|kilograms?)\b/i,
        // With "about", "around", "approximately": "about 500g"
        /(?:about|around|approximately|roughly|~)\s*(\d+(?:\.\d+)?)\s*(g|grams?|oz|ounces?|lbs?|pounds?|kg)?\b/i,
        // Standalone number (assume grams if in recipe context)
        /\b(\d{3,4})\b(?!\s*%)/  // 3-4 digit number not followed by %
    ];

    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            const value = parseFloat(match[1]);
            const unit = (match[2] || 'g').toLowerCase();

            let grams = value;

            // Convert to grams
            if (unit.startsWith('oz') || unit === 'ounces') {
                grams = Math.round(value * 28.35);
            } else if (unit.startsWith('lb') || unit === 'pounds' || unit === 'pound') {
                grams = Math.round(value * 453.6);
            } else if (unit === 'kg' || unit.startsWith('kilo')) {
                grams = Math.round(value * 1000);
            }

            // Sanity check - batch size should be reasonable (50g to 10kg)
            if (grams >= 50 && grams <= 10000) {
                return { grams, original: match[0] };
            }
        }
    }

    return null;
}

/**
 * Parse superfat percentage from text with flexible format support
 * Handles: "5% superfat", "superfat 5%", "5 percent superfat", "lye discount 5%", etc.
 */
function parseSuperfat(text) {
    const input = text.toLowerCase();

    // Multiple patterns for flexibility
    const patterns = [
        // "5% superfat" or "5 % superfat"
        /(\d+(?:\.\d+)?)\s*%?\s*(?:superfat|super\s*fat)/i,
        // "superfat 5%" or "superfat: 5%"
        /(?:superfat|super\s*fat)[:\s]+(\d+(?:\.\d+)?)\s*%?/i,
        // "5 percent superfat"
        /(\d+(?:\.\d+)?)\s*percent\s*(?:superfat|super\s*fat)?/i,
        // "lye discount 5%" or "5% lye discount"
        /(\d+(?:\.\d+)?)\s*%?\s*lye\s*discount/i,
        /lye\s*discount[:\s]+(\d+(?:\.\d+)?)\s*%?/i,
        // "sf 5" or "sf: 5"
        /\bsf[:\s]+(\d+(?:\.\d+)?)\s*%?/i,
        // "with 5% extra oil"
        /with\s+(\d+(?:\.\d+)?)\s*%\s*(?:extra|excess)\s*oil/i
    ];

    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            const value = parseFloat(match[1]);
            // Sanity check - superfat should be 0-25%
            if (value >= 0 && value <= 25) {
                return value;
            }
        }
    }

    return null;
}

/**
 * Parse lye concentration or water:lye ratio from text
 * Handles: "33% lye concentration", "2:1 water to lye", etc.
 */
function parseLyeConcentration(text) {
    const input = text.toLowerCase();

    // Lye concentration as percentage
    const concMatch = input.match(/(\d+(?:\.\d+)?)\s*%\s*(?:lye\s*)?(?:concentration|conc)/i);
    if (concMatch) {
        const conc = parseFloat(concMatch[1]);
        if (conc >= 25 && conc <= 50) {
            return { type: 'concentration', value: conc };
        }
    }

    // Water:lye ratio (e.g., "2:1 water to lye" or "water:lye 2.5:1")
    const ratioMatch = input.match(/(\d+(?:\.\d+)?)\s*:\s*1\s*(?:water|w)\s*(?:to|:)\s*(?:lye|l)/i) ||
                       input.match(/(?:water|w)\s*(?:to|:)\s*(?:lye|l)\s*(?:ratio)?\s*(\d+(?:\.\d+)?)\s*:\s*1/i);
    if (ratioMatch) {
        const ratio = parseFloat(ratioMatch[1]);
        if (ratio >= 1 && ratio <= 4) {
            return { type: 'ratio', value: ratio };
        }
    }

    return null;
}

/**
 * Handle direct recipe request with batch size and oils
 */
function handleDirectRecipeRequest(batchSizeMatch, mentionedOils, input, parsedGrams = null) {
    let batchGrams;

    // Use pre-parsed grams if provided (from parseBatchSize for "small", "medium", etc.)
    if (parsedGrams) {
        batchGrams = parsedGrams;
    } else if (batchSizeMatch) {
        const size = parseInt(batchSizeMatch[1]);
        const unit = batchSizeMatch[2].toLowerCase();
        batchGrams = size;
        if (unit.startsWith('oz')) batchGrams = Math.round(size * 28.35);
        else if (unit.startsWith('lb') || unit === 'pounds') batchGrams = Math.round(size * 453.6);
    } else {
        batchGrams = 500; // Default fallback
    }

    // Try to extract amounts for each oil
    const oilsWithAmounts = parseOilAmounts(input, mentionedOils, batchGrams);

    if (oilsWithAmounts.length > 0 && oilsWithAmounts.every(o => o.grams > 0)) {
        // User provided complete recipe - calculate it!
        return calculateAndFormatRecipe(oilsWithAmounts, 5); // Default 5% superfat
    }

    // Oils mentioned but no amounts - suggest a balanced recipe
    return suggestBalancedRecipe(mentionedOils, batchGrams);
}

/**
 * Parse oil amounts from text
 */
function parseOilAmounts(text, oilNames, totalBatchGrams) {
    const input = text.toLowerCase();
    const results = [];

    // Try to find amounts for each mentioned oil
    for (const oilName of oilNames) {
        const oilKey = oilName.toLowerCase().replace(/\s*(oil|butter)\s*/gi, '').trim();
        let found = false;

        // Look for amount patterns near this oil name
        const amountMatch = input.match(new RegExp(`(\\d+)\\s*(g|grams?|oz|ounces?|%)\\s*(?:of\\s+)?${oilKey}|${oilKey}\\s*(?:oil|butter)?\\s*[:\\s]*(\\d+)\\s*(g|grams?|oz|ounces?|%)`, 'i'));

        if (amountMatch) {
            const amount = parseInt(amountMatch[1] || amountMatch[3]);
            const unit = (amountMatch[2] || amountMatch[4] || '').toLowerCase();

            let grams = amount;
            if (unit === '%') {
                grams = Math.round(totalBatchGrams * (amount / 100));
            } else if (unit.startsWith('oz')) {
                grams = Math.round(amount * 28.35);
            }

            if (grams > 0) {
                results.push({ name: oilName, grams });
                found = true;
            }
        }

        // If no amount found, add with 0 grams (will trigger suggestion)
        if (!found) {
            results.push({ name: oilName, grams: 0 });
        }
    }

    return results;
}

/**
 * Suggest a balanced recipe using the mentioned oils
 */
function suggestBalancedRecipe(oilNames, batchGrams) {
    // Create balanced percentages based on oil types
    const percentages = calculateBalancedPercentages(oilNames);

    let response = `Great choice of oils! Here's a balanced **${batchGrams}g** recipe suggestion:\n\n`;
    response += `### Suggested Recipe\n`;

    const oils = [];
    for (const oilName of oilNames) {
        const percent = percentages[oilName] || Math.round(100 / oilNames.length);
        const grams = Math.round(batchGrams * (percent / 100));
        response += `- **${oilName}**: ${grams}g (${percent}%)\n`;
        oils.push({ name: oilName, grams, percent });
    }

    response += `\nShould I calculate this with **5% superfat** (standard)? Or would you like to:\n`;
    response += `- Adjust the percentages\n`;
    response += `- Change the superfat (type "7% superfat")\n`;
    response += `\nJust say **"calculate"** or **"yes"** to proceed!`;

    // Store in recipe state for follow-up
    recipeState.active = true;
    recipeState.batchSizeGrams = batchGrams;
    recipeState.oils = oils;
    recipeState.superfat = 5;

    return { response, category: 'recipe' };
}

/**
 * Calculate balanced percentages for oils based on type
 */
function calculateBalancedPercentages(oilNames) {
    const percentages = {};
    const hardOils = ['Coconut Oil', 'Palm Oil', 'Cocoa Butter', 'Shea Butter', 'Mango Butter', 'Lard', 'Tallow', 'Babassu Oil'];
    const latherBoosters = ['Castor Oil'];

    let hardOilsInRecipe = oilNames.filter(o => hardOils.includes(o));
    let softOilsInRecipe = oilNames.filter(o => !hardOils.includes(o) && !latherBoosters.includes(o));
    let castorInRecipe = oilNames.filter(o => latherBoosters.includes(o));

    // Aim for: ~30-40% hard oils, 5-10% castor, rest soft oils
    let hardPercent = hardOilsInRecipe.length > 0 ? 35 : 0;
    let castorPercent = castorInRecipe.length > 0 ? 8 : 0;
    let softPercent = 100 - hardPercent - castorPercent;

    // Distribute within categories
    hardOilsInRecipe.forEach(oil => {
        percentages[oil] = Math.round(hardPercent / hardOilsInRecipe.length);
    });

    castorInRecipe.forEach(oil => {
        percentages[oil] = castorPercent;
    });

    softOilsInRecipe.forEach(oil => {
        percentages[oil] = Math.round(softPercent / softOilsInRecipe.length);
    });

    // Ensure percentages sum to 100
    const total = Object.values(percentages).reduce((a, b) => a + b, 0);
    if (total !== 100 && oilNames.length > 0) {
        const diff = 100 - total;
        percentages[oilNames[0]] += diff;
    }

    return percentages;
}

/**
 * Calculate recipe and format for display
 */
function calculateAndFormatRecipe(oils, superfatPercent) {
    try {
        const recipe = calculateRecipeWithCalculator(oils, superfatPercent);

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

        const recipeOutput = formatCalculatedRecipe(recipe);

        // Reset recipe state and clear draft
        recipeState = {
            active: false,
            batchSizeGrams: 0,
            oils: [],
            superfat: 5
        };
        clearRecipeDraft(); // Recipe complete - clear draft

        return { response: validationMessage + recipeOutput, category: 'recipe' };
    } catch (error) {
        console.error('Recipe calculation error:', error);
        return {
            response: `Sorry, I encountered an error calculating your recipe: ${error.message}. Please try again or ask for help.`,
            category: 'recipe'
        };
    }
}

/**
 * Handle conversation when recipe building is active
 */
function handleActiveRecipeConversation(input) {
    // Check for cancel/abort/start over
    if (/\b(cancel|stop|abort|nevermind|never mind|start over|restart)\b/i.test(input)) {
        const wasInProgress = recipeState.batchSizeGrams > 0 || recipeState.oils.length > 0;
        recipeState = { active: false, batchSizeGrams: 0, oils: [], superfat: 5 };
        clearRecipeDraft(); // Clear saved draft

        if (/\b(start over|restart)\b/i.test(input)) {
            // User wants to start a new recipe
            recipeState.active = true;
            return {
                response: "Starting fresh! 🧼\n\nHow much soap do you want to make?\n- **500g** (small, ~6 bars)\n- **1000g** (medium, ~12 bars)\n- **1500g** (large, ~18 bars)",
                category: 'recipe'
            };
        }

        return {
            response: wasInProgress
                ? "Recipe cancelled. No worries! What else can I help you with?"
                : "No problem! What else can I help you with?",
            category: 'recipe'
        };
    }

    // Check for "suggest a recipe" during recipe building
    if (/\b(suggest|recommend|beginner|default)\b.*\b(recipe|blend|mix)\b/i.test(input) ||
        /\b(suggest|recommend)\b/i.test(input) && recipeState.oils.length === 0) {

        const batchSize = recipeState.batchSizeGrams || 500;
        const beginnerOils = [
            { name: 'Olive Oil', grams: Math.round(batchSize * 0.35) },
            { name: 'Coconut Oil', grams: Math.round(batchSize * 0.30) },
            { name: 'Shea Butter', grams: Math.round(batchSize * 0.25) },
            { name: 'Castor Oil', grams: Math.round(batchSize * 0.10) }
        ];

        recipeState.batchSizeGrams = batchSize;
        recipeState.oils = beginnerOils;
        recipeState.superfat = 5;
        saveRecipeDraft();

        return {
            response: `Here's a great **beginner-friendly recipe** for a ${batchSize}g batch:\n\n` +
                `- **Olive Oil:** ${beginnerOils[0].grams}g (35%) - conditioning\n` +
                `- **Coconut Oil:** ${beginnerOils[1].grams}g (30%) - cleansing & lather\n` +
                `- **Shea Butter:** ${beginnerOils[2].grams}g (25%) - hardness & luxury\n` +
                `- **Castor Oil:** ${beginnerOils[3].grams}g (10%) - lather boost\n\n` +
                `Ready to calculate with **5% superfat**? Say "**yes**" or "**calculate**"!\n\n` +
                `Or say "**change**" to adjust oils, or "**7% superfat**" for a different superfat.`,
            category: 'recipe'
        };
    }

    // Check for batch size if we don't have one yet - use improved parser
    if (recipeState.batchSizeGrams === 0) {
        const batchResult = parseBatchSize(input);
        if (batchResult) {
            const grams = batchResult.grams;
            recipeState.batchSizeGrams = grams;
            saveRecipeDraft(); // Auto-save progress

            // Check if we had pending oils from earlier
            if (recipeState.pendingOils && recipeState.pendingOils.length > 0) {
                return suggestBalancedRecipe(recipeState.pendingOils, grams);
            }

            const availableOils = soapCalculator
                ? soapCalculator.getAvailableOils().slice(0, 8).map(o => o.name).join(', ')
                : 'olive oil, coconut oil, palm oil, castor oil, shea butter';

            return {
                response: `Perfect! Making a **${grams}g** batch.\n\nNow, what oils do you want to use? You can:\n- List oils with amounts: "300g olive oil, 150g coconut oil, 50g castor oil"\n- Or just list oils: "olive oil, coconut oil, shea butter"\n\n**Available oils:** ${availableOils}`,
                category: 'recipe'
            };
        }
    }

    // Check for oil inputs
    if (recipeState.batchSizeGrams > 0 && recipeState.oils.length === 0) {
        // Try to extract oils and amounts
        const mentionedOils = extractOilsFromText(input);

        if (mentionedOils.length > 0) {
            const oilsWithAmounts = parseOilAmounts(input, mentionedOils, recipeState.batchSizeGrams);

            // Check if all oils have amounts
            if (oilsWithAmounts.every(o => o.grams > 0)) {
                const totalGrams = oilsWithAmounts.reduce((sum, o) => sum + o.grams, 0);
                recipeState.oils = oilsWithAmounts;
                saveRecipeDraft(); // Auto-save progress

                if (Math.abs(totalGrams - recipeState.batchSizeGrams) > 10) {
                    return {
                        response: `Your oils total **${totalGrams}g** (batch size was ${recipeState.batchSizeGrams}g). That's fine - I'll calculate based on ${totalGrams}g.\n\nReady to calculate with **5% superfat**? Just say "yes" or "calculate", or specify a different superfat like "7% superfat".`,
                        category: 'recipe'
                    };
                }

                return {
                    response: `Got it! Your oils total **${totalGrams}g**.\n\nReady to calculate with **5% superfat**? Just say "yes" or "calculate", or specify a different superfat like "7% superfat".`,
                    category: 'recipe'
                };
            } else {
                // Have oils but no amounts - suggest balanced recipe
                return suggestBalancedRecipe(mentionedOils, recipeState.batchSizeGrams);
            }
        }
    }

    // Check for superfat adjustment or final calculation - use improved parser
    if (recipeState.oils.length > 0) {
        const superfatValue = parseSuperfat(input);
        if (superfatValue !== null) {
            recipeState.superfat = superfatValue;
        }

        if (/\b(calculate|yes|go|do it|proceed|ok|okay|sure|yep|yeah)\b/i.test(input) || superfatValue !== null) {
            return calculateAndFormatRecipe(recipeState.oils, recipeState.superfat);
        }
    }

    // Didn't understand - provide context-aware guidance
    if (recipeState.batchSizeGrams === 0) {
        // Check if they typed something that looks like an attempt at batch size
        const hasNumber = /\d/.test(input);
        const hasUnit = /\b(gram|ounce|pound|kilo|batch)\b/i.test(input);

        if (hasNumber && !hasUnit) {
            return {
                response: `I see a number but need a unit! Try:\n- "**500g**" or "**500 grams**"\n- "**16 oz**" or "**1 pound**"\n- "**small batch**" (500g), "**medium batch**" (1000g)\n\nOr say "**cancel**" to start over.`,
                category: 'recipe'
            };
        }

        return {
            response: `I need to know the batch size first. How much soap do you want to make?\n\n**Quick options:**\n- **500g** (small, ~6 bars)\n- **1000g** (medium, ~12 bars)\n- **1500g** (large, ~18 bars)\n\nOr type your own size like "32 oz" or "2 pounds".\n\n💡 *Say "cancel" anytime to exit recipe mode.*`,
            category: 'recipe'
        };
    }

    if (recipeState.oils.length === 0) {
        // Check if they mentioned something that isn't an oil
        const mentionedOils = extractOilsFromText(input);
        const hasPercentOrGrams = /\d+\s*(%|g|grams?)/i.test(input);

        if (mentionedOils.length === 0 && hasPercentOrGrams) {
            return {
                response: `I didn't recognize those oils. Here are oils I can work with:\n\n**Popular choices:** Olive Oil, Coconut Oil, Palm Oil, Castor Oil, Shea Butter\n\n**Example:** "35% olive oil, 30% coconut, 25% shea butter, 10% castor"\n\n💡 *Say "cancel" to exit, or "suggest a recipe" for a beginner blend!*`,
                category: 'recipe'
            };
        }

        const availableOils = soapCalculator
            ? soapCalculator.getAvailableOils().slice(0, 10).map(o => o.name).join(', ')
            : 'Olive Oil, Coconut Oil, Palm Oil, Castor Oil, Shea Butter, Cocoa Butter, Sweet Almond Oil, Avocado Oil';

        return {
            response: `What oils would you like to use for your **${recipeState.batchSizeGrams}g** batch?\n\n**Options:**\n- List oils with amounts: "200g olive, 150g coconut, 50g castor"\n- List oils with percentages: "40% olive, 35% coconut, 25% shea"\n- Just name oils: "olive, coconut, castor" (I'll suggest amounts)\n- Say "**suggest a recipe**" for a beginner-friendly blend\n\n**Available oils:** ${availableOils}\n\n💡 *Say "cancel" to exit recipe mode.*`,
            category: 'recipe'
        };
    }

    // We have oils but user said something unexpected
    const currentOilsList = recipeState.oils.map(o => `${o.name}: ${o.grams}g`).join(', ');
    return {
        response: `Almost there! Your current recipe:\n- **Oils:** ${currentOilsList}\n- **Superfat:** ${recipeState.superfat}%\n\n**What would you like to do?**\n- Say "**yes**" or "**calculate**" to get your recipe\n- Say "**7% superfat**" to change the superfat\n- Say "**add castor oil**" to add another oil\n- Say "**start over**" or "**cancel**" to begin again`,
        category: 'recipe'
    };
}

// Find matching response based on keywords (legacy - for non-recipe commands)
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

    // NOTE: Recipe calculation is now handled by checkRecipeIntent() which runs BEFORE this function
    // This function now only handles save/view commands and legacy knowledge lookup

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

// Direct API request - no rate limiting or artificial delays
async function makeDirectRequest(userMessage) {
    try {
        // Make the API call directly without any delays
        const result = await getLLMResponse(userMessage);
        return result;
    } catch (error) {
        // Simply throw the error - no retry logic
        throw error;
    }
}

// ===========================================
// AI-TO-CALCULATOR BRIDGE
// Post-processes AI responses to detect recipe suggestions
// and run them through the actual calculator for safety
// ===========================================

/**
 * Process AI response to detect and calculate recipe suggestions
 * If AI suggests a recipe with oils/percentages, calculate actual lye amounts
 */
async function processAIResponseForRecipe(aiResponse, originalQuestion) {
    // Check if the response looks like it contains a recipe suggestion
    const hasRecipeIndicators = (
        /\b(olive|coconut|palm|shea|castor|avocado)\s*(oil|butter)?.*\d+\s*%/i.test(aiResponse) ||
        /\d+\s*%.*\b(olive|coconut|palm|shea|castor|avocado)\b/i.test(aiResponse) ||
        /\d+\s*(g|grams?).*\b(olive|coconut|palm|shea|castor)\b/i.test(aiResponse)
    );

    // Check if user was asking for a recipe
    const userWantsRecipe = (
        /\b(recipe|calculate|make|create|suggest)\b.*\bsoap\b/i.test(originalQuestion) ||
        /\bsoap\b.*\b(recipe|calculate)\b/i.test(originalQuestion) ||
        /\b(recommend|suggestion|beginner)\b.*\b(recipe|soap)\b/i.test(originalQuestion)
    );

    if (!hasRecipeIndicators || !userWantsRecipe) {
        return aiResponse; // Return as-is if not a recipe
    }

    console.log('🔄 AI response contains recipe - attempting to calculate...');

    // Try to extract oils and percentages from the AI response
    const extractedRecipe = extractRecipeFromAIResponse(aiResponse);

    if (!extractedRecipe || extractedRecipe.oils.length === 0) {
        console.log('⚠️ Could not extract recipe from AI response');
        // Provide helpful guidance to get a calculated recipe
        return aiResponse + `\n\n---\n\n> **💡 Want exact lye amounts?** I mentioned some oils above but couldn't auto-calculate. To get a safe, calculated recipe with precise lye and water amounts, try saying:\n> - "Calculate a 500g batch with those oils"\n> - "Create a recipe with 35% olive, 30% coconut, 25% palm, 10% castor"\n> - Or just say "calculate recipe" and I'll guide you step by step!`;
    }

    // Try to extract or default batch size
    let batchSize = extractedRecipe.batchSize || 500; // Default 500g if not specified

    // Check if user mentioned a batch size
    const userBatchMatch = originalQuestion.match(/(\d+)\s*(g|grams?|oz|ounces?)/i);
    if (userBatchMatch) {
        batchSize = parseInt(userBatchMatch[1]);
        const unit = userBatchMatch[2].toLowerCase();
        if (unit.startsWith('oz')) batchSize = Math.round(batchSize * 28.35);
    }

    try {
        // Convert percentages to grams
        const oilsWithGrams = extractedRecipe.oils.map(oil => ({
            name: oil.name,
            grams: Math.round(batchSize * (oil.percent / 100))
        }));

        // Calculate actual recipe with lye amounts
        const calculatedRecipe = calculateRecipeWithCalculator(oilsWithGrams, extractedRecipe.superfat || 5);

        // Validate for safety
        let validationMessage = '';
        if (recipeValidator) {
            const validation = recipeValidator.validateRecipe(calculatedRecipe);
            if (validation.warnings.length > 0 || !validation.valid) {
                validationMessage = recipeValidator.formatValidationMessage(validation);
                validationMessage += '\n---\n\n';
            }
        }

        // Format the calculated recipe
        const recipeOutput = formatCalculatedRecipe(calculatedRecipe);

        console.log('✅ Successfully calculated recipe from AI suggestion');

        // Return AI explanation + calculated recipe
        return `${aiResponse}\n\n---\n\n## 🧮 Calculated Recipe (${batchSize}g batch)\n\n${validationMessage}${recipeOutput}`;

    } catch (error) {
        console.error('❌ Failed to calculate recipe from AI response:', error);
        // Provide more helpful guidance when calculation fails
        const oilNames = extractedRecipe.oils.map(o => o.name).join(', ');
        return aiResponse + `\n\n---\n\n> **⚠️ Calculation Error:** I couldn't auto-calculate this recipe (${error.message}).\n>\n> **To get safe lye amounts**, say one of these:\n> - "Calculate a ${batchSize}g batch with ${oilNames}"\n> - "Calculate recipe" (I'll guide you step by step)\n>\n> **Never guess lye amounts** - incorrect calculations can make caustic, dangerous soap!`;
    }
}

/**
 * Extract oil names and percentages from AI response text
 */
function extractRecipeFromAIResponse(text) {
    let oils = [];
    let superfat = 5; // Default
    let batchSize = null;

    // Oil name mapping to standardized names
    const oilNameMap = {
        'olive': 'Olive Oil',
        'coconut': 'Coconut Oil',
        'palm': 'Palm Oil',
        'castor': 'Castor Oil',
        'shea': 'Shea Butter',
        'cocoa': 'Cocoa Butter',
        'sweet almond': 'Sweet Almond Oil',
        'almond': 'Sweet Almond Oil',
        'avocado': 'Avocado Oil',
        'sunflower': 'Sunflower Oil',
        'rice bran': 'Rice Bran Oil',
        'grapeseed': 'Grapeseed Oil',
        'jojoba': 'Jojoba Oil',
        'hemp': 'Hemp Seed Oil',
        'canola': 'Canola Oil',
        'lard': 'Lard',
        'tallow': 'Tallow',
        'mango': 'Mango Butter',
        'babassu': 'Babassu Oil',
        'apricot': 'Apricot Kernel Oil'
    };

    // FIRST: Try to parse structured format (from AI prompt instructions)
    // Format: RECIPE: name, BATCH SIZE: Xg, OILS: - Oil Name: X%, SUPERFAT: X%
    const structuredMatch = text.match(/RECIPE:.*?BATCH SIZE:\s*(\d+)\s*g.*?OILS:([\s\S]*?)SUPERFAT:\s*(\d+)\s*%/i);
    if (structuredMatch) {
        console.log('📋 Found structured recipe format from AI');
        batchSize = parseInt(structuredMatch[1]);
        superfat = parseInt(structuredMatch[3]);

        // Parse oils from the OILS section
        const oilsSection = structuredMatch[2];
        const oilLinePattern = /-\s*([^:]+):\s*(\d+)\s*%/g;
        let oilMatch;
        while ((oilMatch = oilLinePattern.exec(oilsSection)) !== null) {
            const oilName = oilMatch[1].trim();
            const percent = parseInt(oilMatch[2]);

            // Try to match to standardized name
            let standardName = null;
            for (const [key, name] of Object.entries(oilNameMap)) {
                if (oilName.toLowerCase().includes(key)) {
                    standardName = name;
                    break;
                }
            }

            if (standardName && !oils.find(o => o.name === standardName)) {
                oils.push({ name: standardName, percent });
            } else if (!standardName) {
                // Use the name as-is if not in our map (might still work)
                oils.push({ name: oilName, percent });
            }
        }

        if (oils.length > 0) {
            console.log(`✅ Extracted ${oils.length} oils from structured format`);
            return { oils, superfat, batchSize };
        }
    }

    // FALLBACK: Try legacy patterns for unstructured AI responses
    // Try first pattern: "Oil Name: XX%"
    let match;
    const pattern1 = /(\w+(?:\s+\w+)?)\s*(?:oil|butter)?[\s:–-]+(\d+)\s*%/gi;
    while ((match = pattern1.exec(text)) !== null) {
        const oilKey = match[1].toLowerCase().trim();
        const percent = parseInt(match[2]);

        for (const [key, standardName] of Object.entries(oilNameMap)) {
            if (oilKey.includes(key) && !oils.find(o => o.name === standardName)) {
                oils.push({ name: standardName, percent });
                break;
            }
        }
    }

    // Try second pattern: "XX% Oil Name"
    if (oils.length === 0) {
        const pattern2 = /(\d+)\s*%\s*(\w+(?:\s+\w+)?)\s*(?:oil|butter)?/gi;
        while ((match = pattern2.exec(text)) !== null) {
            const percent = parseInt(match[1]);
            const oilKey = match[2].toLowerCase().trim();

            for (const [key, standardName] of Object.entries(oilNameMap)) {
                if (oilKey.includes(key) && !oils.find(o => o.name === standardName)) {
                    oils.push({ name: standardName, percent });
                    break;
                }
            }
        }
    }

    // Extract superfat if mentioned
    const superfatMatch = text.match(/(\d+)\s*%?\s*superfat/i);
    if (superfatMatch) {
        superfat = parseInt(superfatMatch[1]);
    }

    // Extract batch size if mentioned
    const batchMatch = text.match(/(\d+)\s*(g|grams?)\s*(batch|total)?/i);
    if (batchMatch) {
        batchSize = parseInt(batchMatch[1]);
    }

    // Validate total percentage is reasonable (within 5% of 100)
    const totalPercent = oils.reduce((sum, o) => sum + o.percent, 0);
    if (totalPercent < 95 || totalPercent > 105) {
        console.log(`⚠️ Extracted percentages total ${totalPercent}% - may be incomplete`);
    }

    return {
        oils,
        superfat,
        batchSize
    };
}

// Handle sending messages
// STRATEGY: Check recipe intent FIRST, then knowledge bank, then AI API
// Recipe calculations should always take priority to ensure safety
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
    showTypingIndicator('Processing your request');

    let messageDisplayed = false;

    try {
        // STEP 0: Check for follow-up questions that reference previous context
        const followUpResult = checkFollowUpQuestion(userMessage.toLowerCase());
        if (followUpResult) {
            console.log('✅ Follow-up question detected - using context');
            lastResponseSource = 'context';

            removeTypingIndicator();
            addMessage(followUpResult.response, true, followUpResult.category);
            messageDisplayed = true;

            // Update context tracking
            updateConversationContext(followUpResult.category, userMessage);

            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: followUpResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - follow-up handled!
        }

        // STEP 1: Check for recipe intent FIRST (highest priority for safety!)
        // This must happen BEFORE knowledge bank to prevent oil names triggering info lookups
        const recipeResult = checkRecipeIntent(userMessage);

        if (recipeResult) {
            console.log('✅ Recipe intent detected - using calculator');
            lastResponseSource = 'calculator';

            removeTypingIndicator();
            addMessage(recipeResult.response, true, recipeResult.category);
            messageDisplayed = true;

            // Update context tracking
            updateConversationContext(recipeResult.category, userMessage);

            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: recipeResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - recipe handled locally!
        }

        // STEP 2: Check local knowledge bank (saves API tokens!)
        updateTypingIndicator('Searching knowledge base');
        const knowledgeBankResult = searchKnowledgeBank(userMessage);

        if (knowledgeBankResult) {
            // Found answer in knowledge bank - no API call needed!
            console.log('✅ Answer found in knowledge bank - saving API tokens!');
            lastResponseSource = 'knowledge_bank';

            removeTypingIndicator();

            // Add response with category badge
            addMessage(knowledgeBankResult.response, true, knowledgeBankResult.category);
            messageDisplayed = true;

            // Update context tracking
            updateConversationContext(knowledgeBankResult.category, userMessage);

            // Update conversation history for context
            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: knowledgeBankResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - no API call needed!
        }

        // STEP 3: Check legacy findResponse for special commands (save recipe, view recipes, etc.)
        const legacyResult = findResponse(userMessage);

        // If legacy findResponse found something other than the default "That's a great question" response
        if (legacyResult.category !== null) {
            console.log('✅ Answer found in legacy knowledge - saving API tokens!');
            lastResponseSource = 'local';

            removeTypingIndicator();
            addMessage(legacyResult.response, true, legacyResult.category);
            messageDisplayed = true;

            // Update context tracking
            updateConversationContext(legacyResult.category, userMessage);

            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: legacyResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - no API call needed!
        }

        // STEP 4: No local match found - call AI API as fallback
        console.log('🤖 No local match found - calling AI API...');
        updateTypingIndicator('AI is thinking');
        lastResponseSource = 'api';

        const aiResult = await makeDirectRequest(userMessage);

        removeTypingIndicator();

        // POST-PROCESS: Check if AI response contains a recipe suggestion that should be calculated
        const processedResponse = await processAIResponseForRecipe(aiResult.response, userMessage);

        // Add response (markdown will be rendered in addMessage)
        addMessage(processedResponse, true);
        messageDisplayed = true;

        // Update conversation history (keep last 10 messages for context)
        conversationHistory.push({ role: 'user', content: userMessage });
        conversationHistory.push({ role: 'assistant', content: aiResult.response });

        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }
    } catch (error) {
        console.error('❌ Error in sendMessage:', error);
        console.error('📋 Error details:', {
            message: error.message,
            stack: error.stack,
            messageDisplayed: messageDisplayed,
            aiConfigExists: !!aiConfig,
            apiKey: aiConfig ? (aiConfig.getApiKey() ? 'Present' : 'Missing') : 'No config'
        });
        removeTypingIndicator();

        // If the message was already displayed successfully, don't show error to user
        if (messageDisplayed) {
            console.error('Error occurred after message was displayed - not showing error to user');
            return; // Exit early - message was already shown successfully
        }

        // Generate user-friendly error message
        const errorResponse = generateUserFriendlyError(error, userMessage);

        // Show error with fallback response
        addMessage(errorResponse, true, 'error');
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

        // No rate limiting state to reset

        // Clear chat messages (except initial bot message)
        const chatMessages = document.getElementById('chatMessages');
        const firstMessage = chatMessages.querySelector('.bot-message');
        chatMessages.innerHTML = '';
        if (firstMessage) {
            chatMessages.appendChild(firstMessage.cloneNode(true));
        }

        showCopyFeedback('Conversation reset! ✓');
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

    // Check if Knowledge Bank is loaded (for token savings)
    if (typeof SOAP_KNOWLEDGE_BANK !== 'undefined') {
        knowledgeBank = SOAP_KNOWLEDGE_BANK;
        const sectionCount = Object.keys(knowledgeBank).length;
        console.log(`✅ Soap Knowledge Bank loaded (${sectionCount} sections) - will save API tokens!`);
    } else {
        console.warn('⚠️ Soap Knowledge Bank not loaded - all queries will use API tokens');
    }

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
            statusBadge.textContent = '🤖 AI';
            statusBadge.className = 'ai-status-badge';
        }
    }
}
