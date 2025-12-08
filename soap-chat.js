// Saponify AI - Soap Making Chat Assistant

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

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const suggestionButtons = document.querySelectorAll('.suggestion-btn');

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
        contentDiv.innerHTML = `${icon}<div class="bot-text">${categoryBadge}${message}</div>`;
    } else {
        contentDiv.innerHTML = message;
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
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <span class="bot-icon">🧼</span>
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

// Find matching response based on keywords
function findResponse(userInput) {
    const input = userInput.toLowerCase();

    // Check each knowledge category
    for (const [category, data] of Object.entries(soapKnowledge)) {
        if (data.keywords.some(keyword => input.includes(keyword.toLowerCase()))) {
            return { response: data.response, category: category };
        }
    }

    // Default response if no match found
    return {
        response: "That's a great question about soap making! While I have information about saponification, oils, essential oils, cold/hot process, troubleshooting, recipes, curing, and colorants, I didn't quite understand your specific question. Could you try asking in a different way? For example: 'What is saponification?' or 'How do I make my first batch?'",
        category: null
    };
}

// Handle sending messages
function sendMessage() {
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

    setTimeout(() => {
        removeTypingIndicator();
        const result = findResponse(userMessage);
        addMessage(result.response, true, result.category);

        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }, 800);
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
