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
        keywords: ['recipe', 'formula', 'how to make', 'lavender soap', 'oatmeal'],
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

    // Search for matching keywords
    for (const mapping of searchMappings) {
        const matchedKeyword = mapping.keywords.find(kw => input.includes(kw.toLowerCase()));
        if (matchedKeyword) {
            console.log(`📚 Knowledge bank match: "${matchedKeyword}" -> ${mapping.section}`);

            // Navigate to the section in the knowledge bank
            const data = getNestedProperty(kb, mapping.section);
            if (data) {
                const response = mapping.format(data, input);
                if (response) {
                    return { response, category: mapping.category, source: 'knowledge_bank' };
                }
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

// Handle sending messages
// STRATEGY: Check local knowledge bank FIRST to save API tokens
// Only call AI API if knowledge bank doesn't have a good answer
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
    showTypingIndicator('Searching knowledge base');

    let messageDisplayed = false;

    try {
        // STEP 1: Check local knowledge bank FIRST (saves API tokens!)
        const knowledgeBankResult = searchKnowledgeBank(userMessage);

        if (knowledgeBankResult) {
            // Found answer in knowledge bank - no API call needed!
            console.log('✅ Answer found in knowledge bank - saving API tokens!');
            lastResponseSource = 'knowledge_bank';

            removeTypingIndicator();

            // Add response with category badge
            addMessage(knowledgeBankResult.response, true, knowledgeBankResult.category);
            messageDisplayed = true;

            // Update conversation history for context
            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: knowledgeBankResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - no API call needed!
        }

        // STEP 2: Check legacy findResponse for recipe calculations and special commands
        const legacyResult = findResponse(userMessage);

        // If legacy findResponse found something other than the default "That's a great question" response
        if (legacyResult.category !== null) {
            console.log('✅ Answer found in legacy knowledge - saving API tokens!');
            lastResponseSource = 'local';

            removeTypingIndicator();
            addMessage(legacyResult.response, true, legacyResult.category);
            messageDisplayed = true;

            conversationHistory.push({ role: 'user', content: userMessage });
            conversationHistory.push({ role: 'assistant', content: legacyResult.response });

            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }

            return; // Done - no API call needed!
        }

        // STEP 3: No local match found - call AI API as fallback
        console.log('🤖 No local match found - calling AI API...');
        updateTypingIndicator('AI is thinking');
        lastResponseSource = 'api';

        const aiResult = await makeDirectRequest(userMessage);

        removeTypingIndicator();

        // Add response (markdown will be rendered in addMessage)
        addMessage(aiResult.response, true);
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

        // Show error to user temporarily for debugging
        console.warn('⚠️ Falling back to local knowledge base due to API error:', error.message);

        // Fallback to legacy local knowledge base - no error message shown
        const result = findResponse(userMessage);

        // Just show the fallback response directly without error message
        addMessage(result.response, true, result.category);
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
            statusBadge.textContent = '🤖 AI: Offline (Local Mode)';
            statusBadge.className = 'ai-status-badge';
        }
    }
}
