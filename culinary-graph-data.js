
// Culinary Knowledge Graph Data
// Represents ingredients, their relationships, and substitution rules

const culinaryGraphData = {
    // Nodes represent ingredients
    nodes: [
        // Dairy Products
        { id: 'butter', label: 'Butter', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'milk', label: 'Milk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cream', label: 'Heavy Cream', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'yogurt', label: 'Yogurt', category: 'dairy', cuisine: ['mediterranean', 'asian'], dietary: ['vegetarian'] },
        { id: 'cheese-parmesan', label: 'Parmesan', category: 'dairy', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'cheese-mozzarella', label: 'Mozzarella', category: 'dairy', cuisine: ['italian'], dietary: ['vegetarian'] },

        // Dairy Alternatives
        { id: 'coconut-milk', label: 'Coconut Milk', category: 'dairy-alt', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'almond-milk', label: 'Almond Milk', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'olive-oil', label: 'Olive Oil', category: 'oil', cuisine: ['mediterranean', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-oil', label: 'Coconut Oil', category: 'oil', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // Proteins
        { id: 'eggs', label: 'Eggs', category: 'protein', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'chicken', label: 'Chicken', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'tofu', label: 'Tofu', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chickpeas', label: 'Chickpeas', category: 'protein', cuisine: ['mediterranean', 'mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'black-beans', label: 'Black Beans', category: 'protein', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },

        // Flours & Grains
        { id: 'flour', label: 'All-Purpose Flour', category: 'grain', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'almond-flour', label: 'Almond Flour', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'rice-flour', label: 'Rice Flour', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'rice', label: 'White Rice', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'quinoa', label: 'Quinoa', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },

        // Sweeteners
        { id: 'sugar', label: 'White Sugar', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'honey', label: 'Honey', category: 'sweetener', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'maple-syrup', label: 'Maple Syrup', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'agave', label: 'Agave Nectar', category: 'sweetener', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },

        // Aromatics & Herbs
        { id: 'garlic', label: 'Garlic', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'onion', label: 'Onion', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'ginger', label: 'Ginger', category: 'aromatic', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'basil', label: 'Basil', category: 'herb', cuisine: ['italian', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cilantro', label: 'Cilantro', category: 'herb', cuisine: ['mexican', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'oregano', label: 'Oregano', category: 'herb', cuisine: ['italian', 'mediterranean'], dietary: ['vegan', 'vegetarian'] },

        // Sauces & Condiments
        { id: 'soy-sauce', label: 'Soy Sauce', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tomato-sauce', label: 'Tomato Sauce', category: 'sauce', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'salsa', label: 'Salsa', category: 'sauce', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },

        // Vegetables
        { id: 'tomato', label: 'Tomato', category: 'vegetable', cuisine: ['italian', 'mexican', 'mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bell-pepper', label: 'Bell Pepper', category: 'vegetable', cuisine: ['mexican', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'mushrooms', label: 'Mushrooms', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'spinach', label: 'Spinach', category: 'vegetable', cuisine: ['mediterranean', 'italian'], dietary: ['vegan', 'vegetarian'] },
    ],

    // Edges represent relationships between ingredients
    edges: [
        // Substitution relationships (can replace)
        { source: 'butter', target: 'olive-oil', type: 'substitutes', ratio: '1:0.75', context: 'baking, cooking' },
        { source: 'butter', target: 'coconut-oil', type: 'substitutes', ratio: '1:1', context: 'baking' },
        { source: 'milk', target: 'almond-milk', type: 'substitutes', ratio: '1:1', context: 'all' },
        { source: 'milk', target: 'coconut-milk', type: 'substitutes', ratio: '1:1', context: 'curries, baking' },
        { source: 'eggs', target: 'tofu', type: 'substitutes', ratio: '1 egg:1/4 cup', context: 'scrambles, baking' },
        { source: 'eggs', target: 'chickpeas', type: 'substitutes', ratio: '1 egg:3 tbsp aquafaba', context: 'baking' },
        { source: 'flour', target: 'almond-flour', type: 'substitutes', ratio: '1:1', context: 'gluten-free baking' },
        { source: 'flour', target: 'rice-flour', type: 'substitutes', ratio: '1:0.75', context: 'gluten-free' },
        { source: 'sugar', target: 'honey', type: 'substitutes', ratio: '1:0.75', context: 'baking' },
        { source: 'sugar', target: 'maple-syrup', type: 'substitutes', ratio: '1:0.75', context: 'baking' },
        { source: 'sugar', target: 'agave', type: 'substitutes', ratio: '1:0.75', context: 'beverages, baking' },
        { source: 'chicken', target: 'tofu', type: 'substitutes', ratio: '1:1', context: 'stir-fry, curry' },
        { source: 'chicken', target: 'mushrooms', type: 'substitutes', ratio: '1:1.5', context: 'umami flavor' },
        { source: 'cream', target: 'coconut-milk', type: 'substitutes', ratio: '1:1', context: 'dairy-free sauces' },
        { source: 'cheese-parmesan', target: 'nutritional-yeast', type: 'substitutes', ratio: '1:1', context: 'vegan alternative' },

        // Pairing relationships (goes well with)
        { source: 'garlic', target: 'olive-oil', type: 'pairs-with', strength: 'strong' },
        { source: 'garlic', target: 'butter', type: 'pairs-with', strength: 'strong' },
        { source: 'tomato', target: 'basil', type: 'pairs-with', strength: 'strong' },
        { source: 'tomato', target: 'oregano', type: 'pairs-with', strength: 'strong' },
        { source: 'ginger', target: 'soy-sauce', type: 'pairs-with', strength: 'strong' },
        { source: 'ginger', target: 'garlic', type: 'pairs-with', strength: 'medium' },
        { source: 'cilantro', target: 'salsa', type: 'pairs-with', strength: 'strong' },
        { source: 'onion', target: 'garlic', type: 'pairs-with', strength: 'strong' },
        { source: 'cheese-mozzarella', target: 'tomato', type: 'pairs-with', strength: 'strong' },
        { source: 'basil', target: 'olive-oil', type: 'pairs-with', strength: 'strong' },

        // Recipe component relationships (used in)
        { source: 'butter', target: 'flour', type: 'used-with', context: 'baking' },
        { source: 'eggs', target: 'flour', type: 'used-with', context: 'baking' },
        { source: 'milk', target: 'eggs', type: 'used-with', context: 'custards, baking' },
        { source: 'rice', target: 'soy-sauce', type: 'used-with', context: 'fried rice' },
        { source: 'chickpeas', target: 'tahini', type: 'used-with', context: 'hummus' },
        { source: 'black-beans', target: 'salsa', type: 'used-with', context: 'mexican dishes' },

        // Similar flavor profile
        { source: 'oregano', target: 'basil', type: 'similar-flavor', note: 'italian herbs' },
        { source: 'coconut-milk', target: 'cream', type: 'similar-texture', note: 'creamy consistency' },
        { source: 'tofu', target: 'chicken', type: 'similar-texture', note: 'absorbs flavors' },
        { source: 'almond-milk', target: 'milk', type: 'similar-use', note: 'plant-based alternative' },
    ]
};

// Substitution rules with detailed information
const substitutionRules = {
    'butter': [
        {
            substitute: 'olive-oil',
            ratio: '1 cup butter = 3/4 cup olive oil',
            notes: 'Best for savory dishes. Reduce other liquids slightly in baking.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'coconut-oil',
            ratio: '1 cup butter = 1 cup coconut oil',
            notes: 'Works 1:1 in most recipes. Use refined for neutral flavor.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'applesauce',
            ratio: '1 cup butter = 1/2 cup applesauce',
            notes: 'Best for baking. Reduces fat content significantly.',
            confidence: 'medium',
            dietary: ['vegan', 'vegetarian']
        }
    ],
    'milk': [
        {
            substitute: 'almond-milk',
            ratio: '1:1 substitution',
            notes: 'Works in most recipes. Choose unsweetened for savory dishes.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'coconut-milk',
            ratio: '1:1 substitution',
            notes: 'Adds subtle coconut flavor. Great for curries and tropical dishes.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'oat-milk',
            ratio: '1:1 substitution',
            notes: 'Creamy texture, neutral flavor. Excellent for baking.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        }
    ],
    'eggs': [
        {
            substitute: 'flax-eggs',
            ratio: '1 egg = 1 tbsp ground flaxseed + 3 tbsp water',
            notes: 'Let sit for 5 minutes to gel. Best for baking.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'aquafaba',
            ratio: '1 egg = 3 tbsp chickpea liquid',
            notes: 'Excellent for meringues and light baking. Whips like egg whites.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'applesauce',
            ratio: '1 egg = 1/4 cup applesauce',
            notes: 'Best for moist baked goods. Adds slight sweetness.',
            confidence: 'medium',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'silken-tofu',
            ratio: '1 egg = 1/4 cup blended tofu',
            notes: 'Creates dense, moist texture. Great for brownies and custards.',
            confidence: 'medium',
            dietary: ['vegan', 'vegetarian']
        }
    ],
    'flour': [
        {
            substitute: 'almond-flour',
            ratio: '1 cup flour = 1 cup almond flour (+ binding agent)',
            notes: 'Gluten-free. May need xanthan gum. Denser texture.',
            confidence: 'medium',
            dietary: ['gluten-free', 'vegan', 'vegetarian']
        },
        {
            substitute: 'rice-flour',
            ratio: '1 cup flour = 7/8 cup rice flour',
            notes: 'Gluten-free. Creates lighter texture. Combine with starches.',
            confidence: 'medium',
            dietary: ['gluten-free', 'vegan', 'vegetarian']
        },
        {
            substitute: 'oat-flour',
            ratio: '1 cup flour = 1 1/3 cup oat flour',
            notes: 'Gluten-free if certified. Adds nutty flavor.',
            confidence: 'medium',
            dietary: ['gluten-free', 'vegan', 'vegetarian']
        }
    ],
    'sugar': [
        {
            substitute: 'honey',
            ratio: '1 cup sugar = 3/4 cup honey (reduce liquid by 1/4 cup)',
            notes: 'Sweeter than sugar. Adds moisture to baked goods.',
            confidence: 'high',
            dietary: ['vegetarian']
        },
        {
            substitute: 'maple-syrup',
            ratio: '1 cup sugar = 3/4 cup maple syrup (reduce liquid by 3 tbsp)',
            notes: 'Adds distinct maple flavor. Great for pancakes and baking.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        },
        {
            substitute: 'coconut-sugar',
            ratio: '1:1 substitution',
            notes: 'Similar texture to brown sugar. Lower glycemic index.',
            confidence: 'high',
            dietary: ['vegan', 'vegetarian']
        }
    ]
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { culinaryGraphData, substitutionRules };
}
