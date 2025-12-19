
// Culinary Knowledge Graph Data
// Represents ingredients, their relationships, and substitution rules
// Enhanced with ontology, confidence scores, and metadata

// ============================================
// ONTOLOGY DEFINITION
// ============================================

const culinaryOntology = {
    // Metadata about the ontology itself
    metadata: {
        version: '2.0',
        created: '2025-01-01',
        lastUpdated: '2025-12-11',
        author: 'Victoria Ruiz Griffith',
        description: 'Comprehensive culinary knowledge graph ontology'
    },

    // Entity classes and their properties
    classes: {
        'Ingredient': {
            description: 'Any ingredient used in cooking',
            properties: ['id', 'label', 'category', 'cuisine', 'dietary', 'aliases', 'nutritionalInfo'],
            subclasses: ['Protein', 'Grain', 'Vegetable', 'Spice', 'Oil', 'Dairy', 'Herb', 'Sauce', 'Sweetener']
        },
        'Protein': {
            parent: 'Ingredient',
            subclasses: ['AnimalProtein', 'PlantProtein', 'Seafood']
        },
        'Recipe': {
            description: 'A combination of ingredients with preparation method',
            properties: ['name', 'ingredients', 'cuisine', 'difficulty', 'cookingTime', 'serves']
        },
        'Technique': {
            description: 'A cooking method or technique',
            properties: ['name', 'temperature', 'duration', 'equipment']
        }
    },

    // Relationship types with their semantics
    relationshipTypes: {
        'SUBSTITUTES': {
            description: 'One ingredient can replace another',
            cardinality: 'many-to-many',
            directional: true,
            properties: ['ratio', 'context', 'confidence', 'note'],
            inverseOf: null
        },
        'PAIRS_WITH': {
            description: 'Ingredients that complement each other',
            cardinality: 'many-to-many',
            directional: false,
            properties: ['strength', 'confidence'],
            inverseOf: 'PAIRS_WITH'
        },
        'USED_WITH': {
            description: 'Ingredients commonly used together in recipes',
            cardinality: 'many-to-many',
            directional: false,
            properties: ['context', 'frequency'],
            inverseOf: 'USED_WITH'
        },
        'SIMILAR_TO': {
            description: 'Ingredients with similar properties',
            cardinality: 'many-to-many',
            directional: false,
            properties: ['similarity_type', 'confidence'],
            inverseOf: 'SIMILAR_TO'
        }
    },

    // Dietary restriction vocabulary
    dietaryRestrictions: [
        'vegan', 'vegetarian', 'pescatarian', 'gluten-free', 'dairy-free',
        'nut-free', 'keto', 'paleo', 'low-carb', 'kosher', 'halal'
    ],

    // Cuisine types vocabulary
    cuisineTypes: [
        'italian', 'mexican', 'mediterranean', 'asian', 'greek', 'spanish',
        'french', 'indian', 'chinese', 'turkish', 'japanese', 'thai', 'all'
    ],

    // Category vocabulary
    categories: [
        'dairy', 'dairy-alt', 'protein', 'grain', 'vegetable', 'aromatic',
        'herb', 'spice', 'oil', 'sauce', 'sweetener', 'baking', 'nuts', 'acid', 'liquid'
    ]
};

// ============================================
// KNOWLEDGE GRAPH DATA
// ============================================

const culinaryGraphData = {
    // Nodes represent ingredients - Comprehensive cooking ingredient database
    nodes: [
        // ===== DAIRY PRODUCTS =====
        { id: 'butter', label: 'Butter', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'unsalted-butter', label: 'Unsalted Butter', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'ghee', label: 'Ghee', category: 'dairy', cuisine: ['asian'], dietary: ['vegetarian'] },
        { id: 'milk', label: 'Whole Milk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'skim-milk', label: 'Skim Milk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cream', label: 'Heavy Cream', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'half-and-half', label: 'Half & Half', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'sour-cream', label: 'Sour Cream', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'yogurt', label: 'Yogurt', category: 'dairy', cuisine: ['mediterranean', 'asian'], dietary: ['vegetarian'] },
        { id: 'greek-yogurt', label: 'Greek Yogurt', category: 'dairy', cuisine: ['mediterranean'], dietary: ['vegetarian'] },
        { id: 'buttermilk', label: 'Buttermilk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'condensed-milk', label: 'Condensed Milk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'evaporated-milk', label: 'Evaporated Milk', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },

        // Cheeses
        { id: 'cheese-cheddar', label: 'Cheddar Cheese', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cheese-parmesan', label: 'Parmesan', category: 'dairy', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'cheese-mozzarella', label: 'Mozzarella', category: 'dairy', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'cheese-ricotta', label: 'Ricotta', category: 'dairy', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'cheese-feta', label: 'Feta', category: 'dairy', cuisine: ['mediterranean'], dietary: ['vegetarian'] },
        { id: 'cheese-goat', label: 'Goat Cheese', category: 'dairy', cuisine: ['mediterranean'], dietary: ['vegetarian'] },
        { id: 'cheese-cream', label: 'Cream Cheese', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cheese-brie', label: 'Brie', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cheese-blue', label: 'Blue Cheese', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cheese-swiss', label: 'Swiss Cheese', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cheese-gruyere', label: 'Gruyere', category: 'dairy', cuisine: ['all'], dietary: ['vegetarian'] },

        // ===== DAIRY ALTERNATIVES =====
        { id: 'almond-milk', label: 'Almond Milk', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'soy-milk', label: 'Soy Milk', category: 'dairy-alt', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'oat-milk', label: 'Oat Milk', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-milk', label: 'Coconut Milk', category: 'dairy-alt', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'rice-milk', label: 'Rice Milk', category: 'dairy-alt', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cashew-milk', label: 'Cashew Milk', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-cream', label: 'Coconut Cream', category: 'dairy-alt', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'nutritional-yeast', label: 'Nutritional Yeast', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'vegan-butter', label: 'Vegan Butter', category: 'dairy-alt', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // ===== PROTEINS - MEAT =====
        { id: 'chicken-breast', label: 'Chicken Breast', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'chicken-thigh', label: 'Chicken Thigh', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'ground-chicken', label: 'Ground Chicken', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'turkey', label: 'Turkey', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'ground-turkey', label: 'Ground Turkey', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'beef', label: 'Beef', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'ground-beef', label: 'Ground Beef', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'steak', label: 'Steak', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'pork', label: 'Pork', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'pork-chop', label: 'Pork Chop', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'bacon', label: 'Bacon', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'sausage', label: 'Sausage', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'lamb', label: 'Lamb', category: 'protein', cuisine: ['mediterranean'], dietary: [] },
        { id: 'duck', label: 'Duck', category: 'protein', cuisine: ['asian'], dietary: [] },

        // ===== PROTEINS - SEAFOOD =====
        { id: 'salmon', label: 'Salmon', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'tuna', label: 'Tuna', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'cod', label: 'Cod', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'tilapia', label: 'Tilapia', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'shrimp', label: 'Shrimp', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'crab', label: 'Crab', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'lobster', label: 'Lobster', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'scallops', label: 'Scallops', category: 'protein', cuisine: ['all'], dietary: [] },
        { id: 'mussels', label: 'Mussels', category: 'protein', cuisine: ['mediterranean'], dietary: [] },
        { id: 'clams', label: 'Clams', category: 'protein', cuisine: ['mediterranean'], dietary: [] },
        { id: 'anchovies', label: 'Anchovies', category: 'protein', cuisine: ['mediterranean', 'italian'], dietary: [] },

        // ===== PROTEINS - PLANT-BASED =====
        { id: 'eggs', label: 'Eggs', category: 'protein', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'tofu', label: 'Tofu', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tempeh', label: 'Tempeh', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'seitan', label: 'Seitan', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chickpeas', label: 'Chickpeas', category: 'protein', cuisine: ['mediterranean', 'mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'black-beans', label: 'Black Beans', category: 'protein', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'kidney-beans', label: 'Kidney Beans', category: 'protein', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pinto-beans', label: 'Pinto Beans', category: 'protein', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'navy-beans', label: 'Navy Beans', category: 'protein', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'lentils', label: 'Lentils', category: 'protein', cuisine: ['mediterranean', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'red-lentils', label: 'Red Lentils', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'edamame', label: 'Edamame', category: 'protein', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // ===== GRAINS & FLOURS =====
        { id: 'flour', label: 'All-Purpose Flour', category: 'grain', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'bread-flour', label: 'Bread Flour', category: 'grain', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'cake-flour', label: 'Cake Flour', category: 'grain', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'whole-wheat-flour', label: 'Whole Wheat Flour', category: 'grain', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'almond-flour', label: 'Almond Flour', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'coconut-flour', label: 'Coconut Flour', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'rice-flour', label: 'Rice Flour', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'oat-flour', label: 'Oat Flour', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chickpea-flour', label: 'Chickpea Flour', category: 'grain', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'cornmeal', label: 'Cornmeal', category: 'grain', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'cornstarch', label: 'Cornstarch', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'rice', label: 'White Rice', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'brown-rice', label: 'Brown Rice', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'jasmine-rice', label: 'Jasmine Rice', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'basmati-rice', label: 'Basmati Rice', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'arborio-rice', label: 'Arborio Rice', category: 'grain', cuisine: ['italian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'quinoa', label: 'Quinoa', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'couscous', label: 'Couscous', category: 'grain', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bulgur', label: 'Bulgur', category: 'grain', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'farro', label: 'Farro', category: 'grain', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'barley', label: 'Barley', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'oats', label: 'Oats', category: 'grain', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pasta', label: 'Pasta', category: 'grain', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'rice-noodles', label: 'Rice Noodles', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'soba-noodles', label: 'Soba Noodles', category: 'grain', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // ===== VEGETABLES =====
        { id: 'tomato', label: 'Tomato', category: 'vegetable', cuisine: ['italian', 'mexican', 'mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cherry-tomatoes', label: 'Cherry Tomatoes', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bell-pepper', label: 'Bell Pepper', category: 'vegetable', cuisine: ['mexican', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'jalapeño', label: 'Jalapeño', category: 'vegetable', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'serrano-pepper', label: 'Serrano Pepper', category: 'vegetable', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'poblano-pepper', label: 'Poblano Pepper', category: 'vegetable', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'mushrooms', label: 'Mushrooms', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'shiitake', label: 'Shiitake Mushrooms', category: 'vegetable', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'portobello', label: 'Portobello', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'spinach', label: 'Spinach', category: 'vegetable', cuisine: ['mediterranean', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'kale', label: 'Kale', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'arugula', label: 'Arugula', category: 'vegetable', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'lettuce', label: 'Lettuce', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cabbage', label: 'Cabbage', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bok-choy', label: 'Bok Choy', category: 'vegetable', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'broccoli', label: 'Broccoli', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cauliflower', label: 'Cauliflower', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'brussels-sprouts', label: 'Brussels Sprouts', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'asparagus', label: 'Asparagus', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'green-beans', label: 'Green Beans', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'zucchini', label: 'Zucchini', category: 'vegetable', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'eggplant', label: 'Eggplant', category: 'vegetable', cuisine: ['mediterranean', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'carrots', label: 'Carrots', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'celery', label: 'Celery', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cucumber', label: 'Cucumber', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'radish', label: 'Radish', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'beets', label: 'Beets', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sweet-potato', label: 'Sweet Potato', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'potato', label: 'Potato', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'corn', label: 'Corn', category: 'vegetable', cuisine: ['mexican', 'all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'peas', label: 'Peas', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'squash', label: 'Squash', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pumpkin', label: 'Pumpkin', category: 'vegetable', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // ===== AROMATICS & HERBS =====
        { id: 'garlic', label: 'Garlic', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'onion', label: 'Onion', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'red-onion', label: 'Red Onion', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'shallot', label: 'Shallot', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'scallion', label: 'Scallion', category: 'aromatic', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'leek', label: 'Leek', category: 'aromatic', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'ginger', label: 'Ginger', category: 'aromatic', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'lemongrass', label: 'Lemongrass', category: 'aromatic', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // Herbs
        { id: 'basil', label: 'Basil', category: 'herb', cuisine: ['italian', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cilantro', label: 'Cilantro', category: 'herb', cuisine: ['mexican', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'parsley', label: 'Parsley', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'oregano', label: 'Oregano', category: 'herb', cuisine: ['italian', 'mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'thyme', label: 'Thyme', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'rosemary', label: 'Rosemary', category: 'herb', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sage', label: 'Sage', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'mint', label: 'Mint', category: 'herb', cuisine: ['mediterranean', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'dill', label: 'Dill', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chives', label: 'Chives', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tarragon', label: 'Tarragon', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bay-leaves', label: 'Bay Leaves', category: 'herb', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // Spices
        { id: 'cumin', label: 'Cumin', category: 'spice', cuisine: ['mexican', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coriander', label: 'Coriander', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'paprika', label: 'Paprika', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chili-powder', label: 'Chili Powder', category: 'spice', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cayenne', label: 'Cayenne Pepper', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'turmeric', label: 'Turmeric', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cinnamon', label: 'Cinnamon', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'nutmeg', label: 'Nutmeg', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cardamom', label: 'Cardamom', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cloves', label: 'Cloves', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'black-pepper', label: 'Black Pepper', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'white-pepper', label: 'White Pepper', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'red-pepper-flakes', label: 'Red Pepper Flakes', category: 'spice', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'garam-masala', label: 'Garam Masala', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'curry-powder', label: 'Curry Powder', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'five-spice', label: 'Five Spice', category: 'spice', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // ===== OILS & FATS =====
        { id: 'olive-oil', label: 'Olive Oil', category: 'oil', cuisine: ['mediterranean', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'extra-virgin-olive-oil', label: 'Extra Virgin Olive Oil', category: 'oil', cuisine: ['mediterranean', 'italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'vegetable-oil', label: 'Vegetable Oil', category: 'oil', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'canola-oil', label: 'Canola Oil', category: 'oil', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-oil', label: 'Coconut Oil', category: 'oil', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sesame-oil', label: 'Sesame Oil', category: 'oil', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'peanut-oil', label: 'Peanut Oil', category: 'oil', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'avocado-oil', label: 'Avocado Oil', category: 'oil', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'grapeseed-oil', label: 'Grapeseed Oil', category: 'oil', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sunflower-oil', label: 'Sunflower Oil', category: 'oil', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // ===== SAUCES & CONDIMENTS =====
        { id: 'soy-sauce', label: 'Soy Sauce', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tamari', label: 'Tamari', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian', 'gluten-free'] },
        { id: 'fish-sauce', label: 'Fish Sauce', category: 'sauce', cuisine: ['asian'], dietary: [] },
        { id: 'oyster-sauce', label: 'Oyster Sauce', category: 'sauce', cuisine: ['asian'], dietary: [] },
        { id: 'hoisin-sauce', label: 'Hoisin Sauce', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tomato-sauce', label: 'Tomato Sauce', category: 'sauce', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tomato-paste', label: 'Tomato Paste', category: 'sauce', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'salsa', label: 'Salsa', category: 'sauce', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'hot-sauce', label: 'Hot Sauce', category: 'sauce', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'worcestershire', label: 'Worcestershire Sauce', category: 'sauce', cuisine: ['all'], dietary: [] },
        { id: 'mustard', label: 'Mustard', category: 'sauce', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'dijon-mustard', label: 'Dijon Mustard', category: 'sauce', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'mayonnaise', label: 'Mayonnaise', category: 'sauce', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'ketchup', label: 'Ketchup', category: 'sauce', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'bbq-sauce', label: 'BBQ Sauce', category: 'sauce', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'tahini', label: 'Tahini', category: 'sauce', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pesto', label: 'Pesto', category: 'sauce', cuisine: ['italian'], dietary: ['vegetarian'] },
        { id: 'sriracha', label: 'Sriracha', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'miso-paste', label: 'Miso Paste', category: 'sauce', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },

        // ===== SWEETENERS =====
        { id: 'sugar', label: 'White Sugar', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'brown-sugar', label: 'Brown Sugar', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'powdered-sugar', label: 'Powdered Sugar', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'honey', label: 'Honey', category: 'sweetener', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'maple-syrup', label: 'Maple Syrup', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'agave', label: 'Agave Nectar', category: 'sweetener', cuisine: ['mexican'], dietary: ['vegan', 'vegetarian'] },
        { id: 'molasses', label: 'Molasses', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'stevia', label: 'Stevia', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-sugar', label: 'Coconut Sugar', category: 'sweetener', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'date-syrup', label: 'Date Syrup', category: 'sweetener', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },

        // ===== BAKING INGREDIENTS =====
        { id: 'baking-powder', label: 'Baking Powder', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'baking-soda', label: 'Baking Soda', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'yeast', label: 'Yeast', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'vanilla-extract', label: 'Vanilla Extract', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cocoa-powder', label: 'Cocoa Powder', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chocolate-chips', label: 'Chocolate Chips', category: 'baking', cuisine: ['all'], dietary: ['vegetarian'] },
        { id: 'coconut-flakes', label: 'Coconut Flakes', category: 'baking', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // ===== NUTS & SEEDS =====
        { id: 'almonds', label: 'Almonds', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'walnuts', label: 'Walnuts', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pecans', label: 'Pecans', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'cashews', label: 'Cashews', category: 'nuts', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'peanuts', label: 'Peanuts', category: 'nuts', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'peanut-butter', label: 'Peanut Butter', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'almond-butter', label: 'Almond Butter', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pine-nuts', label: 'Pine Nuts', category: 'nuts', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sesame-seeds', label: 'Sesame Seeds', category: 'nuts', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'chia-seeds', label: 'Chia Seeds', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'flax-seeds', label: 'Flax Seeds', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'pumpkin-seeds', label: 'Pumpkin Seeds', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'sunflower-seeds', label: 'Sunflower Seeds', category: 'nuts', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },

        // ===== ACIDS & LIQUIDS =====
        { id: 'lemon-juice', label: 'Lemon Juice', category: 'acid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'lime-juice', label: 'Lime Juice', category: 'acid', cuisine: ['mexican', 'asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'vinegar', label: 'White Vinegar', category: 'acid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'apple-cider-vinegar', label: 'Apple Cider Vinegar', category: 'acid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'balsamic-vinegar', label: 'Balsamic Vinegar', category: 'acid', cuisine: ['italian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'red-wine-vinegar', label: 'Red Wine Vinegar', category: 'acid', cuisine: ['mediterranean'], dietary: ['vegan', 'vegetarian'] },
        { id: 'rice-vinegar', label: 'Rice Vinegar', category: 'acid', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
        { id: 'wine-white', label: 'White Wine', category: 'liquid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'wine-red', label: 'Red Wine', category: 'liquid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'stock-chicken', label: 'Chicken Stock', category: 'liquid', cuisine: ['all'], dietary: [] },
        { id: 'stock-beef', label: 'Beef Stock', category: 'liquid', cuisine: ['all'], dietary: [] },
        { id: 'stock-vegetable', label: 'Vegetable Stock', category: 'liquid', cuisine: ['all'], dietary: ['vegan', 'vegetarian'] },
        { id: 'coconut-water', label: 'Coconut Water', category: 'liquid', cuisine: ['asian'], dietary: ['vegan', 'vegetarian'] },
    ],

    // Edges represent relationships between ingredients
    // Enhanced with confidence scores (0-1 scale) and metadata
    edges: [
        // Substitution relationships (can replace) - with confidence scores
        { source: 'butter', target: 'olive-oil', type: 'substitutes', ratio: '1:0.75', context: 'baking, cooking', confidence: 0.9 },
        { source: 'butter', target: 'coconut-oil', type: 'substitutes', ratio: '1:1', context: 'baking', confidence: 0.95 },
        { source: 'milk', target: 'almond-milk', type: 'substitutes', ratio: '1:1', context: 'all', confidence: 0.95 },
        { source: 'milk', target: 'coconut-milk', type: 'substitutes', ratio: '1:1', context: 'curries, baking', confidence: 0.9 },
        { source: 'eggs', target: 'tofu', type: 'substitutes', ratio: '1 egg:1/4 cup', context: 'scrambles, baking', confidence: 0.85 },
        { source: 'eggs', target: 'chickpeas', type: 'substitutes', ratio: '1 egg:3 tbsp aquafaba', context: 'baking', confidence: 0.8 },
        { source: 'flour', target: 'almond-flour', type: 'substitutes', ratio: '1:1', context: 'gluten-free baking', confidence: 0.75 },
        { source: 'flour', target: 'rice-flour', type: 'substitutes', ratio: '1:0.75', context: 'gluten-free', confidence: 0.7 },
        { source: 'sugar', target: 'honey', type: 'substitutes', ratio: '1:0.75', context: 'baking', confidence: 0.9 },
        { source: 'sugar', target: 'maple-syrup', type: 'substitutes', ratio: '1:0.75', context: 'baking', confidence: 0.9 },
        { source: 'sugar', target: 'agave', type: 'substitutes', ratio: '1:0.75', context: 'beverages, baking', confidence: 0.85 },
        { source: 'chicken-breast', target: 'tofu', type: 'substitutes', ratio: '1:1', context: 'stir-fry, curry', confidence: 0.8 },
        { source: 'chicken-breast', target: 'mushrooms', type: 'substitutes', ratio: '1:1.5', context: 'umami flavor', confidence: 0.7 },
        { source: 'cream', target: 'coconut-milk', type: 'substitutes', ratio: '1:1', context: 'dairy-free sauces', confidence: 0.9 },
        { source: 'cheese-parmesan', target: 'nutritional-yeast', type: 'substitutes', ratio: '1:1', context: 'vegan alternative', confidence: 0.75 },

        // Pairing relationships (goes well with) - with confidence scores
        { source: 'garlic', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'garlic', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'tomato', target: 'basil', type: 'pairs-with', strength: 'strong', confidence: 1.0 },
        { source: 'tomato', target: 'oregano', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'ginger', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'ginger', target: 'garlic', type: 'pairs-with', strength: 'medium', confidence: 0.9 },
        { source: 'cilantro', target: 'salsa', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'onion', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 1.0 },
        { source: 'cheese-mozzarella', target: 'tomato', type: 'pairs-with', strength: 'strong', confidence: 1.0 },
        { source: 'basil', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },

        // Recipe component relationships (used in)
        { source: 'butter', target: 'flour', type: 'used-with', context: 'baking', confidence: 1.0 },
        { source: 'eggs', target: 'flour', type: 'used-with', context: 'baking', confidence: 1.0 },
        { source: 'milk', target: 'eggs', type: 'used-with', context: 'custards, baking', confidence: 0.95 },
        { source: 'rice', target: 'soy-sauce', type: 'used-with', context: 'fried rice', confidence: 0.95 },
        { source: 'chickpeas', target: 'tahini', type: 'used-with', context: 'hummus', confidence: 1.0 },
        { source: 'black-beans', target: 'salsa', type: 'used-with', context: 'mexican dishes', confidence: 0.9 },

        // Similar flavor profile
        { source: 'oregano', target: 'basil', type: 'similar-flavor', note: 'italian herbs', confidence: 0.85 },
        { source: 'coconut-milk', target: 'cream', type: 'similar-texture', note: 'creamy consistency', confidence: 0.9 },
        { source: 'tofu', target: 'chicken-breast', type: 'similar-texture', note: 'absorbs flavors', confidence: 0.75 },
        { source: 'almond-milk', target: 'milk', type: 'similar-use', note: 'plant-based alternative', confidence: 0.9 },

        // === SWEETENER CONNECTIONS ===
        // Liquid sweeteners can substitute for each other
        { source: 'honey', target: 'maple-syrup', type: 'substitutes', ratio: '1:1', context: 'baking, beverages', confidence: 0.95 },
        { source: 'honey', target: 'agave', type: 'substitutes', ratio: '1:1', context: 'beverages, raw recipes', confidence: 0.9 },
        { source: 'honey', target: 'date-syrup', type: 'substitutes', ratio: '1:1', context: 'baking, drizzling', confidence: 0.85 },
        { source: 'honey', target: 'molasses', type: 'substitutes', ratio: '1:0.75', context: 'baking (stronger flavor)', confidence: 0.7 },
        { source: 'maple-syrup', target: 'agave', type: 'substitutes', ratio: '1:1', context: 'pancakes, beverages', confidence: 0.9 },
        { source: 'maple-syrup', target: 'date-syrup', type: 'substitutes', ratio: '1:1', context: 'baking, breakfast', confidence: 0.85 },
        { source: 'maple-syrup', target: 'molasses', type: 'substitutes', ratio: '1:0.5', context: 'baking (use less)', confidence: 0.65 },
        { source: 'agave', target: 'date-syrup', type: 'substitutes', ratio: '1:1', context: 'vegan baking', confidence: 0.85 },
        { source: 'date-syrup', target: 'molasses', type: 'substitutes', ratio: '1:0.75', context: 'baking, BBQ sauces', confidence: 0.75 },
        { source: 'sugar', target: 'date-syrup', type: 'substitutes', ratio: '1:0.66', context: 'baking, reduce liquid', confidence: 0.8 },
        { source: 'sugar', target: 'molasses', type: 'substitutes', ratio: '1:0.5', context: 'cookies, gingerbread', confidence: 0.75 },

        // === OIL CONNECTIONS ===
        // Cooking oils can often substitute for each other
        { source: 'olive-oil', target: 'avocado-oil', type: 'substitutes', ratio: '1:1', context: 'cooking, dressings', confidence: 0.95 },
        { source: 'olive-oil', target: 'vegetable-oil', type: 'substitutes', ratio: '1:1', context: 'cooking (not finishing)', confidence: 0.85 },
        { source: 'avocado-oil', target: 'vegetable-oil', type: 'substitutes', ratio: '1:1', context: 'high-heat cooking', confidence: 0.9 },
        { source: 'coconut-oil', target: 'vegetable-oil', type: 'substitutes', ratio: '1:1', context: 'baking, frying', confidence: 0.85 },
        { source: 'coconut-oil', target: 'butter', type: 'substitutes', ratio: '1:1', context: 'vegan baking', confidence: 0.9 },
        { source: 'sesame-oil', target: 'peanut-oil', type: 'substitutes', ratio: '1:1', context: 'asian cooking', confidence: 0.8 },
        { source: 'butter', target: 'ghee', type: 'substitutes', ratio: '1:1', context: 'high-heat, lactose-free', confidence: 0.95 },

        // === PROTEIN CONNECTIONS ===
        // Poultry substitutions
        { source: 'chicken-breast', target: 'turkey-breast', type: 'substitutes', ratio: '1:1', context: 'lean protein swap', confidence: 0.95 },
        { source: 'chicken-thighs', target: 'turkey-thighs', type: 'substitutes', ratio: '1:1', context: 'dark meat swap', confidence: 0.9 },
        { source: 'chicken-breast', target: 'chicken-thighs', type: 'substitutes', ratio: '1:1', context: 'more flavor, moisture', confidence: 0.85 },
        // Ground meat substitutions
        { source: 'ground-beef', target: 'ground-turkey', type: 'substitutes', ratio: '1:1', context: 'leaner option', confidence: 0.9 },
        { source: 'ground-beef', target: 'ground-pork', type: 'substitutes', ratio: '1:1', context: 'meatballs, burgers', confidence: 0.85 },
        { source: 'ground-turkey', target: 'ground-pork', type: 'substitutes', ratio: '1:1', context: 'lighter dishes', confidence: 0.8 },
        // Fish substitutions
        { source: 'salmon', target: 'trout', type: 'substitutes', ratio: '1:1', context: 'similar omega-3 fish', confidence: 0.9 },
        { source: 'salmon', target: 'arctic-char', type: 'substitutes', ratio: '1:1', context: 'mild salmon alternative', confidence: 0.85 },
        { source: 'cod', target: 'halibut', type: 'substitutes', ratio: '1:1', context: 'white fish', confidence: 0.9 },
        { source: 'cod', target: 'tilapia', type: 'substitutes', ratio: '1:1', context: 'mild white fish', confidence: 0.85 },
        { source: 'shrimp', target: 'prawns', type: 'substitutes', ratio: '1:1', context: 'nearly identical', confidence: 0.98 },
        { source: 'shrimp', target: 'scallops', type: 'substitutes', ratio: '1:1', context: 'shellfish', confidence: 0.8 },
        // Vegan protein substitutions
        { source: 'tofu', target: 'tempeh', type: 'substitutes', ratio: '1:1', context: 'vegan protein, firmer texture', confidence: 0.9 },
        { source: 'tofu', target: 'seitan', type: 'substitutes', ratio: '1:1', context: 'meat-like texture', confidence: 0.85 },
        { source: 'tempeh', target: 'seitan', type: 'substitutes', ratio: '1:1', context: 'hearty vegan protein', confidence: 0.85 },
        { source: 'lentils', target: 'black-beans', type: 'substitutes', ratio: '1:1', context: 'plant protein', confidence: 0.85 },
        { source: 'lentils', target: 'chickpeas', type: 'substitutes', ratio: '1:1', context: 'soups, stews', confidence: 0.85 },
        { source: 'black-beans', target: 'kidney-beans', type: 'substitutes', ratio: '1:1', context: 'chili, salads', confidence: 0.9 },

        // === DAIRY & ALTERNATIVE CONNECTIONS ===
        // Milk alternatives
        { source: 'milk', target: 'oat-milk', type: 'substitutes', ratio: '1:1', context: 'baking, coffee', confidence: 0.9 },
        { source: 'milk', target: 'soy-milk', type: 'substitutes', ratio: '1:1', context: 'baking, cooking', confidence: 0.9 },
        { source: 'almond-milk', target: 'oat-milk', type: 'substitutes', ratio: '1:1', context: 'plant-based swaps', confidence: 0.95 },
        { source: 'almond-milk', target: 'soy-milk', type: 'substitutes', ratio: '1:1', context: 'smoothies, cereal', confidence: 0.9 },
        { source: 'oat-milk', target: 'soy-milk', type: 'substitutes', ratio: '1:1', context: 'coffee, baking', confidence: 0.9 },
        { source: 'coconut-milk', target: 'cream', type: 'substitutes', ratio: '1:1', context: 'curries, vegan cream', confidence: 0.9 },
        // Cream alternatives
        { source: 'cream', target: 'half-and-half', type: 'substitutes', ratio: '1:1', context: 'sauces (less rich)', confidence: 0.85 },
        { source: 'sour-cream', target: 'greek-yogurt', type: 'substitutes', ratio: '1:1', context: 'toppings, dips', confidence: 0.9 },
        { source: 'yogurt', target: 'greek-yogurt', type: 'substitutes', ratio: '1:1', context: 'breakfast, baking', confidence: 0.95 },
        // Cheese within types
        { source: 'cheese-cheddar', target: 'cheese-gruyere', type: 'substitutes', ratio: '1:1', context: 'melting cheese', confidence: 0.85 },
        { source: 'cheese-mozzarella', target: 'cheese-provolone', type: 'substitutes', ratio: '1:1', context: 'pizza, sandwiches', confidence: 0.85 },
        { source: 'cheese-feta', target: 'cheese-goat', type: 'substitutes', ratio: '1:1', context: 'salads, crumbling', confidence: 0.85 },

        // === CLASSIC FLAVOR PAIRINGS ===
        // Italian flavors
        { source: 'tomato', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'tomato', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'basil', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'oregano', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'oregano', target: 'olive-oil', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'cheese-parmesan', target: 'basil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'cheese-parmesan', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        // Asian flavors
        { source: 'ginger', target: 'sesame-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'soy-sauce', target: 'sesame-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'soy-sauce', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'ginger', target: 'scallions', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'soy-sauce', target: 'rice', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'tofu', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        // Mexican flavors
        { source: 'cilantro', target: 'lime', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'cilantro', target: 'jalapeno', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cumin', target: 'cilantro', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'cumin', target: 'black-beans', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'lime', target: 'jalapeno', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'avocado', target: 'lime', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'avocado', target: 'cilantro', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        // Mediterranean flavors
        { source: 'lemon', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'lemon', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'lemon', target: 'oregano', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'tahini', target: 'lemon', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'tahini', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'feta', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        // Other classic pairings
        { source: 'chocolate', target: 'vanilla', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'chocolate', target: 'coffee', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cinnamon', target: 'apple', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'cinnamon', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'bacon', target: 'eggs', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'rosemary', target: 'lamb', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'thyme', target: 'chicken-breast', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'dill', target: 'salmon', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'mint', target: 'lamb', type: 'pairs-with', strength: 'strong', confidence: 0.9 },

        // === COMMON USED-TOGETHER COMBINATIONS ===
        // Mirepoix / Aromatics base
        { source: 'onion', target: 'celery', type: 'used-with', context: 'mirepoix base', confidence: 0.95 },
        { source: 'onion', target: 'carrot', type: 'used-with', context: 'mirepoix base', confidence: 0.95 },
        { source: 'celery', target: 'carrot', type: 'used-with', context: 'mirepoix, stocks', confidence: 0.95 },
        { source: 'garlic', target: 'ginger', type: 'used-with', context: 'asian aromatics', confidence: 0.95 },
        { source: 'garlic', target: 'onion', type: 'used-with', context: 'sofrito, base', confidence: 0.95 },
        // Baking combinations
        { source: 'flour', target: 'sugar', type: 'used-with', context: 'baking', confidence: 0.95 },
        { source: 'eggs', target: 'sugar', type: 'used-with', context: 'baking, custards', confidence: 0.95 },
        { source: 'butter', target: 'sugar', type: 'used-with', context: 'creaming method', confidence: 0.95 },
        { source: 'vanilla', target: 'sugar', type: 'used-with', context: 'baking, desserts', confidence: 0.9 },
        { source: 'baking-powder', target: 'flour', type: 'used-with', context: 'leavening', confidence: 0.95 },
        // Common cooking combinations
        { source: 'rice', target: 'onion', type: 'used-with', context: 'pilaf, fried rice', confidence: 0.9 },
        { source: 'pasta', target: 'garlic', type: 'used-with', context: 'pasta dishes', confidence: 0.95 },
        { source: 'pasta', target: 'olive-oil', type: 'used-with', context: 'pasta dishes', confidence: 0.95 },
        { source: 'pasta', target: 'cheese-parmesan', type: 'used-with', context: 'finishing pasta', confidence: 0.95 },
        { source: 'potato', target: 'butter', type: 'used-with', context: 'mashed potatoes', confidence: 0.95 },
        { source: 'potato', target: 'cream', type: 'used-with', context: 'mashed, gratin', confidence: 0.9 },
        // Salad/fresh combinations
        { source: 'lettuce', target: 'tomato', type: 'used-with', context: 'salads', confidence: 0.95 },
        { source: 'cucumber', target: 'tomato', type: 'used-with', context: 'salads', confidence: 0.9 },
        { source: 'avocado', target: 'tomato', type: 'used-with', context: 'guac, salads', confidence: 0.95 },

        // === CHEESE CONNECTIONS ===
        // Soft/creamy cheese substitutions
        { source: 'cheese-brie', target: 'cheese-cream', type: 'substitutes', ratio: '1:1', context: 'spreading, appetizers', confidence: 0.8 },
        { source: 'cheese-brie', target: 'cheese-goat', type: 'substitutes', ratio: '1:1', context: 'cheese boards', confidence: 0.75 },
        { source: 'cheese-cream', target: 'cheese-ricotta', type: 'substitutes', ratio: '1:1', context: 'spreads, dips', confidence: 0.85 },
        // Blue cheese connections
        { source: 'cheese-blue', target: 'cheese-feta', type: 'substitutes', ratio: '1:1', context: 'salads, crumbling', confidence: 0.7 },
        { source: 'cheese-blue', target: 'cheese-goat', type: 'substitutes', ratio: '1:1', context: 'strong flavor', confidence: 0.7 },
        // Swiss/gruyere connections
        { source: 'cheese-swiss', target: 'cheese-gruyere', type: 'substitutes', ratio: '1:1', context: 'melting, sandwiches', confidence: 0.9 },
        { source: 'cheese-gruyere', target: 'cheese-cheddar', type: 'substitutes', ratio: '1:1', context: 'gratins, fondue', confidence: 0.8 },
        // Cheese pairings
        { source: 'cheese-brie', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'cheese-blue', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cheese-blue', target: 'walnuts', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cheese-goat', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cheese-cream', target: 'chives', type: 'pairs-with', strength: 'strong', confidence: 0.9 },

        // === SEAFOOD CONNECTIONS ===
        // Shellfish substitutions
        { source: 'lobster', target: 'crab', type: 'substitutes', ratio: '1:1', context: 'luxury seafood', confidence: 0.85 },
        { source: 'lobster', target: 'shrimp', type: 'substitutes', ratio: '1:1', context: 'shellfish dishes', confidence: 0.8 },
        { source: 'crab', target: 'shrimp', type: 'substitutes', ratio: '1:1', context: 'crab cakes, salads', confidence: 0.85 },
        { source: 'mussels', target: 'clams', type: 'substitutes', ratio: '1:1', context: 'steamed, pasta', confidence: 0.9 },
        { source: 'scallops', target: 'shrimp', type: 'substitutes', ratio: '1:1', context: 'seared, pasta', confidence: 0.8 },
        // Seafood pairings
        { source: 'lobster', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'lobster', target: 'lemon-juice', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'crab', target: 'lemon-juice', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'scallops', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'scallops', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'mussels', target: 'wine-white', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'mussels', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'clams', target: 'wine-white', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'clams', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'anchovies', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'anchovies', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        // Duck and lamb
        { source: 'duck', target: 'chicken-thigh', type: 'substitutes', ratio: '1:1', context: 'rich poultry', confidence: 0.75 },
        { source: 'duck', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'lamb', target: 'beef', type: 'substitutes', ratio: '1:1', context: 'red meat', confidence: 0.8 },

        // === VEGETABLE CONNECTIONS ===
        { source: 'brussels-sprouts', target: 'bacon', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'brussels-sprouts', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'brussels-sprouts', target: 'garlic', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'asparagus', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'asparagus', target: 'lemon-juice', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'asparagus', target: 'garlic', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'asparagus', target: 'cheese-parmesan', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'beets', target: 'cheese-goat', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'beets', target: 'olive-oil', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'beets', target: 'balsamic-vinegar', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'peas', target: 'mint', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'peas', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'squash', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'squash', target: 'sage', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'squash', target: 'maple-syrup', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'pumpkin', target: 'cinnamon', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'pumpkin', target: 'nutmeg', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'pumpkin', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'bok-choy', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'bok-choy', target: 'ginger', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'bok-choy', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'radish', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.9 },

        // === GRAIN CONNECTIONS ===
        { source: 'farro', target: 'barley', type: 'substitutes', ratio: '1:1', context: 'ancient grains', confidence: 0.9 },
        { source: 'farro', target: 'quinoa', type: 'substitutes', ratio: '1:1', context: 'grain bowls', confidence: 0.85 },
        { source: 'barley', target: 'rice', type: 'substitutes', ratio: '1:1', context: 'soups, risotto-style', confidence: 0.8 },
        { source: 'bulgur', target: 'couscous', type: 'substitutes', ratio: '1:1', context: 'mediterranean dishes', confidence: 0.9 },
        { source: 'bulgur', target: 'quinoa', type: 'substitutes', ratio: '1:1', context: 'tabbouleh, salads', confidence: 0.85 },
        { source: 'couscous', target: 'quinoa', type: 'substitutes', ratio: '1:1', context: 'side dishes', confidence: 0.85 },
        { source: 'jasmine-rice', target: 'basmati-rice', type: 'substitutes', ratio: '1:1', context: 'aromatic rice', confidence: 0.95 },
        { source: 'jasmine-rice', target: 'rice', type: 'substitutes', ratio: '1:1', context: 'asian dishes', confidence: 0.9 },
        { source: 'basmati-rice', target: 'rice', type: 'substitutes', ratio: '1:1', context: 'indian dishes', confidence: 0.9 },
        { source: 'arborio-rice', target: 'rice', type: 'substitutes', ratio: '1:1', context: 'risotto only', confidence: 0.7 },

        // === SPICE CONNECTIONS ===
        { source: 'cardamom', target: 'cinnamon', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'cardamom', target: 'ginger', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'cloves', target: 'cinnamon', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'cloves', target: 'nutmeg', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'nutmeg', target: 'cinnamon', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'five-spice', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'five-spice', target: 'ginger', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'white-pepper', target: 'black-pepper', type: 'substitutes', ratio: '1:1', context: 'asian cuisine', confidence: 0.9 },
        { source: 'garam-masala', target: 'curry-powder', type: 'substitutes', ratio: '1:0.5', context: 'indian spice blend', confidence: 0.75 },

        // === HERB CONNECTIONS ===
        { source: 'sage', target: 'thyme', type: 'substitutes', ratio: '1:1', context: 'poultry, stuffing', confidence: 0.8 },
        { source: 'sage', target: 'butter', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'sage', target: 'pork', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'tarragon', target: 'chicken-breast', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'tarragon', target: 'eggs', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'tarragon', target: 'cream', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'bay-leaves', target: 'thyme', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'bay-leaves', target: 'onion', type: 'used-with', context: 'stocks, soups', confidence: 0.95 },
        { source: 'chives', target: 'sour-cream', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'chives', target: 'eggs', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'chives', target: 'potato', type: 'pairs-with', strength: 'strong', confidence: 0.9 },

        // === DAIRY ALTERNATIVE CONNECTIONS ===
        { source: 'cashew-milk', target: 'almond-milk', type: 'substitutes', ratio: '1:1', context: 'plant milk swap', confidence: 0.95 },
        { source: 'cashew-milk', target: 'milk', type: 'substitutes', ratio: '1:1', context: 'creamy alternative', confidence: 0.9 },
        { source: 'rice-milk', target: 'almond-milk', type: 'substitutes', ratio: '1:1', context: 'nut-free option', confidence: 0.9 },
        { source: 'rice-milk', target: 'milk', type: 'substitutes', ratio: '1:1', context: 'mild flavor', confidence: 0.85 },
        { source: 'coconut-cream', target: 'cream', type: 'substitutes', ratio: '1:1', context: 'vegan whipping', confidence: 0.9 },
        { source: 'coconut-cream', target: 'coconut-milk', type: 'substitutes', ratio: '1:0.5', context: 'thicker consistency', confidence: 0.85 },

        // === SAUCE CONNECTIONS ===
        { source: 'hoisin-sauce', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'hoisin-sauce', target: 'ginger', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'hoisin-sauce', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'oyster-sauce', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'oyster-sauce', target: 'garlic', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'worcestershire', target: 'beef', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'worcestershire', target: 'garlic', type: 'pairs-with', strength: 'medium', confidence: 0.85 },
        { source: 'dijon-mustard', target: 'honey', type: 'pairs-with', strength: 'strong', confidence: 0.95 },
        { source: 'dijon-mustard', target: 'olive-oil', type: 'used-with', context: 'vinaigrettes', confidence: 0.95 },
        { source: 'sriracha', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'sriracha', target: 'lime-juice', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'sriracha', target: 'mayonnaise', type: 'used-with', context: 'spicy mayo', confidence: 0.95 },
        { source: 'miso-paste', target: 'soy-sauce', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'miso-paste', target: 'ginger', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'miso-paste', target: 'sesame-oil', type: 'pairs-with', strength: 'strong', confidence: 0.9 },
        { source: 'tahini', target: 'lemon-juice', type: 'used-with', context: 'hummus, dressings', confidence: 0.95 },
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
    module.exports = { culinaryGraphData, substitutionRules, culinaryOntology };
}

// Make ontology available globally
window.culinaryOntology = culinaryOntology;
