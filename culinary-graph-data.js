
// Culinary Knowledge Graph Data
// Represents ingredients, their relationships, and substitution rules

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
