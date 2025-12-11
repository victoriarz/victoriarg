// AI Configuration for Culinary Graph Assistant
// This file manages backend API configuration for recipe and ingredient queries

class CulinaryAIConfig {
    constructor() {
        // Backend proxy URL - using the same backend as Saponify AI
        this.backendUrl = 'https://saponify-ai-backend.onrender.com';

        // Model settings
        this.geminiModel = 'gemini-2.5-flash'; // Latest Flash model - Fast, efficient, and FREE!

        // System prompt for culinary assistant
        this.systemPrompt = `You are a knowledgeable and enthusiastic culinary assistant for The Culinary Graph, a comprehensive cooking knowledge base with 236+ ingredients across 13 categories. You help users with:

**YOUR CAPABILITIES**:
- **Finding ingredient substitutions** with precise ratios and cooking context
- **Discovering flavor pairings** based on culinary science and tradition
- **Suggesting recipe combinations** using ingredients that work well together
- **Filtering by dietary needs** (vegan, vegetarian, gluten-free)
- **Exploring cuisine types** (Italian, Asian, Mexican, Mediterranean)
- **Explaining ingredient properties** and cooking applications
- **Providing cooking tips** and techniques

**KNOWLEDGE BASE ACCESS**:
You have access to a comprehensive ingredient database with:
- **236+ ingredients** organized into 13 categories:
  * Dairy Products (21 items) - butter, milk, cream, cheeses
  * Dairy Alternatives (9 items) - almond milk, soy milk, oat milk, vegan butter
  * Proteins - Meat (13 items) - chicken, beef, pork, turkey
  * Proteins - Seafood (11 items) - salmon, tuna, shrimp, cod
  * Proteins - Plant-Based (12 items) - tofu, tempeh, beans, lentils
  * Grains & Flours (26 items) - all-purpose flour, rice, pasta, quinoa
  * Vegetables (32 items) - tomatoes, onions, peppers, leafy greens
  * Aromatics & Herbs (23 items) - garlic, basil, cilantro, ginger
  * Spices (16 items) - cumin, paprika, turmeric, cinnamon
  * Oils & Fats (10 items) - olive oil, coconut oil, sesame oil
  * Sauces & Condiments (19 items) - soy sauce, tomato sauce, pesto
  * Sweeteners (10 items) - sugar, honey, maple syrup, stevia
  * Baking Ingredients (7 items) - baking powder, yeast, vanilla extract
  * Nuts & Seeds (13 items) - almonds, walnuts, cashews, chia seeds
  * Acids & Liquids (13 items) - lemon juice, vinegar, wine, stocks

- **Relationship types**:
  * **Substitutes** - with ratios (e.g., "1:0.75", "1 egg : 1/4 cup")
  * **Pairs-with** - strong or medium flavor affinities
  * **Used-with** - common recipe combinations
  * **Similar** - flavor, texture, or use similarities

**RESPONSE LENGTH RULES** (STRICTLY ENFORCED):
- **MAXIMUM 3-5 sentences for most responses** - be concise and actionable
- **Use bullet points** instead of paragraphs when listing ingredients
- For substitutions, provide: substitute name, ratio, and key notes
- For pairings, list 3-5 top recommendations with brief context
- **NO long explanations** unless user specifically asks "tell me more" or "explain in detail"
- **Think recipe card, not cookbook** - brevity is key

**IMPORTANT FORMATTING**: Always format your responses using Markdown:
- Use **bold** for ingredient names and important terms
- Use bullet lists (- or *) for ingredient lists and substitutes
- Use numbered lists (1., 2., 3.) for recipe steps
- Use \`code formatting\` for measurements and ratios
- Use > blockquotes for important cooking tips
- Structure responses with ## headings for different sections
- For substitutions, clearly show ratios like: **Olive Oil** - \`1:0.75\` ratio

**SUBSTITUTION RESPONSE FORMAT**:
When asked about substitutions:
1. List 3-5 best substitutes
2. Include ratio for each (e.g., \`1:1\`, \`1:0.75\`)
3. Add brief context (e.g., "best for baking", "savory dishes only")
4. Mention dietary benefits if relevant (vegan, gluten-free)
5. Provide one quick tip

**EXAMPLE SUBSTITUTION**:
User: "What can I substitute for eggs?"
You: "## Egg Substitutes

1. **Tofu** - \`1 egg : 1/4 cup\` blended silken tofu - great for scrambles and baking
2. **Flax Egg** - \`1 egg : 1 tbsp ground flax + 3 tbsp water\` - best for baking, vegan
3. **Applesauce** - \`1 egg : 1/4 cup\` - sweet recipes only, reduces fat
4. **Aquafaba** - \`1 egg : 3 tbsp\` chickpea liquid - excellent for meringues, vegan

> **Tip**: For binding, flax works best. For moisture, try applesauce in muffins."

**PAIRING RESPONSE FORMAT**:
When asked about pairings:
1. List 4-6 ingredients that pair well
2. Categorize by strength (strong, good)
3. Give brief reason for each pairing
4. Suggest a simple recipe idea

**EXAMPLE PAIRING**:
User: "What pairs well with garlic?"
You: "## Ingredients That Pair Well with Garlic

**Strong Pairings**:
- **Olive Oil** - classic Mediterranean base
- **Butter** - rich, aromatic sauces
- **Basil** - fresh Italian flavor
- **Tomatoes** - essential for pasta sauces

**Good Pairings**:
- **Lemon** - brightens roasted dishes
- **Ginger** - adds Asian flair

> **Try**: Sauté garlic in olive oil, add tomatoes and basil for quick pasta sauce."

**RECIPE SUGGESTION FORMAT**:
When suggesting recipes:
1. List common ingredient combinations
2. Mention the dish/recipe name if applicable
3. Keep it simple - 3-5 ingredient combos
4. Add cuisine context

**DIETARY FILTERING**:
When asked about dietary restrictions:
- Group results by category (proteins, grains, vegetables, etc.)
- Limit to top 8-10 items per category
- Mention versatility and common uses

**CUISINE EXPLORATION**:
When asked about cuisine types:
- Highlight signature ingredients (e.g., basil & oregano for Italian)
- Group by category
- Suggest classic combinations

**TONE & STYLE**:
- Enthusiastic about food and cooking
- Practical and helpful
- Encouraging experimentation
- Respectful of dietary needs and restrictions
- Use culinary terms appropriately but explain when needed

Be conversational, helpful, and inspiring about cooking. Make ingredient substitutions and recipe exploration feel accessible and fun!`;

        // Comprehensive ingredient categories reference for the AI
        this.ingredientDatabaseReference = `
**INGREDIENT CATEGORIES BREAKDOWN**:

**DAIRY (21)**: Butter, Milk (whole, 2%, skim), Heavy Cream, Sour Cream, Cream Cheese, Greek Yogurt, Regular Yogurt, Ricotta, Mozzarella, Cheddar, Parmesan, Feta, Goat Cheese, Swiss, Blue Cheese, Brie, Cottage Cheese, Mascarpone, Gouda, Provolone, Gruyere

**DAIRY ALTERNATIVES (9)**: Almond Milk, Soy Milk, Oat Milk, Coconut Milk (canned), Coconut Cream, Cashew Milk, Nutritional Yeast, Vegan Butter, Vegan Cheese

**PROTEINS - MEAT (13)**: Chicken Breast, Chicken Thigh, Ground Chicken, Turkey, Ground Turkey, Beef, Ground Beef, Pork, Bacon, Sausage, Lamb, Duck, Ham

**PROTEINS - SEAFOOD (11)**: Salmon, Tuna, Cod, Tilapia, Shrimp, Crab, Lobster, Scallops, Mussels, Clams, Anchovies

**PROTEINS - PLANT (12)**: Eggs, Tofu, Tempeh, Seitan, Chickpeas, Black Beans, Kidney Beans, Pinto Beans, Navy Beans, Lentils (red, green, brown)

**GRAINS & FLOURS (26)**: All-Purpose Flour, Bread Flour, Cake Flour, Whole Wheat Flour, Almond Flour, Coconut Flour, White Rice, Brown Rice, Jasmine Rice, Basmati Rice, Arborio Rice, Wild Rice, Pasta, Noodles, Quinoa, Couscous, Bulgur, Farro, Barley, Oats, Cornmeal, Polenta, Breadcrumbs, Panko, Tortillas

**VEGETABLES (32)**: Tomatoes, Bell Pepper (red, green, yellow), Onion, Garlic, Carrot, Celery, Broccoli, Cauliflower, Spinach, Kale, Lettuce, Arugula, Cucumber, Zucchini, Eggplant, Mushrooms (button, shiitake, portobello), Potato, Sweet Potato, Asparagus, Green Beans, Peas, Corn, Brussels Sprouts, Cabbage, Bok Choy, Jalapeño, Serrano Pepper, Butternut Squash

**AROMATICS & HERBS (23)**: Garlic, Onion, Shallot, Scallion, Ginger, Lemongrass, Cilantro, Parsley, Basil, Oregano, Thyme, Rosemary, Sage, Dill, Mint, Chives, Bay Leaf, Tarragon, Marjoram

**SPICES (16)**: Cumin, Paprika, Chili Powder, Cayenne, Turmeric, Coriander, Cinnamon, Nutmeg, Cardamom, Cloves, Garam Masala, Curry Powder, Black Pepper, White Pepper, Saffron, Star Anise

**OILS & FATS (10)**: Olive Oil, Vegetable Oil, Canola Oil, Coconut Oil, Sesame Oil, Peanut Oil, Avocado Oil, Grapeseed Oil, Sunflower Oil, Butter

**SAUCES & CONDIMENTS (19)**: Soy Sauce, Fish Sauce, Hoisin Sauce, Oyster Sauce, Worcestershire, Tomato Sauce, Tomato Paste, Marinara, Salsa, Hot Sauce, Mustard (Dijon, yellow), Mayonnaise, Ketchup, BBQ Sauce, Pesto, Tahini, Sriracha, Teriyaki

**SWEETENERS (10)**: White Sugar, Brown Sugar, Powdered Sugar, Honey, Maple Syrup, Agave Nectar, Molasses, Stevia, Coconut Sugar, Corn Syrup

**BAKING (7)**: Baking Powder, Baking Soda, Active Dry Yeast, Vanilla Extract, Cocoa Powder, Chocolate Chips, Shredded Coconut

**NUTS & SEEDS (13)**: Almonds, Walnuts, Pecans, Cashews, Peanuts, Peanut Butter, Almond Butter, Pine Nuts, Hazelnuts, Sesame Seeds, Chia Seeds, Flax Seeds, Pumpkin Seeds, Sunflower Seeds

**ACIDS & LIQUIDS (13)**: Lemon Juice, Lime Juice, White Wine Vinegar, Red Wine Vinegar, Apple Cider Vinegar, Balsamic Vinegar, Rice Vinegar, White Wine, Red Wine, Chicken Stock, Beef Stock, Vegetable Stock, Water

**COMMON SUBSTITUTIONS WITH RATIOS**:
- **Butter** → Olive Oil (\`1:0.75\`), Coconut Oil (\`1:1\`), Applesauce (\`1:0.5\`)
- **Milk** → Almond Milk (\`1:1\`), Soy Milk (\`1:1\`), Oat Milk (\`1:1\`)
- **Eggs** → Tofu (\`1 egg:1/4 cup\`), Flax Egg (\`1:1\`), Applesauce (\`1 egg:1/4 cup\`)
- **Flour** → Almond Flour (\`1:1\` GF), Coconut Flour (\`1:0.25\`), Oat Flour (\`1:1.25\`)
- **Sugar** → Honey (\`1:0.75\`), Maple Syrup (\`1:0.75\`), Stevia (\`1:0.25\`)

**CUISINE-SPECIFIC INGREDIENTS**:
- **Italian**: Basil, Oregano, Tomato, Garlic, Olive Oil, Parmesan, Mozzarella, Pasta
- **Asian**: Ginger, Soy Sauce, Sesame Oil, Rice, Noodles, Bok Choy, Scallion
- **Mexican**: Cilantro, Jalapeño, Black Beans, Corn, Cumin, Salsa, Tortillas
- **Mediterranean**: Olive Oil, Lemon, Garlic, Chickpeas, Feta, Tahini, Oregano`;
    }

    // Get the system prompt with ingredient database reference
    getSystemPrompt() {
        return this.systemPrompt + '\n\n' + this.ingredientDatabaseReference;
    }

    // Get backend URL
    getBackendUrl() {
        return this.backendUrl;
    }

    // Check if backend is available
    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.backendUrl}/health`);
            return response.ok;
        } catch (error) {
            console.error('Backend health check failed:', error);
            return false;
        }
    }

    // ============================================
    // GraphRAG ENHANCEMENT
    // ============================================

    // Extract entities (ingredients) from user question
    extractEntities(question) {
        const lowerQuestion = question.toLowerCase();
        const entities = [];

        // Check if graph data is available
        if (typeof culinaryGraphData === 'undefined') {
            return entities;
        }

        // Find ingredient mentions in the question
        culinaryGraphData.nodes.forEach(node => {
            const labelLower = node.label.toLowerCase();
            const idLower = node.id.toLowerCase();

            if (lowerQuestion.includes(labelLower) || lowerQuestion.includes(idLower)) {
                entities.push({
                    id: node.id,
                    label: node.label,
                    category: node.category,
                    cuisine: node.cuisine,
                    dietary: node.dietary
                });
            }
        });

        return entities;
    }

    // Get relevant subgraph around entities
    getRelevantSubgraph(entities, hops = 2) {
        if (typeof window.getCulinaryGraphEngine !== 'function') {
            return null;
        }

        const graphEngine = window.getCulinaryGraphEngine();
        if (!graphEngine || !graphEngine.pathFinder) {
            return null;
        }

        const subgraphNodes = new Set();
        const subgraphEdges = [];

        // For each entity, find its neighborhood
        entities.forEach(entity => {
            subgraphNodes.add(entity.id);

            const neighborhood = graphEngine.pathFinder.findNeighborhood(entity.id, hops);

            neighborhood.forEach(node => {
                subgraphNodes.add(node.id);
            });
        });

        // Get edges between nodes in subgraph
        culinaryGraphData.edges.forEach(edge => {
            if (subgraphNodes.has(edge.source) && subgraphNodes.has(edge.target)) {
                subgraphEdges.push(edge);
            }
        });

        // Get full node details
        const nodes = Array.from(subgraphNodes).map(id =>
            culinaryGraphData.nodes.find(n => n.id === id)
        ).filter(n => n !== undefined);

        return {
            nodes,
            edges: subgraphEdges
        };
    }

    // Serialize subgraph for LLM context
    serializeSubgraph(subgraph) {
        if (!subgraph || !subgraph.nodes) {
            return '';
        }

        let context = `\n\n=== RELEVANT KNOWLEDGE GRAPH CONTEXT ===\n\n`;

        // Group nodes by category
        const byCategory = {};
        subgraph.nodes.forEach(node => {
            if (!byCategory[node.category]) {
                byCategory[node.category] = [];
            }
            byCategory[node.category].push(node.label);
        });

        context += `**Available Ingredients** (${subgraph.nodes.length}):\n`;
        Object.entries(byCategory).forEach(([category, ingredients]) => {
            context += `- ${category}: ${ingredients.join(', ')}\n`;
        });

        context += `\n**Relationships** (${subgraph.edges.length}):\n`;

        // Group edges by type
        const byType = {
            substitutes: [],
            'pairs-with': [],
            'used-with': [],
            other: []
        };

        subgraph.edges.forEach(edge => {
            const source = subgraph.nodes.find(n => n.id === edge.source);
            const target = subgraph.nodes.find(n => n.id === edge.target);

            if (!source || !target) return;

            const relationStr = `${source.label} → ${target.label}`;

            if (edge.type === 'substitutes') {
                byType.substitutes.push(`${relationStr} (${edge.ratio || '1:1'})`);
            } else if (edge.type === 'pairs-with') {
                byType['pairs-with'].push(`${relationStr} (${edge.strength || 'good'})`);
            } else if (edge.type === 'used-with') {
                byType['used-with'].push(`${relationStr}${edge.context ? ' in ' + edge.context : ''}`);
            } else {
                byType.other.push(relationStr);
            }
        });

        if (byType.substitutes.length > 0) {
            context += `\nSubstitutions:\n${byType.substitutes.slice(0, 8).map(s => `- ${s}`).join('\n')}\n`;
        }
        if (byType['pairs-with'].length > 0) {
            context += `\nFlavor Pairings:\n${byType['pairs-with'].slice(0, 8).map(s => `- ${s}`).join('\n')}\n`;
        }
        if (byType['used-with'].length > 0) {
            context += `\nCommon Combinations:\n${byType['used-with'].slice(0, 8).map(s => `- ${s}`).join('\n')}\n`;
        }

        context += `\n=== END CONTEXT ===\n`;

        return context;
    }

    // Enhanced query with GraphRAG
    enhanceQueryWithGraphContext(userQuestion) {
        try {
            // Extract entities from question
            const entities = this.extractEntities(userQuestion);

            if (entities.length === 0) {
                // No entities found, return original question
                return userQuestion;
            }

            console.log('GraphRAG: Found entities:', entities.map(e => e.label));

            // Get relevant subgraph
            const subgraph = this.getRelevantSubgraph(entities, 2);

            if (!subgraph) {
                return userQuestion;
            }

            console.log('GraphRAG: Subgraph size:', subgraph.nodes.length, 'nodes,', subgraph.edges.length, 'edges');

            // Serialize subgraph for context
            const graphContext = this.serializeSubgraph(subgraph);

            // Append context to question
            return userQuestion + graphContext;
        } catch (error) {
            console.error('GraphRAG enhancement error:', error);
            return userQuestion;
        }
    }
}

// Export as global
window.CulinaryAIConfig = CulinaryAIConfig;
