// AI Configuration for Culinary Graph Assistant
// RAG-Enhanced Architecture - Local knowledge graph data enhances LLM responses
// This file manages backend API configuration and focused context extraction

class CulinaryAIConfig {
    constructor() {
        // Backend proxy URL - using the same backend as Saponify AI
        this.backendUrl = 'https://saponify-ai-backend.onrender.com';

        // Model settings
        this.geminiModel = 'gemini-2.5-flash'; // Latest Flash model - Fast, efficient, and FREE!

        // Minimal system prompt - no hardcoded ingredient data
        // Context comes dynamically from RAG retrieval
        this.systemPrompt = `You are a knowledgeable and enthusiastic culinary assistant for Culinary Graph.

**YOUR CAPABILITIES**:
- Finding ingredient substitutions with precise ratios
- Discovering flavor pairings based on culinary science
- Suggesting recipe combinations
- Filtering by dietary needs (vegan, vegetarian, gluten-free)
- Exploring cuisine types (Italian, Asian, Mexican, Mediterranean)
- Explaining ingredient properties and cooking tips

**RESPONSE RULES** (STRICTLY ENFORCED):
- **MAXIMUM 3-5 sentences** for most responses - be concise and actionable
- **Use bullet points** for ingredient lists
- For substitutions: name, ratio, and brief notes
- For pairings: list 3-5 top recommendations
- **Think recipe card, not cookbook** - brevity is key

**CONTEXT USAGE**:
- You will receive relevant context from a knowledge graph with each query
- **USE THE PROVIDED CONTEXT** for specific ingredient data, relationships, and ratios
- Reference the context data directly in your response
- If no context is provided, give general cooking advice

**FORMATTING**:
- Use **bold** for ingredient names
- Use \`code\` for measurements and ratios (e.g., \`1:0.75\`)
- Use bullet lists for options
- Use > blockquotes for tips
- Use ## headings for sections

**TONE**:
- Enthusiastic about food and cooking
- Practical and helpful
- Encouraging experimentation
- Respectful of dietary needs`;
    }

    // Get the minimal system prompt (no hardcoded data)
    getSystemPrompt() {
        return this.systemPrompt;
    }

    // Get system prompt with dynamic RAG context
    getContextualSystemPrompt(ragContext) {
        if (ragContext && ragContext.trim() !== '') {
            return this.systemPrompt + '\n\n=== KNOWLEDGE GRAPH CONTEXT ===\n' + ragContext + '\n=== END CONTEXT ===';
        }
        return this.systemPrompt;
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
    // GraphRAG ENHANCEMENT - Focused Context Extraction
    // ============================================

    // Extract entities (ingredients) from user question
    extractEntities(question) {
        const lowerQuestion = question.toLowerCase();
        const entities = [];

        if (typeof culinaryGraphData === 'undefined') {
            return entities;
        }

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

    // ============================================
    // FOCUSED CONTEXT EXTRACTORS BY QUERY TYPE
    // ============================================

    // Get focused context for substitution queries
    getSubstitutionContext(entities) {
        if (!entities || entities.length === 0) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        let context = '**Substitution Data**:\n\n';
        let hasData = false;

        entities.forEach(entity => {
            // Get from substitutionRules (detailed) if available
            if (typeof substitutionRules !== 'undefined' && substitutionRules[entity.id]) {
                hasData = true;
                context += `For **${entity.label}**:\n`;
                substitutionRules[entity.id].forEach(rule => {
                    const sub = culinaryGraphData.nodes.find(n => n.id === rule.substitute);
                    context += `- **${sub?.label || rule.substitute}**: ${rule.ratio}`;
                    if (rule.notes) context += ` - ${rule.notes}`;
                    context += '\n';
                    if (rule.dietary && rule.dietary.length > 0) {
                        context += `  (${rule.dietary.join(', ')})\n`;
                    }
                });
                context += '\n';
            }

            // Also check graph edges for substitutions
            const graphSubs = culinaryGraphData.edges.filter(e =>
                e.type === 'substitutes' && e.source === entity.id
            );
            if (graphSubs.length > 0 && !(typeof substitutionRules !== 'undefined' && substitutionRules[entity.id])) {
                hasData = true;
                context += `For **${entity.label}** (from graph):\n`;
                graphSubs.slice(0, 5).forEach(edge => {
                    const target = culinaryGraphData.nodes.find(n => n.id === edge.target);
                    context += `- **${target?.label}**: ${edge.ratio || '1:1'} (${edge.context || 'general use'})\n`;
                });
                context += '\n';
            }
        });

        return hasData ? context : '';
    }

    // Get focused context for pairing queries
    getPairingContext(entities) {
        if (!entities || entities.length === 0) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        let context = '**Pairing Data**:\n\n';
        let hasData = false;

        entities.forEach(entity => {
            const pairings = culinaryGraphData.edges.filter(e =>
                e.type === 'pairs-with' &&
                (e.source === entity.id || e.target === entity.id)
            );

            if (pairings.length > 0) {
                hasData = true;
                const strong = pairings.filter(p => p.strength === 'strong');
                const medium = pairings.filter(p => p.strength === 'medium' || !p.strength);

                context += `**${entity.label}** pairs with:\n`;
                if (strong.length > 0) {
                    const strongNames = strong.slice(0, 5).map(p => {
                        const otherId = p.source === entity.id ? p.target : p.source;
                        return culinaryGraphData.nodes.find(n => n.id === otherId)?.label;
                    }).filter(Boolean);
                    context += `- Strong: ${strongNames.join(', ')}\n`;
                }
                if (medium.length > 0) {
                    const mediumNames = medium.slice(0, 5).map(p => {
                        const otherId = p.source === entity.id ? p.target : p.source;
                        return culinaryGraphData.nodes.find(n => n.id === otherId)?.label;
                    }).filter(Boolean);
                    context += `- Good: ${mediumNames.join(', ')}\n`;
                }
                context += '\n';
            }
        });

        return hasData ? context : '';
    }

    // Get focused context for recipe queries
    getRecipeContext(entities) {
        if (!entities || entities.length === 0) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        let context = '**Recipe Combination Data**:\n\n';
        let hasData = false;

        entities.forEach(entity => {
            const usedWith = culinaryGraphData.edges.filter(e =>
                e.type === 'used-with' &&
                (e.source === entity.id || e.target === entity.id)
            );

            if (usedWith.length > 0) {
                hasData = true;
                context += `**${entity.label}** commonly used with:\n`;
                usedWith.slice(0, 6).forEach(edge => {
                    const otherId = edge.source === entity.id ? edge.target : edge.source;
                    const other = culinaryGraphData.nodes.find(n => n.id === otherId);
                    if (other) {
                        context += `- ${other.label}${edge.context ? ` (${edge.context})` : ''}\n`;
                    }
                });
                context += '\n';
            }
        });

        return hasData ? context : '';
    }

    // Get focused context for dietary queries
    getDietaryContext(dietaryType) {
        if (!dietaryType) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        const filtered = culinaryGraphData.nodes.filter(n =>
            n.dietary && n.dietary.includes(dietaryType)
        );

        if (filtered.length === 0) return '';

        const byCategory = {};
        filtered.forEach(n => {
            if (!byCategory[n.category]) byCategory[n.category] = [];
            byCategory[n.category].push(n.label);
        });

        let context = `**${dietaryType.charAt(0).toUpperCase() + dietaryType.slice(1)} Ingredients** (${filtered.length} total):\n\n`;

        Object.entries(byCategory).forEach(([cat, items]) => {
            const displayItems = items.slice(0, 8);
            context += `- **${cat}**: ${displayItems.join(', ')}`;
            if (items.length > 8) context += ` (+${items.length - 8} more)`;
            context += '\n';
        });

        return context;
    }

    // Get focused context for cuisine queries
    getCuisineContext(cuisineType) {
        if (!cuisineType) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        const specific = culinaryGraphData.nodes.filter(n =>
            n.cuisine && n.cuisine.includes(cuisineType) && !n.cuisine.includes('all')
        );

        if (specific.length === 0) return '';

        const byCategory = {};
        specific.forEach(n => {
            if (!byCategory[n.category]) byCategory[n.category] = [];
            byCategory[n.category].push(n.label);
        });

        let context = `**${cuisineType.charAt(0).toUpperCase() + cuisineType.slice(1)} Cuisine Signature Ingredients**:\n\n`;

        Object.entries(byCategory).forEach(([cat, items]) => {
            context += `- **${cat}**: ${items.slice(0, 6).join(', ')}\n`;
        });

        return context;
    }

    // Get focused context for category queries
    getCategoryContext(categoryType) {
        if (!categoryType) return '';
        if (typeof culinaryGraphData === 'undefined') return '';

        const filtered = culinaryGraphData.nodes.filter(n => n.category === categoryType);
        if (filtered.length === 0) return '';

        let context = `**${categoryType.charAt(0).toUpperCase() + categoryType.slice(1)} Ingredients** (${filtered.length} total):\n\n`;

        filtered.slice(0, 15).forEach(item => {
            context += `- **${item.label}**`;
            if (item.dietary && item.dietary.length > 0) {
                context += ` (${item.dietary.join(', ')})`;
            }
            context += '\n';
        });

        if (filtered.length > 15) {
            context += `- ...and ${filtered.length - 15} more\n`;
        }

        return context;
    }

    // Get general context for unclassified queries
    getGeneralContext(entities) {
        if (typeof culinaryGraphData === 'undefined') {
            return '**Knowledge Base**: Culinary ingredient database available.';
        }

        if (!entities || entities.length === 0) {
            return `**Knowledge Base**: ${culinaryGraphData.nodes.length} ingredients, ${culinaryGraphData.edges.length} relationships across dairy, protein, grain, vegetable, herb, spice, oil, sauce categories.`;
        }

        let context = '**Ingredient Info**:\n';
        entities.slice(0, 3).forEach(entity => {
            const node = culinaryGraphData.nodes.find(n => n.id === entity.id);
            if (node) {
                context += `- **${node.label}**: ${node.category}`;
                if (node.cuisine && node.cuisine.length > 0 && !node.cuisine.includes('all')) {
                    context += `, cuisines: ${node.cuisine.join(', ')}`;
                }
                if (node.dietary && node.dietary.length > 0) {
                    context += `, dietary: ${node.dietary.join(', ')}`;
                }
                context += '\n';
            }
        });

        return context;
    }

    // ============================================
    // MAIN RAG ENHANCEMENT FUNCTION
    // ============================================

    // Get focused RAG context based on query type
    getFocusedContext(queryType, entities) {
        if (!queryType) return this.getGeneralContext(entities);

        switch (queryType.type) {
            case 'substitution':
                return this.getSubstitutionContext(entities);
            case 'pairing':
                return this.getPairingContext(entities);
            case 'recipe':
                return this.getRecipeContext(entities);
            case 'dietary':
                return this.getDietaryContext(queryType.dietaryType);
            case 'cuisine':
                return this.getCuisineContext(queryType.cuisineType);
            case 'category':
                return this.getCategoryContext(queryType.categoryType);
            default:
                return this.getGeneralContext(entities);
        }
    }

    // Get relevant subgraph around entities (legacy, kept for compatibility)
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

    // Enhanced query with focused GraphRAG context
    // Returns { enhancedMessage, ragContext } for use with getContextualSystemPrompt
    enhanceQueryWithGraphContext(userQuestion, queryType = null) {
        try {
            const entities = this.extractEntities(userQuestion);

            if (entities.length > 0) {
                console.log('GraphRAG: Found entities:', entities.map(e => e.label));
            }

            // Get focused context based on query type
            const ragContext = this.getFocusedContext(queryType, entities);

            if (ragContext && ragContext.trim() !== '') {
                console.log('GraphRAG: Focused context added for query type:', queryType?.type || 'general');
            }

            return {
                enhancedMessage: userQuestion,
                ragContext: ragContext,
                entities: entities
            };
        } catch (error) {
            console.error('GraphRAG enhancement error:', error);
            return {
                enhancedMessage: userQuestion,
                ragContext: '',
                entities: []
            };
        }
    }
}

// Export as global
window.CulinaryAIConfig = CulinaryAIConfig;
