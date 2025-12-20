// Pantry Atlas Engine - Advanced Knowledge Graph Operations
// Implements path-finding, inference, analytics, and recipe generation

(function() {
    'use strict';

    // ============================================
    // GRAPH ANALYTICS & STATISTICS
    // ============================================

    class GraphAnalytics {
        constructor(graphData) {
            this.nodes = graphData.nodes;
            this.edges = graphData.edges;
        }

        // Calculate comprehensive graph statistics
        getStatistics() {
            const stats = {
                nodes: {
                    total: this.nodes.length,
                    byCategory: this.getNodesByCategory(),
                    byCuisine: this.getNodesByCuisine(),
                    byDietary: this.getNodesByDietary()
                },
                edges: {
                    total: this.edges.length,
                    byType: this.getEdgesByType(),
                    avgConnectionsPerNode: (this.edges.length * 2 / this.nodes.length).toFixed(2)
                },
                mostConnected: this.getMostConnectedNodes(10),
                substitutionChains: this.countSubstitutionChains(),
                cuisineOverlap: this.analyzeCuisineOverlap()
            };

            return stats;
        }

        getNodesByCategory() {
            const categories = {};
            this.nodes.forEach(node => {
                categories[node.category] = (categories[node.category] || 0) + 1;
            });
            return categories;
        }

        getNodesByCuisine() {
            const cuisines = {};
            this.nodes.forEach(node => {
                node.cuisine.forEach(c => {
                    cuisines[c] = (cuisines[c] || 0) + 1;
                });
            });
            return cuisines;
        }

        getNodesByDietary() {
            const dietary = {};
            this.nodes.forEach(node => {
                node.dietary.forEach(d => {
                    dietary[d] = (dietary[d] || 0) + 1;
                });
            });
            return dietary;
        }

        getEdgesByType() {
            const types = {};
            this.edges.forEach(edge => {
                types[edge.type] = (types[edge.type] || 0) + 1;
            });
            return types;
        }

        getMostConnectedNodes(limit = 10) {
            const connections = {};

            this.edges.forEach(edge => {
                connections[edge.source] = (connections[edge.source] || 0) + 1;
                connections[edge.target] = (connections[edge.target] || 0) + 1;
            });

            return Object.entries(connections)
                .sort((a, b) => b[1] - a[1])
                .slice(0, limit)
                .map(([id, count]) => {
                    const node = this.nodes.find(n => n.id === id);
                    return { id, label: node?.label, connections: count };
                });
        }

        countSubstitutionChains() {
            const subEdges = this.edges.filter(e => e.type === 'substitutes');
            return subEdges.length;
        }

        analyzeCuisineOverlap() {
            // Find ingredients that appear in multiple cuisines
            const multiCuisine = this.nodes.filter(n =>
                n.cuisine.length > 1 && !n.cuisine.includes('all')
            );
            return {
                count: multiCuisine.length,
                examples: multiCuisine.slice(0, 5).map(n => ({
                    label: n.label,
                    cuisines: n.cuisine
                }))
            };
        }
    }

    // ============================================
    // PATH FINDING & GRAPH TRAVERSAL
    // ============================================

    class PathFinder {
        constructor(graphData) {
            this.nodes = graphData.nodes;
            this.edges = graphData.edges;
            this.adjacencyList = this.buildAdjacencyList();
        }

        buildAdjacencyList() {
            const adj = {};

            this.nodes.forEach(node => {
                adj[node.id] = [];
            });

            this.edges.forEach(edge => {
                if (!adj[edge.source]) adj[edge.source] = [];
                adj[edge.source].push({
                    target: edge.target,
                    type: edge.type,
                    ratio: edge.ratio,
                    context: edge.context,
                    weight: this.calculateEdgeWeight(edge)
                });
            });

            return adj;
        }

        calculateEdgeWeight(edge) {
            // Assign weights based on edge type (lower is better)
            const weights = {
                'substitutes': 1,
                'pairs-with': 2,
                'used-with': 2,
                'similar-flavor': 3,
                'similar-texture': 3,
                'similar-use': 3
            };
            return weights[edge.type] || 5;
        }

        // Find shortest substitution path between two ingredients
        findSubstitutionPath(fromId, toId, maxHops = 4) {
            if (fromId === toId) {
                return { found: false, reason: 'Same ingredient' };
            }

            const queue = [{ id: fromId, path: [fromId], totalWeight: 0 }];
            const visited = new Set();

            while (queue.length > 0) {
                // Sort by weight to prioritize better paths
                queue.sort((a, b) => a.totalWeight - b.totalWeight);
                const current = queue.shift();

                if (current.path.length > maxHops) continue;

                if (current.id === toId) {
                    return {
                        found: true,
                        path: current.path.map(id => this.getNodeLabel(id)),
                        pathIds: current.path,
                        hops: current.path.length - 1,
                        totalWeight: current.totalWeight
                    };
                }

                if (visited.has(current.id)) continue;
                visited.add(current.id);

                const neighbors = this.adjacencyList[current.id] || [];

                // Only follow substitution edges for substitution paths
                neighbors
                    .filter(n => n.type === 'substitutes')
                    .forEach(neighbor => {
                        if (!visited.has(neighbor.target)) {
                            queue.push({
                                id: neighbor.target,
                                path: [...current.path, neighbor.target],
                                totalWeight: current.totalWeight + neighbor.weight
                            });
                        }
                    });
            }

            return { found: false, reason: 'No path found within ' + maxHops + ' hops' };
        }

        // Find all ingredients within N hops
        findNeighborhood(nodeId, hops = 2, edgeTypes = null) {
            const queue = [{ id: nodeId, distance: 0 }];
            const visited = new Map(); // id -> distance
            visited.set(nodeId, 0);

            while (queue.length > 0) {
                const current = queue.shift();

                if (current.distance >= hops) continue;

                const neighbors = this.adjacencyList[current.id] || [];
                neighbors.forEach(neighbor => {
                    // Filter by edge type if specified
                    if (edgeTypes && !edgeTypes.includes(neighbor.type)) {
                        return;
                    }

                    const newDistance = current.distance + 1;

                    if (!visited.has(neighbor.target) || visited.get(neighbor.target) > newDistance) {
                        visited.set(neighbor.target, newDistance);
                        queue.push({
                            id: neighbor.target,
                            distance: newDistance
                        });
                    }
                });
            }

            // Convert to array with node details
            const neighborhood = [];
            visited.forEach((distance, id) => {
                if (id !== nodeId) {
                    const node = this.nodes.find(n => n.id === id);
                    if (node) {
                        neighborhood.push({
                            ...node,
                            distance
                        });
                    }
                }
            });

            return neighborhood;
        }

        getNodeLabel(id) {
            const node = this.nodes.find(n => n.id === id);
            return node ? node.label : id;
        }
    }

    // ============================================
    // INFERENCE ENGINE
    // ============================================

    class InferenceEngine {
        constructor(graphData) {
            this.nodes = graphData.nodes;
            this.edges = graphData.edges;
        }

        // Infer transitive substitutions (if A->B and B->C, then A->C might work)
        inferTransitiveSubstitutions(minConfidence = 0.6) {
            const inferred = [];
            const existingPairs = new Set(
                this.edges
                    .filter(e => e.type === 'substitutes')
                    .map(e => `${e.source}->${e.target}`)
            );

            this.edges
                .filter(e => e.type === 'substitutes')
                .forEach(edge1 => {
                    // Find substitutes of the substitute
                    this.edges
                        .filter(e => e.type === 'substitutes' && e.source === edge1.target)
                        .forEach(edge2 => {
                            const newPair = `${edge1.source}->${edge2.target}`;

                            // Don't infer if already exists or creates self-loop
                            if (!existingPairs.has(newPair) && edge1.source !== edge2.target) {
                                inferred.push({
                                    source: edge1.source,
                                    target: edge2.target,
                                    type: 'substitutes',
                                    inferred: true,
                                    confidence: minConfidence,
                                    via: edge1.target,
                                    note: `Inferred via ${this.getNodeLabel(edge1.target)}`
                                });
                            }
                        });
                });

            return inferred;
        }

        // Infer pairings based on shared substitution patterns
        inferPairings() {
            const inferred = [];
            const substitutionMap = {};

            // Build map of what each ingredient substitutes
            this.edges
                .filter(e => e.type === 'substitutes')
                .forEach(edge => {
                    if (!substitutionMap[edge.target]) {
                        substitutionMap[edge.target] = [];
                    }
                    substitutionMap[edge.target].push(edge.source);
                });

            // If two ingredients substitute the same thing, they might pair well
            Object.entries(substitutionMap).forEach(([target, sources]) => {
                if (sources.length >= 2) {
                    // Check if these sources already have a pairing
                    for (let i = 0; i < sources.length; i++) {
                        for (let j = i + 1; j < sources.length; j++) {
                            const existingPairing = this.edges.find(e =>
                                e.type === 'pairs-with' &&
                                ((e.source === sources[i] && e.target === sources[j]) ||
                                 (e.source === sources[j] && e.target === sources[i]))
                            );

                            if (!existingPairing) {
                                inferred.push({
                                    source: sources[i],
                                    target: sources[j],
                                    type: 'pairs-with',
                                    inferred: true,
                                    strength: 'medium',
                                    confidence: 0.5,
                                    note: `Both substitute ${this.getNodeLabel(target)}`
                                });
                            }
                        }
                    }
                }
            });

            return inferred.slice(0, 20); // Limit to top 20 inferences
        }

        getNodeLabel(id) {
            const node = this.nodes.find(n => n.id === id);
            return node ? node.label : id;
        }
    }

    // ============================================
    // RECIPE GENERATOR
    // ============================================

    class RecipeGenerator {
        constructor(graphData) {
            this.nodes = graphData.nodes;
            this.edges = graphData.edges;
        }

        // Generate recipe suggestions from available ingredients
        generateRecipe(ingredientIds, cuisineType = null, dietary = []) {
            // Find all pairings and combinations among provided ingredients
            const combinations = this.findIngredientCombinations(ingredientIds);

            // Check for balance (protein + grain/starch + vegetables)
            const balance = this.analyzeBalance(ingredientIds);

            // Suggest missing complementary ingredients
            const suggestions = this.suggestComplementaryIngredients(ingredientIds, cuisineType, dietary);

            return {
                providedIngredients: ingredientIds.map(id => this.getNodeLabel(id)),
                balance: balance,
                pairings: combinations,
                suggestions: suggestions,
                possibleDishes: this.inferDishTypes(ingredientIds, balance)
            };
        }

        findIngredientCombinations(ingredientIds) {
            const combinations = [];

            for (let i = 0; i < ingredientIds.length; i++) {
                for (let j = i + 1; j < ingredientIds.length; j++) {
                    const edge = this.edges.find(e =>
                        ((e.source === ingredientIds[i] && e.target === ingredientIds[j]) ||
                         (e.source === ingredientIds[j] && e.target === ingredientIds[i])) &&
                        (e.type === 'pairs-with' || e.type === 'used-with')
                    );

                    if (edge) {
                        combinations.push({
                            ingredient1: this.getNodeLabel(ingredientIds[i]),
                            ingredient2: this.getNodeLabel(ingredientIds[j]),
                            type: edge.type,
                            strength: edge.strength || 'good'
                        });
                    }
                }
            }

            return combinations;
        }

        analyzeBalance(ingredientIds) {
            const categories = {
                protein: false,
                grain: false,
                vegetable: false,
                fat: false,
                aromatic: false
            };

            ingredientIds.forEach(id => {
                const node = this.nodes.find(n => n.id === id);
                if (!node) return;

                if (node.category === 'protein') categories.protein = true;
                if (node.category === 'grain') categories.grain = true;
                if (node.category === 'vegetable') categories.vegetable = true;
                if (node.category === 'oil' || node.category === 'dairy') categories.fat = true;
                if (node.category === 'aromatic' || node.category === 'herb') categories.aromatic = true;
            });

            const balanced = Object.values(categories).filter(v => v).length >= 3;

            return {
                isBalanced: balanced,
                has: categories,
                missing: Object.entries(categories)
                    .filter(([k, v]) => !v)
                    .map(([k]) => k)
            };
        }

        suggestComplementaryIngredients(ingredientIds, cuisineType, dietary) {
            const suggestions = new Set();

            // Find ingredients that pair with what we have
            ingredientIds.forEach(id => {
                const pairings = this.edges.filter(e =>
                    e.type === 'pairs-with' &&
                    (e.source === id || e.target === id)
                );

                pairings.forEach(pairing => {
                    const suggestedId = pairing.source === id ? pairing.target : pairing.source;

                    // Don't suggest what we already have
                    if (ingredientIds.includes(suggestedId)) return;

                    const node = this.nodes.find(n => n.id === suggestedId);
                    if (!node) return;

                    // Filter by cuisine if specified
                    if (cuisineType && !node.cuisine.includes(cuisineType) && !node.cuisine.includes('all')) {
                        return;
                    }

                    // Filter by dietary if specified
                    if (dietary.length > 0) {
                        const meetsRequirements = dietary.every(d => node.dietary.includes(d));
                        if (!meetsRequirements) return;
                    }

                    suggestions.add(JSON.stringify({
                        id: suggestedId,
                        label: node.label,
                        category: node.category,
                        reason: `Pairs with ${this.getNodeLabel(id)}`
                    }));
                });
            });

            return Array.from(suggestions)
                .map(s => JSON.parse(s))
                .slice(0, 8);
        }

        inferDishTypes(ingredientIds, balance) {
            const categories = ingredientIds.map(id => {
                const node = this.nodes.find(n => n.id === id);
                return node?.category;
            });

            const dishes = [];

            // Simple heuristics for dish types
            if (balance.has.protein && balance.has.grain && balance.has.vegetable) {
                dishes.push('Stir-fry', 'Bowl', 'One-pan dinner');
            }
            if (balance.has.protein && balance.has.vegetable && !balance.has.grain) {
                dishes.push('Salad', 'Sauté', 'Roasted dish');
            }
            if (balance.has.grain && balance.has.vegetable && categories.includes('sauce')) {
                dishes.push('Pasta dish', 'Rice bowl');
            }
            if (categories.includes('dairy') && balance.has.grain) {
                dishes.push('Baked dish', 'Casserole');
            }
            if (balance.has.protein && categories.includes('spice')) {
                dishes.push('Curry', 'Stew', 'Braised dish');
            }

            return dishes.length > 0 ? dishes : ['Mixed ingredients dish'];
        }

        getNodeLabel(id) {
            const node = this.nodes.find(n => n.id === id);
            return node ? node.label : id;
        }
    }

    // ============================================
    // PANTRY OPTIMIZER
    // ============================================

    class PantryOptimizer {
        constructor(graphData) {
            this.nodes = graphData.nodes;
            this.edges = graphData.edges;
            this.recipeGenerator = new RecipeGenerator(graphData);
        }

        // Analyze pantry and suggest strategic additions
        optimizePantry(currentIngredients, cuisinePreference = null, dietaryRestrictions = []) {
            // Analyze what we have
            const analysis = this.recipeGenerator.generateRecipe(currentIngredients, cuisinePreference, dietaryRestrictions);

            // Find high-value additions
            const strategicAdditions = this.findStrategicAdditions(currentIngredients, cuisinePreference, dietaryRestrictions);

            // Find versatile ingredients we're missing
            const versatileIngredients = this.findVersatileIngredients(currentIngredients);

            return {
                currentInventory: currentIngredients.length,
                balance: analysis.balance,
                strategicAdditions: strategicAdditions.slice(0, 5),
                versatileIngredients: versatileIngredients.slice(0, 5),
                completableRecipes: analysis.suggestions
            };
        }

        findStrategicAdditions(currentIngredients, cuisinePreference, dietaryRestrictions) {
            const connectionCounts = {};

            // Count how many current ingredients each potential addition connects to
            this.edges
                .filter(e => e.type === 'pairs-with' || e.type === 'used-with')
                .forEach(edge => {
                    const connectedTo = [];

                    if (currentIngredients.includes(edge.source)) {
                        connectedTo.push(edge.source);
                        if (!connectionCounts[edge.target]) {
                            connectionCounts[edge.target] = { count: 0, connections: [] };
                        }
                        connectionCounts[edge.target].count++;
                        connectionCounts[edge.target].connections.push(this.getNodeLabel(edge.source));
                    }

                    if (currentIngredients.includes(edge.target)) {
                        connectedTo.push(edge.target);
                        if (!connectionCounts[edge.source]) {
                            connectionCounts[edge.source] = { count: 0, connections: [] };
                        }
                        connectionCounts[edge.source].count++;
                        connectionCounts[edge.source].connections.push(this.getNodeLabel(edge.target));
                    }
                });

            // Filter and sort by connection count
            return Object.entries(connectionCounts)
                .filter(([id]) => !currentIngredients.includes(id))
                .map(([id, data]) => {
                    const node = this.nodes.find(n => n.id === id);
                    return {
                        id,
                        label: node?.label,
                        category: node?.category,
                        connections: data.count,
                        connectsWith: data.connections.slice(0, 3)
                    };
                })
                .filter(item => {
                    if (!item.label) return false;

                    const node = this.nodes.find(n => n.id === item.id);
                    if (!node) return false;

                    // Apply filters
                    if (cuisinePreference && !node.cuisine.includes(cuisinePreference) && !node.cuisine.includes('all')) {
                        return false;
                    }

                    if (dietaryRestrictions.length > 0) {
                        return dietaryRestrictions.every(d => node.dietary.includes(d));
                    }

                    return true;
                })
                .sort((a, b) => b.connections - a.connections);
        }

        findVersatileIngredients(currentIngredients) {
            // Find ingredients with the most connections overall
            const connectionCounts = {};

            this.edges.forEach(edge => {
                connectionCounts[edge.source] = (connectionCounts[edge.source] || 0) + 1;
                connectionCounts[edge.target] = (connectionCounts[edge.target] || 0) + 1;
            });

            return Object.entries(connectionCounts)
                .filter(([id]) => !currentIngredients.includes(id))
                .map(([id, count]) => {
                    const node = this.nodes.find(n => n.id === id);
                    return {
                        id,
                        label: node?.label,
                        category: node?.category,
                        connections: count
                    };
                })
                .filter(item => item.label)
                .sort((a, b) => b.connections - a.connections);
        }

        getNodeLabel(id) {
            const node = this.nodes.find(n => n.id === id);
            return node ? node.label : id;
        }
    }

    // ============================================
    // EXPORT PUBLIC API
    // ============================================

    window.CulinaryGraphEngine = {
        createAnalytics: (graphData) => new GraphAnalytics(graphData),
        createPathFinder: (graphData) => new PathFinder(graphData),
        createInferenceEngine: (graphData) => new InferenceEngine(graphData),
        createRecipeGenerator: (graphData) => new RecipeGenerator(graphData),
        createPantryOptimizer: (graphData) => new PantryOptimizer(graphData)
    };

    console.log('Pantry Atlas Engine loaded successfully');

})();
