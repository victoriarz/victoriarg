// Culinary Substitution Finder with AI Enhancement
// Smart ingredient substitution based on knowledge graph relationships + AI

(function() {
    'use strict';

    let culinaryAiConfig;
    let useAI = true;

    document.addEventListener('DOMContentLoaded', async function() {
        // Initialize AI configuration
        if (typeof CulinaryAIConfig !== 'undefined') {
            culinaryAiConfig = new CulinaryAIConfig();
            const isHealthy = await culinaryAiConfig.checkBackendHealth();
            useAI = isHealthy;
            console.log(`Substitution Finder AI: ${isHealthy ? 'Active' : 'Local Mode'}`);
        }

        setupSubstitutionFinder();
    });

    function setupSubstitutionFinder() {
        const searchInput = document.getElementById('ingredientSearch');
        const findButton = document.getElementById('findSubstitute');
        const suggestionPills = document.querySelectorAll('.suggestion-pill');

        // Skip setup if elements don't exist (substitution finder was removed)
        if (!findButton || !searchInput) {
            console.log('Substitution finder elements not found - skipping setup');
            return;
        }

        // Search button click
        findButton.addEventListener('click', function() {
            const ingredient = searchInput.value.trim().toLowerCase();
            if (ingredient) {
                findSubstitutes(ingredient);
            }
        });

        // Enter key in search input
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const ingredient = searchInput.value.trim().toLowerCase();
                if (ingredient) {
                    findSubstitutes(ingredient);
                }
            }
        });

        // Suggestion pill clicks
        suggestionPills.forEach(pill => {
            pill.addEventListener('click', function() {
                const ingredient = this.getAttribute('data-ingredient');
                searchInput.value = ingredient;
                findSubstitutes(ingredient);
            });
        });
    }

    async function findSubstitutes(ingredient) {
        const normalizedIngredient = normalizeIngredient(ingredient);

        // Reveal the ingredient in the knowledge graph
        if (typeof window.revealIngredientInGraph === 'function') {
            window.revealIngredientInGraph(normalizedIngredient);
        }

        // If AI is available, use it for enhanced substitution recommendations
        if (useAI && culinaryAiConfig) {
            try {
                await findSubstitutesWithAI(ingredient, normalizedIngredient);
                return;
            } catch (error) {
                console.error('AI substitution failed, falling back to local:', error);
                // Fall through to local mode
            }
        }

        // Local mode: Check if we have substitution rules for this ingredient
        if (substitutionRules[normalizedIngredient]) {
            displaySubstitutionResults(normalizedIngredient, substitutionRules[normalizedIngredient]);
        } else {
            // Try to find in graph data
            const graphSubstitutes = findSubstitutesInGraph(normalizedIngredient);
            if (graphSubstitutes.length > 0) {
                displayGraphSubstitutes(normalizedIngredient, graphSubstitutes);
            } else {
                displayNoResults(ingredient);
            }
        }
    }

    // AI-enhanced substitution finder
    async function findSubstitutesWithAI(originalIngredient, normalizedIngredient) {
        // Show loading state
        const resultsContainer = document.getElementById('substitutionResults');
        const substitutesList = document.getElementById('substitutesList');
        resultsContainer.style.display = 'block';
        substitutesList.innerHTML = `
            <div class="ai-loading">
                <div class="loading-spinner"></div>
                <p>Finding best substitutions for ${originalIngredient}...</p>
            </div>
        `;

        // Get local data first for context
        const localRules = substitutionRules[normalizedIngredient] || [];
        const graphSubs = findSubstitutesInGraph(normalizedIngredient);

        // Build context for AI
        let contextInfo = '';
        if (localRules.length > 0 || graphSubs.length > 0) {
            contextInfo = '\n\nKnowledge Base Context:\n';
            if (localRules.length > 0) {
                contextInfo += `Available substitutes in database: ${localRules.map(r => r.substitute).join(', ')}\n`;
            }
            if (graphSubs.length > 0) {
                contextInfo += `Graph relationships: ${graphSubs.map(s => s.substitute).join(', ')}\n`;
            }
        }

        // Create prompt for AI
        const userMessage = `Find substitutes for **${originalIngredient}**. Provide 3-5 best substitutes with:
1. Substitute name
2. Ratio (e.g., "1:1" or "1:0.75")
3. Brief notes on best use case
4. Dietary tags if applicable (vegan, gluten-free, etc.)

Format as a clear, scannable list.${contextInfo}`;

        try {
            // Call AI
            const response = await callGeminiForSubstitution(userMessage);

            // Display AI results
            displayAISubstitutionResults(originalIngredient, response);
        } catch (error) {
            throw error; // Let caller handle fallback
        }
    }

    // Call Gemini AI for substitution
    async function callGeminiForSubstitution(userMessage) {
        const messages = [
            { role: 'system', content: culinaryAiConfig.getSystemPrompt() },
            { role: 'user', content: userMessage }
        ];

        // Convert to Gemini format
        const contents = [];
        for (let i = 0; i < messages.length; i++) {
            const msg = messages[i];
            if (msg.role === 'system') continue;
            contents.push({
                role: msg.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: msg.content }]
            });
        }

        // Prepend system message to first user message
        if (contents.length > 0 && messages[0].role === 'system') {
            contents[0].parts[0].text = messages[0].content + '\n\n' + contents[0].parts[0].text;
        }

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
                        maxOutputTokens: 600
                    }
                })
            }
        );

        if (!response.ok) {
            throw new Error(`Backend API error: ${response.status}`);
        }

        const data = await response.json();

        if (data.candidates && data.candidates[0]) {
            const candidate = data.candidates[0];
            if (candidate.content && candidate.content.parts && candidate.content.parts[0]) {
                return candidate.content.parts[0].text;
            }
            if (candidate.content && candidate.content.text) {
                return candidate.content.text;
            }
        }

        throw new Error('Invalid response format from Gemini API');
    }

    // Display AI-generated substitution results
    function displayAISubstitutionResults(ingredient, aiResponse) {
        const resultsContainer = document.getElementById('substitutionResults');
        const substitutesList = document.getElementById('substitutesList');

        // Configure marked for markdown rendering if available
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
        }

        // Render markdown response
        let renderedHTML = aiResponse;
        if (typeof marked !== 'undefined') {
            renderedHTML = marked.parse(aiResponse);
        } else {
            // Simple fallback
            renderedHTML = aiResponse
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
        }

        const html = `
            <div class="ai-results">
                <div class="ai-provider-badge" style="margin-bottom: 15px;">Gemini 2.5 Flash</div>
                <div class="original-ingredient">
                    <h4>Substitutes for: <span class="ingredient-name">${capitalizeFirst(ingredient)}</span></h4>
                </div>
                <div class="ai-response markdown-content">
                    ${renderedHTML}
                </div>
            </div>
        `;

        substitutesList.innerHTML = html;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function normalizeIngredient(ingredient) {
        // Normalize common ingredient names
        const normalizations = {
            'egg': 'eggs',
            'all-purpose flour': 'flour',
            'white sugar': 'sugar',
            'cow milk': 'milk',
            'cow\'s milk': 'milk',
            'regular milk': 'milk',
            'unsalted butter': 'butter',
            'salted butter': 'butter'
        };

        const lower = ingredient.toLowerCase().trim();
        return normalizations[lower] || lower;
    }

    function findSubstitutesInGraph(ingredientId) {
        const substitutes = [];

        culinaryGraphData.edges.forEach(edge => {
            if (edge.type === 'substitutes' && edge.source === ingredientId) {
                const targetNode = culinaryGraphData.nodes.find(n => n.id === edge.target);
                if (targetNode) {
                    substitutes.push({
                        substitute: targetNode.label,
                        ratio: edge.ratio || '1:1',
                        context: edge.context || 'general use',
                        dietary: targetNode.dietary || []
                    });
                }
            }
        });

        return substitutes;
    }

    function displaySubstitutionResults(ingredient, substitutes) {
        const resultsContainer = document.getElementById('substitutionResults');
        const substitutesList = document.getElementById('substitutesList');

        let html = `
            <div class="original-ingredient">
                <h4>Substitutes for: <span class="ingredient-name">${capitalizeFirst(ingredient)}</span></h4>
            </div>
        `;

        substitutes.forEach((sub, index) => {
            const confidenceClass = sub.confidence === 'high' ? 'high' : 'medium';
            const dietaryBadges = sub.dietary.map(d => `<span class="dietary-badge">${d}</span>`).join(' ');

            html += `
                <div class="substitute-card">
                    <div class="substitute-header">
                        <span class="substitute-number">${index + 1}</span>
                        <h5 class="substitute-title">${sub.substitute}</h5>
                        <span class="confidence-badge ${confidenceClass}">${sub.confidence} confidence</span>
                    </div>
                    <div class="substitute-ratio">
                        <strong>Ratio:</strong> ${sub.ratio}
                    </div>
                    <div class="substitute-notes">
                        ${sub.notes}
                    </div>
                    ${sub.dietary.length > 0 ? `
                        <div class="substitute-dietary">
                            ${dietaryBadges}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        substitutesList.innerHTML = html;
        resultsContainer.style.display = 'block';

        // Smooth scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function displayGraphSubstitutes(ingredient, substitutes) {
        const resultsContainer = document.getElementById('substitutionResults');
        const substitutesList = document.getElementById('substitutesList');

        let html = `
            <div class="original-ingredient">
                <h4>Substitutes for: <span class="ingredient-name">${capitalizeFirst(ingredient)}</span></h4>
            </div>
        `;

        substitutes.forEach((sub, index) => {
            const dietaryBadges = sub.dietary.map(d => `<span class="dietary-badge">${d}</span>`).join(' ');

            html += `
                <div class="substitute-card">
                    <div class="substitute-header">
                        <span class="substitute-number">${index + 1}</span>
                        <h5 class="substitute-title">${sub.substitute}</h5>
                    </div>
                    <div class="substitute-ratio">
                        <strong>Ratio:</strong> ${sub.ratio}
                    </div>
                    <div class="substitute-notes">
                        Best used in: ${sub.context}
                    </div>
                    ${sub.dietary.length > 0 ? `
                        <div class="substitute-dietary">
                            ${dietaryBadges}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        substitutesList.innerHTML = html;
        resultsContainer.style.display = 'block';

        // Smooth scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function displayNoResults(ingredient) {
        const resultsContainer = document.getElementById('substitutionResults');
        const substitutesList = document.getElementById('substitutesList');

        substitutesList.innerHTML = `
            <div class="no-results">
                <h4>No substitutes found for "${ingredient}"</h4>
                <p>Try one of these common ingredients:</p>
                <ul>
                    <li>Butter</li>
                    <li>Eggs</li>
                    <li>Milk</li>
                    <li>Flour</li>
                    <li>Sugar</li>
                </ul>
                <p class="tip">💡 <strong>Tip:</strong> Use the graph visualization above to explore ingredient relationships!</p>
            </div>
        `;

        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

})();
