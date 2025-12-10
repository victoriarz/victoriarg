// Culinary Substitution Finder
// Smart ingredient substitution based on knowledge graph relationships

(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', function() {
        setupSubstitutionFinder();
    });

    function setupSubstitutionFinder() {
        const searchInput = document.getElementById('ingredientSearch');
        const findButton = document.getElementById('findSubstitute');
        const suggestionPills = document.querySelectorAll('.suggestion-pill');

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

    function findSubstitutes(ingredient) {
        const normalizedIngredient = normalizeIngredient(ingredient);

        // Check if we have substitution rules for this ingredient
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
