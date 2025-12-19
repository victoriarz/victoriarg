// Culinary Graph Visualization using Cytoscape.js
// Renders the interactive knowledge graph
// Enhanced with tooltips, stats, sidebar panels, and loading screen

(function() {
    'use strict';

    // Wait for DOM to load
    document.addEventListener('DOMContentLoaded', function() {
        initializeGraph();
        setupEventListeners();
        populateStats();
        populateSidebar();
    });

    let cy; // Cytoscape instance
    let graphEngine = null; // Graph engine for advanced operations

    // Define 20 main cooking ingredients to show by default
    const mainIngredients = [
        'butter', 'eggs', 'milk', 'flour', 'sugar', 'salt', 'black-pepper',
        'garlic', 'onion', 'tomato', 'chicken-breast', 'ground-beef', 'rice',
        'pasta', 'olive-oil', 'cheese-parmesan', 'basil', 'lemon-juice',
        'soy-sauce', 'honey'
    ];

    // Track which ingredients have been revealed through searches
    let revealedIngredients = new Set(mainIngredients);

    // Drag tracking for click-to-show-details vs drag-to-move
    let dragStartPos = null;
    let hasDragged = false;
    const DRAG_THRESHOLD = 5; // pixels - movement below this triggers details panel

    // Detect if user is on mobile device
    function isMobile() {
        return window.innerWidth <= 768;
    }

    // Get responsive node size
    function getNodeSize() {
        if (window.innerWidth <= 480) return '45px';
        if (window.innerWidth <= 768) return '50px';
        return '60px';
    }

    // Get responsive font size
    function getFontSize() {
        if (window.innerWidth <= 480) return '10px';
        if (window.innerWidth <= 768) return '11px';
        return '12px';
    }

    // Get responsive layout parameters
    function getLayoutParams() {
        const isMobileDevice = isMobile();
        return {
            name: 'cose',
            idealEdgeLength: isMobileDevice ? 100 : 120,      // More space between nodes
            nodeOverlap: isMobileDevice ? 25 : 30,            // Better overlap handling
            refresh: 20,
            fit: true,
            padding: isMobileDevice ? 30 : 50,                // More padding around edges
            randomize: false,
            componentSpacing: isMobileDevice ? 120 : 150,     // More space between disconnected groups
            nodeRepulsion: isMobileDevice ? 450000 : 600000,  // Stronger repulsion to reduce crowding
            edgeElasticity: 80,                               // Slightly more elastic edges
            nestingFactor: 5,
            gravity: 60,                                      // Less gravity = looser, more spread layout
            numIter: 1000,
            initialTemp: 200,
            coolingFactor: 0.95,
            minTemp: 1.0
        };
    }

    function initializeGraph() {
        // Check if required dependencies are loaded
        if (typeof cytoscape === 'undefined') {
            console.error('Cytoscape.js library not loaded');
            showError('Failed to load graph library. Please refresh the page.');
            return;
        }

        if (typeof culinaryGraphData === 'undefined') {
            console.error('Culinary graph data not loaded');
            showError('Failed to load graph data. Please refresh the page.');
            return;
        }

        console.log('Initializing graph with', culinaryGraphData.nodes.length, 'nodes and', culinaryGraphData.edges.length, 'edges');

        // Initialize graph engine for advanced operations
        if (typeof CulinaryGraphEngine !== 'undefined') {
            graphEngine = {
                analytics: CulinaryGraphEngine.createAnalytics(culinaryGraphData),
                pathFinder: CulinaryGraphEngine.createPathFinder(culinaryGraphData),
                inferenceEngine: CulinaryGraphEngine.createInferenceEngine(culinaryGraphData),
                recipeGenerator: CulinaryGraphEngine.createRecipeGenerator(culinaryGraphData),
                pantryOptimizer: CulinaryGraphEngine.createPantryOptimizer(culinaryGraphData)
            };
            console.log('Graph engine initialized successfully');
        } else {
            console.warn('Graph engine not available - advanced features disabled');
        }

        // Transform data into Cytoscape format
        const elements = transformDataToCytoscape(culinaryGraphData);
        console.log('Transformed elements:', elements.length);

        // Hide loading spinner
        const loadingEl = document.querySelector('.graph-loading');
        if (loadingEl) {
            loadingEl.style.display = 'none';
        }

        // Initialize Cytoscape
        try {
            cy = cytoscape({
                container: document.getElementById('graphContainer'),

                elements: elements,

            style: [
                // Node styles
                {
                    selector: 'node',
                    style: {
                        'background-color': function(ele) {
                            return getCategoryColor(ele.data('category'));
                        },
                        'background-opacity': 0.9,
                        'label': 'data(label)',
                        'color': '#3d2e1f',
                        'font-size': getFontSize(),
                        'font-weight': '600',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': getNodeSize(),
                        'height': getNodeSize(),
                        'border-width': '2px',
                        'border-color': function(ele) {
                            // Darker border based on category color
                            const color = getCategoryColor(ele.data('category'));
                            return darkenColor(color, 20);
                        },
                        'text-wrap': 'wrap',
                        'text-max-width': '80px',
                        'overlay-padding': '6px',
                        'overlay-opacity': 0,
                        'transition-property': 'background-opacity, border-width, overlay-opacity, width, height, opacity',
                        'transition-duration': '0.15s'
                    }
                },
                {
                    selector: 'node:active',
                    style: {
                        'overlay-opacity': 0.2,
                        'overlay-color': '#d97642'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': '4px',
                        'border-color': '#d97642',
                        'background-opacity': 1,
                        'overlay-opacity': 0.15,
                        'overlay-color': '#d97642',
                        'z-index': 999
                    }
                },
                {
                    selector: 'node.highlighted',
                    style: {
                        'border-width': '3px',
                        'border-color': '#7fa563',
                        'background-opacity': 1,
                        'overlay-opacity': 0.1,
                        'overlay-color': '#7fa563'
                    }
                },
                {
                    selector: 'node.dimmed',
                    style: {
                        'opacity': 0.3
                    }
                },
                {
                    selector: 'node.hovered',
                    style: {
                        'border-width': '3px',
                        'border-color': '#c17f59',
                        'background-opacity': 1,
                        'overlay-opacity': 0.12,
                        'overlay-color': '#c17f59',
                        'z-index': 998
                    }
                },
                // Edge styles
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': function(ele) {
                            return getEdgeColor(ele.data('type'));
                        },
                        'target-arrow-color': function(ele) {
                            return getEdgeColor(ele.data('type'));
                        },
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.6
                    }
                },
                {
                    selector: 'edge.highlighted',
                    style: {
                        'width': 3,
                        'opacity': 1,
                        'line-color': '#d97642',
                        'target-arrow-color': '#d97642'
                    }
                },
                {
                    selector: 'edge.dimmed',
                    style: {
                        'opacity': 0.1,
                        'transition-property': 'opacity',
                        'transition-duration': '0.15s'
                    }
                },
                {
                    selector: 'edge[type="substitutes"]',
                    style: {
                        'line-style': 'dashed',
                        'line-dash-pattern': [6, 3]
                    }
                },
                {
                    selector: 'edge[type="pairs-with"]',
                    style: {
                        'line-style': 'solid',
                        'width': 3,
                        'opacity': 0.7
                    }
                },
                {
                    selector: 'edge[type="used-with"]',
                    style: {
                        'line-style': 'dotted',
                        'line-dash-pattern': [2, 4]
                    }
                }
            ],

            minZoom: 0.5,
            maxZoom: 3,
            wheelSensitivity: 0.2
        });

            console.log('Graph initialized successfully with', cy.nodes().length, 'nodes');

            // Run layout and wait for completion before focusing on connected nodes
            const layout = cy.layout(getLayoutParams());

            layout.on('layoutstop', function() {
                // Focus view on connected ingredients after layout completes
                showConnectedIngredients();

                // Staggered entry animation - nodes fade in sequentially
                const visibleNodes = cy.nodes().filter(n => n.visible());
                visibleNodes.forEach((node, i) => {
                    node.style('opacity', 0);
                    setTimeout(() => {
                        node.animate({
                            style: { opacity: 1 },
                            duration: 200,
                            easing: 'ease-out'
                        });
                    }, i * 12); // 12ms stagger
                });

                // Edges fade in after nodes
                const visibleEdges = cy.edges().filter(e => e.visible());
                visibleEdges.style('opacity', 0);
                setTimeout(() => {
                    visibleEdges.animate({
                        style: { opacity: 0.6 },
                        duration: 300,
                        easing: 'ease-out'
                    });
                }, visibleNodes.length * 12 + 100);
            });

            layout.run();

            // Node click event
            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                showEnhancedNodeInfo(node);
                highlightConnections(node);
            });

            // Background click event (deselect)
            cy.on('tap', function(evt) {
                if (evt.target === cy) {
                    removeHighlights();
                    // Reset details panel
                    const detailsContent = document.getElementById('ingredientDetailsContent');
                    if (detailsContent) {
                        detailsContent.innerHTML = '<p class="placeholder-text">Click any ingredient to explore its substitutions, pairings, and dietary info</p>';
                    }
                }
            });

            // Drag tracking events for click-to-show-details while allowing repositioning
            cy.on('grabon', 'node', function(evt) {
                const node = evt.target;
                dragStartPos = { ...node.position() };
                hasDragged = false;
            });

            cy.on('drag', 'node', function(evt) {
                if (!dragStartPos) return;
                const node = evt.target;
                const pos = node.position();
                const dx = Math.abs(pos.x - dragStartPos.x);
                const dy = Math.abs(pos.y - dragStartPos.y);
                if (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD) {
                    hasDragged = true;
                }
            });

            cy.on('free', 'node', function(evt) {
                // Show details only if node wasn't dragged significantly
                if (!hasDragged) {
                    const node = evt.target;
                    showEnhancedNodeInfo(node);
                    highlightConnections(node);
                }
                dragStartPos = null;
                hasDragged = false;
            });

            // Setup hover tooltips
            setupTooltips();

        } catch (error) {
            console.error('Error initializing graph:', error);
            showError('Failed to initialize graph visualization: ' + error.message);
        }
    }

    function showError(message) {
        const loadingEl = document.querySelector('.graph-loading');
        if (loadingEl) {
            loadingEl.innerHTML = `
                <div style="color: #d97642; padding: 20px;">
                    <p style="font-weight: bold;">⚠️ Error Loading Graph</p>
                    <p style="font-size: 14px;">${message}</p>
                </div>
            `;
        }
    }

    function transformDataToCytoscape(data) {
        const elements = [];
        const nodeIds = new Set();

        // Add nodes first
        data.nodes.forEach(node => {
            nodeIds.add(node.id);
            elements.push({
                data: {
                    id: node.id,
                    label: node.label,
                    category: node.category,
                    cuisine: node.cuisine,
                    dietary: node.dietary
                }
            });
        });

        console.log('Node IDs:', Array.from(nodeIds));

        // Add edges - validate that both source and target exist
        const invalidEdges = [];
        data.edges.forEach(edge => {
            if (!nodeIds.has(edge.source)) {
                invalidEdges.push(`Missing source: ${edge.source} in edge ${edge.source} -> ${edge.target}`);
                return;
            }
            if (!nodeIds.has(edge.target)) {
                invalidEdges.push(`Missing target: ${edge.target} in edge ${edge.source} -> ${edge.target}`);
                return;
            }

            elements.push({
                data: {
                    source: edge.source,
                    target: edge.target,
                    type: edge.type,
                    ratio: edge.ratio,
                    context: edge.context,
                    strength: edge.strength,
                    note: edge.note
                }
            });
        });

        if (invalidEdges.length > 0) {
            console.error('Invalid edges found:', invalidEdges);
            console.error('These edges reference nodes that do not exist in the graph data.');
        }

        console.log(`Added ${data.nodes.length} nodes and ${elements.length - data.nodes.length} valid edges (skipped ${invalidEdges.length} invalid)`);

        return elements;
    }

    function getCategoryColor(category) {
        const colors = {
            'dairy': '#e8f0e3',
            'dairy-alt': '#dcfce7',
            'protein': '#fce7f3',
            'grain': '#fef3c7',
            'sweetener': '#fee2e2',
            'aromatic': '#dbeafe',
            'herb': '#d4f5d4',
            'sauce': '#ffe4e1',
            'oil': '#fff8dc',
            'vegetable': '#98fb98'
        };
        return colors[category] || '#f5ede3';
    }

    // Darken a hex color by a percentage
    function darkenColor(hex, percent) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = Math.max(0, (num >> 16) - amt);
        const G = Math.max(0, ((num >> 8) & 0x00FF) - amt);
        const B = Math.max(0, (num & 0x0000FF) - amt);
        return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
    }

    function getEdgeColor(type) {
        const colors = {
            'substitutes': '#d97642',
            'pairs-with': '#7fa563',
            'used-with': '#6b5d52',
            'similar-flavor': '#9b8b7e',
            'similar-texture': '#9b8b7e',
            'similar-use': '#9b8b7e'
        };
        return colors[type] || '#e8dfd2';
    }

    function showNodeInfo(node) {
        const nodeData = node.data();
        const nodeInfoPanel = document.getElementById('nodeInfo');
        const nodeTitle = document.getElementById('nodeTitle');
        const nodeDetails = document.getElementById('nodeDetails');

        nodeTitle.textContent = nodeData.label;

        // Build details HTML
        let detailsHTML = `
            <div class="node-detail-item">
                <strong>Category:</strong> ${formatCategory(nodeData.category)}
            </div>
            <div class="node-detail-item">
                <strong>Common in:</strong> ${nodeData.cuisine.join(', ')}
            </div>
        `;

        if (nodeData.dietary && nodeData.dietary.length > 0) {
            detailsHTML += `
                <div class="node-detail-item">
                    <strong>Dietary:</strong> ${nodeData.dietary.map(d => `<span class="dietary-badge">${d}</span>`).join(' ')}
                </div>
            `;
        }

        // Get connected nodes
        const connectedEdges = node.connectedEdges();
        const substitutes = [];
        const pairsWith = [];

        connectedEdges.forEach(edge => {
            const edgeData = edge.data();
            const targetNode = edge.target().id() === nodeData.id ? edge.source() : edge.target();
            const targetLabel = targetNode.data('label');

            if (edgeData.type === 'substitutes' && edge.source().id() === nodeData.id) {
                substitutes.push({ label: targetLabel, ratio: edgeData.ratio, context: edgeData.context });
            } else if (edgeData.type === 'pairs-with') {
                pairsWith.push({ label: targetLabel, strength: edgeData.strength });
            }
        });

        if (substitutes.length > 0) {
            detailsHTML += `
                <div class="node-detail-section">
                    <strong>Can be substituted with:</strong>
                    <ul class="substitutes-list">
                        ${substitutes.map(sub => `
                            <li>
                                <span class="substitute-name">${sub.label}</span>
                                ${sub.ratio ? `<span class="substitute-ratio">(${sub.ratio})</span>` : ''}
                                ${sub.context ? `<div class="substitute-context">Use in: ${sub.context}</div>` : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        if (pairsWith.length > 0) {
            detailsHTML += `
                <div class="node-detail-section">
                    <strong>Pairs well with:</strong>
                    <ul class="pairs-list">
                        ${pairsWith.map(pair => `
                            <li>
                                ${pair.label}
                                ${pair.strength ? `<span class="pair-strength ${pair.strength}">${pair.strength}</span>` : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        nodeDetails.innerHTML = detailsHTML;
        nodeInfoPanel.style.display = 'block';
    }

    function hideNodeInfo() {
        document.getElementById('nodeInfo').style.display = 'none';
    }

    function highlightConnections(node) {
        // Remove previous highlights
        removeHighlights();

        // Highlight connected nodes and edges
        const connectedEdges = node.connectedEdges();
        const connectedNodes = node.neighborhood('node');

        connectedEdges.addClass('highlighted');
        connectedNodes.addClass('highlighted');
    }

    function removeHighlights() {
        cy.elements().removeClass('highlighted');
    }

    function formatCategory(category) {
        return category.split('-').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    // Track current category filter
    let currentCategoryFilter = null;

    function setupEventListeners() {
        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', function() {
            zoomIn();
        });

        document.getElementById('zoomOut').addEventListener('click', function() {
            zoomOut();
        });

        // Reset graph
        document.getElementById('resetGraph').addEventListener('click', function() {
            resetFilters();
            // Reset details panel
            const detailsContent = document.getElementById('ingredientDetailsContent');
            if (detailsContent) {
                detailsContent.innerHTML = '<p class="placeholder-text">Click any ingredient to explore its substitutions, pairings, and dietary info</p>';
            }
        });

        // Show All button
        const showAllBtn = document.getElementById('showAllBtn');
        if (showAllBtn) {
            showAllBtn.addEventListener('click', function() {
                cy.nodes().show();
                cy.edges().show();
                currentCategoryFilter = null;
                clearLegendActive();
                cy.fit(null, 30);
            });
        }

        // Search box functionality
        const searchInput = document.getElementById('graphSearch');
        if (searchInput) {
            let searchTimeout;
            searchInput.addEventListener('input', function(e) {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    searchIngredients(e.target.value.trim());
                }, 200);
            });

            // Clear search on Escape
            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    this.value = '';
                    searchIngredients('');
                    this.blur();
                }
            });
        }

        // Clickable legend for category filtering
        const legendItems = document.querySelectorAll('.legend-item-compact[data-category]');
        legendItems.forEach(item => {
            item.addEventListener('click', function() {
                const category = this.dataset.category;
                toggleCategoryFilter(category, this);
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (!cy) return;
            // Don't capture if user is typing in search
            if (document.activeElement === searchInput) return;

            switch(e.key) {
                case 'Escape':
                    resetFilters();
                    break;
                case 'a':
                case 'A':
                    // Show all
                    cy.nodes().show();
                    cy.edges().show();
                    currentCategoryFilter = null;
                    clearLegendActive();
                    cy.fit(null, 30);
                    break;
                case '/':
                    // Focus search
                    e.preventDefault();
                    if (searchInput) searchInput.focus();
                    break;
            }
        });

        // Toggle labels button
        const toggleLabelsBtn = document.getElementById('toggleLabels');
        if (toggleLabelsBtn) {
            toggleLabelsBtn.addEventListener('click', function() {
                if (!cy) return;
                const currentFontSize = cy.style().selector('node').style('font-size');
                if (currentFontSize === '0px' || currentFontSize === 0) {
                    // Show labels
                    cy.style()
                        .selector('node')
                        .style({ 'font-size': getFontSize() })
                        .update();
                    this.classList.add('active');
                } else {
                    // Hide labels
                    cy.style()
                        .selector('node')
                        .style({ 'font-size': '0px' })
                        .update();
                    this.classList.remove('active');
                }
            });
        }

        // Handle window resize for responsive graph
        let resizeTimeout;
        window.addEventListener('resize', function() {
            // Debounce resize events
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(function() {
                if (cy) {
                    // Update node sizes and font sizes
                    cy.style()
                        .selector('node')
                        .style({
                            'width': getNodeSize(),
                            'height': getNodeSize(),
                            'font-size': getFontSize()
                        })
                        .update();

                    // Refit the graph to the container
                    cy.fit(null, 30);
                }
            }, 250);
        });
    }

    function filterByCuisine(cuisine) {
        if (cuisine === 'all') {
            cy.nodes().show();
            cy.edges().show();
            return;
        }

        cy.nodes().forEach(node => {
            const nodeCuisines = node.data('cuisine');
            if (nodeCuisines.includes('all') || nodeCuisines.includes(cuisine)) {
                node.show();
            } else {
                node.hide();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        cy.fit();
    }

    function filterByDietary(dietary) {
        if (dietary === 'all') {
            cy.nodes().show();
            cy.edges().show();
            return;
        }

        cy.nodes().forEach(node => {
            const nodeDietary = node.data('dietary');
            if (nodeDietary && nodeDietary.includes(dietary)) {
                node.show();
            } else {
                node.hide();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        cy.fit();
    }

    function zoomIn() {
        if (!cy) return;
        const currentZoom = cy.zoom();
        const newZoom = currentZoom * 1.2; // Zoom in by 20%
        cy.animate({
            zoom: newZoom,
            duration: 300
        });
    }

    function zoomOut() {
        if (!cy) return;
        const currentZoom = cy.zoom();
        const newZoom = currentZoom / 1.2; // Zoom out by 20%
        cy.animate({
            zoom: newZoom,
            duration: 300
        });
    }

    function resetFilters() {
        cy.nodes().show();
        cy.edges().show();
        removeHighlights();
        currentCategoryFilter = null;
        clearLegendActive();
        // Clear search input
        const searchInput = document.getElementById('graphSearch');
        if (searchInput) searchInput.value = '';
        cy.fit(null, 30);
    }

    function clearLegendActive() {
        document.querySelectorAll('.legend-item-compact').forEach(item => {
            item.classList.remove('active');
        });
    }

    function toggleCategoryFilter(category, element) {
        // If clicking the same category, clear the filter
        if (currentCategoryFilter === category) {
            currentCategoryFilter = null;
            clearLegendActive();
            cy.nodes().show();
            cy.edges().show();
            cy.fit(null, 30);
            return;
        }

        // Set new filter
        currentCategoryFilter = category;
        clearLegendActive();
        element.classList.add('active');

        // Filter nodes by category
        cy.nodes().forEach(node => {
            const nodeCategory = node.data('category');
            // Match category or similar (e.g., 'dairy' matches 'dairy-alt')
            if (nodeCategory === category || nodeCategory.startsWith(category + '-') || category.startsWith(nodeCategory + '-')) {
                node.show();
            } else {
                node.hide();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        cy.fit(null, 30);
    }

    function searchIngredients(query) {
        if (!cy) return;

        if (!query) {
            // Show all when search is empty
            cy.nodes().show();
            cy.edges().show();
            removeHighlights();
            cy.fit(null, 30);
            return;
        }

        const lowerQuery = query.toLowerCase();
        const matchingNodes = [];

        // Find matching nodes
        cy.nodes().forEach(node => {
            const label = (node.data('label') || '').toLowerCase();
            const id = (node.data('id') || '').toLowerCase();
            if (label.includes(lowerQuery) || id.includes(lowerQuery)) {
                node.show();
                node.addClass('highlighted');
                matchingNodes.push(node);
            } else {
                node.hide();
                node.removeClass('highlighted');
            }
        });

        console.log('Search for "' + query + '" found', matchingNodes.length, 'matches');

        // Show edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        // Zoom to matching nodes
        if (matchingNodes.length > 0) {
            const collection = cy.collection(matchingNodes);

            // Check for exact match first
            const exactMatch = matchingNodes.find(node => {
                const label = (node.data('label') || '').toLowerCase();
                const id = (node.data('id') || '').toLowerCase();
                return label === lowerQuery || id === lowerQuery;
            });

            if (exactMatch) {
                // Center and zoom to the exact match
                cy.nodes().unselect();
                exactMatch.select();
                cy.center(exactMatch);
                cy.zoom({ level: 1.5, position: exactMatch.position() });

                // Show its details and highlight connections
                showEnhancedNodeInfo(exactMatch);
                highlightConnections(exactMatch);
            } else {
                // Fit to all matching nodes
                cy.fit(collection, 50);
            }
        }
    }

    function applyMainIngredientsFilter() {
        if (!cy) return;

        // Hide all nodes first
        cy.nodes().hide();

        // Show only main ingredients
        mainIngredients.forEach(ingredientId => {
            const node = cy.getElementById(ingredientId);
            if (node.length > 0) {
                node.show();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        cy.fit();
        console.log('Applied main ingredients filter. Showing', mainIngredients.length, 'main ingredients');
    }

    function showConnectedIngredients() {
        if (!cy) return;

        // Calculate connection counts for each node
        const connectionCounts = {};
        cy.edges().forEach(edge => {
            const sourceId = edge.source().id();
            const targetId = edge.target().id();
            connectionCounts[sourceId] = (connectionCounts[sourceId] || 0) + 1;
            connectionCounts[targetId] = (connectionCounts[targetId] || 0) + 1;
        });

        // Get nodes sorted by connection count
        const sortedNodes = cy.nodes().toArray()
            .map(node => ({ node, count: connectionCounts[node.id()] || 0 }))
            .filter(item => item.count >= 2) // Only nodes with 2+ connections
            .sort((a, b) => b.count - a.count)
            .slice(0, 50); // Top 50 most connected

        const highlightedIds = new Set(sortedNodes.map(item => item.node.id()));

        // Hide nodes that aren't highly connected
        cy.nodes().forEach(node => {
            if (highlightedIds.has(node.id())) {
                node.show();
            } else {
                node.hide();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        // Fit view to visible nodes
        const visibleNodes = cy.nodes().filter(node => node.visible());
        if (visibleNodes.length > 0) {
            cy.fit(visibleNodes, 40);
        }

        console.log('Showing top', highlightedIds.size, 'most-connected ingredients');
    }

    function revealIngredientInGraph(ingredientId) {
        if (!cy) {
            console.log('Graph not initialized yet');
            return;
        }

        const normalizedId = ingredientId.toLowerCase().trim();

        // Scroll to graph section first
        const graphSection = document.getElementById('graph-demo');
        if (graphSection) {
            setTimeout(() => {
                graphSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
        }

        // Check if ingredient already revealed
        if (revealedIngredients.has(normalizedId)) {
            console.log('Ingredient already visible:', normalizedId);
            // Focus on the ingredient
            const node = cy.getElementById(normalizedId);
            if (node.length > 0) {
                setTimeout(() => {
                    cy.animate({
                        center: { eles: node },
                        zoom: 1.5,
                        duration: 500
                    });
                }, 800);
            }
            return;
        }

        // Find the node
        const node = cy.getElementById(normalizedId);
        if (node.length === 0) {
            console.log('Ingredient not found in graph:', normalizedId);
            return;
        }

        // Show the searched ingredient
        node.show();
        revealedIngredients.add(normalizedId);

        // Get all connected nodes and show them
        const connectedNodes = node.neighborhood('node');
        connectedNodes.forEach(connectedNode => {
            connectedNode.show();
            revealedIngredients.add(connectedNode.id());
        });

        // Show edges between visible nodes
        const connectedEdges = node.connectedEdges();
        connectedEdges.forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            }
        });

        // Animate to focus on the newly revealed ingredient (after scroll completes)
        setTimeout(() => {
            cy.animate({
                center: { eles: node },
                zoom: 1.5,
                duration: 500
            });
        }, 800);

        console.log('Revealed ingredient and connections:', normalizedId);
    }

    // Expose function globally for substitution finder to call
    window.revealIngredientInGraph = revealIngredientInGraph;

    // ============================================
    // GRAPH STATISTICS & ANALYTICS
    // ============================================

    function displayGraphStatistics() {
        if (!graphEngine || !graphEngine.analytics) {
            console.warn('Graph analytics not available');
            return;
        }

        const stats = graphEngine.analytics.getStatistics();

        console.log('=== CULINARY GRAPH STATISTICS ===');
        console.log('Total Nodes:', stats.nodes.total);
        console.log('Total Edges:', stats.edges.total);
        console.log('Avg Connections per Node:', stats.edges.avgConnectionsPerNode);
        console.log('\nNodes by Category:', stats.nodes.byCategory);
        console.log('\nMost Connected Ingredients:', stats.mostConnected);
        console.log('\nSubstitution Chains:', stats.substitutionChains);
        console.log('\nCuisine Overlap:', stats.cuisineOverlap);

        return stats;
    }

    function findIngredientPath(fromIngredient, toIngredient) {
        if (!graphEngine || !graphEngine.pathFinder) {
            console.warn('Path finder not available');
            return null;
        }

        const fromId = fromIngredient.toLowerCase().replace(/\s+/g, '-');
        const toId = toIngredient.toLowerCase().replace(/\s+/g, '-');

        const path = graphEngine.pathFinder.findSubstitutionPath(fromId, toId);

        if (path.found) {
            console.log(`Path found from ${fromIngredient} to ${toIngredient}:`);
            console.log('Path:', path.path.join(' → '));
            console.log('Hops:', path.hops);
        } else {
            console.log(`No path found: ${path.reason}`);
        }

        return path;
    }

    function inferNewRelationships() {
        if (!graphEngine || !graphEngine.inferenceEngine) {
            console.warn('Inference engine not available');
            return null;
        }

        const transitiveSubstitutions = graphEngine.inferenceEngine.inferTransitiveSubstitutions();
        const inferredPairings = graphEngine.inferenceEngine.inferPairings();

        console.log('=== INFERRED RELATIONSHIPS ===');
        console.log('Transitive Substitutions:', transitiveSubstitutions.length);
        console.log('Sample:', transitiveSubstitutions.slice(0, 5));
        console.log('\nInferred Pairings:', inferredPairings.length);
        console.log('Sample:', inferredPairings.slice(0, 5));

        return {
            substitutions: transitiveSubstitutions,
            pairings: inferredPairings
        };
    }

    function generateRecipeFromIngredients(ingredientNames, cuisine = null, dietary = []) {
        if (!graphEngine || !graphEngine.recipeGenerator) {
            console.warn('Recipe generator not available');
            return null;
        }

        const ingredientIds = ingredientNames.map(name =>
            name.toLowerCase().replace(/\s+/g, '-')
        );

        const recipe = graphEngine.recipeGenerator.generateRecipe(ingredientIds, cuisine, dietary);

        console.log('=== RECIPE GENERATION ===');
        console.log('Provided Ingredients:', recipe.providedIngredients);
        console.log('Balance:', recipe.balance);
        console.log('Pairings Found:', recipe.pairings);
        console.log('Suggestions:', recipe.suggestions);
        console.log('Possible Dishes:', recipe.possibleDishes);

        return recipe;
    }

    function optimizePantry(ingredientNames, cuisine = null, dietary = []) {
        if (!graphEngine || !graphEngine.pantryOptimizer) {
            console.warn('Pantry optimizer not available');
            return null;
        }

        const ingredientIds = ingredientNames.map(name =>
            name.toLowerCase().replace(/\s+/g, '-')
        );

        const optimization = graphEngine.pantryOptimizer.optimizePantry(ingredientIds, cuisine, dietary);

        console.log('=== PANTRY OPTIMIZATION ===');
        console.log('Current Inventory:', optimization.currentInventory, 'ingredients');
        console.log('Balance:', optimization.balance);
        console.log('\nStrategic Additions:');
        optimization.strategicAdditions.forEach((item, i) => {
            console.log(`${i+1}. ${item.label} (${item.connections} connections)`);
            console.log('   Connects with:', item.connectsWith.join(', '));
        });
        console.log('\nVersatile Ingredients to Consider:');
        optimization.versatileIngredients.forEach((item, i) => {
            console.log(`${i+1}. ${item.label} (${item.connections} total connections)`);
        });

        return optimization;
    }

    // Expose advanced functions globally
    window.displayGraphStatistics = displayGraphStatistics;
    window.findIngredientPath = findIngredientPath;
    window.inferNewRelationships = inferNewRelationships;
    window.generateRecipeFromIngredients = generateRecipeFromIngredients;
    window.optimizePantry = optimizePantry;
    window.getCulinaryGraphEngine = () => graphEngine;

    // ============================================
    // STATS BAR POPULATION
    // ============================================

    function populateStats() {
        if (typeof culinaryGraphData === 'undefined') return;

        const data = culinaryGraphData;

        // Count ingredients
        const ingredientCountEl = document.getElementById('ingredientCount');
        if (ingredientCountEl) ingredientCountEl.textContent = data.nodes.length;

        // Count connections
        const connectionCountEl = document.getElementById('connectionCount');
        if (connectionCountEl) connectionCountEl.textContent = data.edges.length;

        // Count substitutions
        const substitutionCountEl = document.getElementById('substitutionCount');
        if (substitutionCountEl) {
            const substitutionCount = data.edges.filter(e => e.type === 'substitutes').length;
            substitutionCountEl.textContent = substitutionCount;
        }

        // Count unique cuisines
        const cuisineCountEl = document.getElementById('cuisineCount');
        if (cuisineCountEl) {
            const cuisines = new Set();
            data.nodes.forEach(node => {
                if (node.cuisine) {
                    node.cuisine.forEach(c => {
                        if (c !== 'all') cuisines.add(c);
                    });
                }
            });
            cuisineCountEl.textContent = cuisines.size;
        }
    }

    // ============================================
    // SIDEBAR POPULATION
    // ============================================

    function populateSidebar() {
        if (typeof culinaryGraphData === 'undefined') return;

        populateTrendingIngredients();
        populateCategories();
    }

    function populateTrendingIngredients() {
        const data = culinaryGraphData;

        // Count connections per ingredient
        const connectionCounts = {};
        data.nodes.forEach(node => {
            connectionCounts[node.id] = 0;
        });

        data.edges.forEach(edge => {
            if (connectionCounts[edge.source] !== undefined) {
                connectionCounts[edge.source]++;
            }
            if (connectionCounts[edge.target] !== undefined) {
                connectionCounts[edge.target]++;
            }
        });

        // Sort by connection count
        const sorted = Object.entries(connectionCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

        // Find node labels
        const nodeMap = {};
        data.nodes.forEach(node => {
            nodeMap[node.id] = node;
        });

        // Populate the trending list
        const trendingContainer = document.querySelector('#trendingIngredients .trending-list');
        if (!trendingContainer) return;

        trendingContainer.innerHTML = sorted.map(([id, count], index) => {
            const node = nodeMap[id];
            return `
                <div class="trending-item" data-ingredient="${id}">
                    <span class="trending-rank">#${index + 1}</span>
                    <span class="trending-name">${node?.label || id}</span>
                    <span class="trending-count">${count} links</span>
                </div>
            `;
        }).join('');

        // Add click handlers
        trendingContainer.querySelectorAll('.trending-item').forEach(item => {
            item.addEventListener('click', () => {
                const ingredientId = item.dataset.ingredient;
                if (window.revealIngredientInGraph) {
                    window.revealIngredientInGraph(ingredientId);
                }
            });
        });
    }

    function populateCategories() {
        const data = culinaryGraphData;

        // Count ingredients per category
        const categoryCounts = {};
        data.nodes.forEach(node => {
            const cat = node.category || 'other';
            categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
        });

        // Sort by count
        const sorted = Object.entries(categoryCounts)
            .sort((a, b) => b[1] - a[1]);

        // Populate the categories list
        const categoriesContainer = document.querySelector('#categoriesList .category-list');
        if (!categoriesContainer) return;

        categoriesContainer.innerHTML = sorted.map(([category, count]) => {
            const displayName = category.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            return `
                <div class="category-item" data-category="${category}">
                    <span class="category-name">${displayName}</span>
                    <span class="category-count">${count}</span>
                </div>
            `;
        }).join('');

        // Add click handlers to filter by category
        categoriesContainer.querySelectorAll('.category-item').forEach(item => {
            item.addEventListener('click', () => {
                const category = item.dataset.category;
                filterByCategory(category);
            });
        });
    }

    function filterByCategory(category) {
        if (!cy) return;

        // Show only nodes of this category
        cy.nodes().forEach(node => {
            if (node.data('category') === category) {
                node.show();
            } else {
                node.hide();
            }
        });

        // Show only edges between visible nodes
        cy.edges().forEach(edge => {
            if (edge.source().visible() && edge.target().visible()) {
                edge.show();
            } else {
                edge.hide();
            }
        });

        cy.fit();
    }

    // ============================================
    // HOVER TOOLTIPS
    // ============================================

    function setupTooltips() {
        if (!cy) return;

        const tooltip = document.getElementById('graphTooltip');
        const edgeTooltip = document.getElementById('edgeTooltip');
        if (!tooltip) return;

        // Node hover tooltip and glow effect
        cy.on('mouseover', 'node', function(evt) {
            const node = evt.target;
            const nodeData = node.data();
            const dietary = nodeData.dietary && nodeData.dietary.length > 0
                ? nodeData.dietary.join(', ')
                : 'No restrictions';

            // Add hover glow effect
            node.addClass('hovered');

            // Dim unconnected edges for focus effect
            const connectedEdges = node.connectedEdges();
            cy.edges().not(connectedEdges).addClass('dimmed');

            tooltip.innerHTML = `
                <div class="tooltip-category">${formatCategory(nodeData.category)}</div>
                <div class="tooltip-name">${nodeData.label}</div>
                <div class="tooltip-dietary">${dietary}</div>
            `;

            tooltip.classList.add('visible');

            // Position tooltip near the node
            const position = node.renderedPosition();
            const container = document.getElementById('graphContainer');
            const containerRect = container.getBoundingClientRect();

            let tooltipX = position.x + 20;
            let tooltipY = position.y - 10;

            // Keep tooltip within bounds
            if (tooltipX + 200 > containerRect.width) {
                tooltipX = position.x - 220;
            }
            if (tooltipY < 10) {
                tooltipY = 10;
            }

            tooltip.style.left = tooltipX + 'px';
            tooltip.style.top = tooltipY + 'px';
        });

        cy.on('mouseout', 'node', function(evt) {
            evt.target.removeClass('hovered');
            tooltip.classList.remove('visible');
            // Remove edge dimming
            cy.edges().removeClass('dimmed');
        });

        cy.on('drag', 'node', function() {
            tooltip.classList.remove('visible');
        });

        // Edge hover tooltip
        if (edgeTooltip) {
            cy.on('mouseover', 'edge', function(evt) {
                const edge = evt.target;
                const edgeData = edge.data();
                const sourceLabel = edge.source().data('label');
                const targetLabel = edge.target().data('label');

                let typeLabel = '';
                let detailText = '';

                switch(edgeData.type) {
                    case 'substitutes':
                        typeLabel = '🔄 Substitution';
                        detailText = edgeData.ratio || '1:1';
                        if (edgeData.context) detailText += ` (${edgeData.context})`;
                        break;
                    case 'pairs-with':
                        typeLabel = '💚 Pairs With';
                        detailText = edgeData.strength ? edgeData.strength.toUpperCase() : 'Compatible';
                        break;
                    case 'used-with':
                        typeLabel = '🍳 Used Together';
                        detailText = edgeData.context || 'Common pairing';
                        break;
                    default:
                        typeLabel = 'Connection';
                        detailText = edgeData.type || '';
                }

                edgeTooltip.innerHTML = `
                    <div class="edge-type">${typeLabel}</div>
                    <div class="edge-detail">${detailText}</div>
                `;

                edgeTooltip.classList.add('visible');

                // Position tooltip at edge midpoint
                const midpoint = edge.midpoint();
                const container = document.getElementById('graphContainer');
                const containerRect = container.getBoundingClientRect();

                let tooltipX = midpoint.x + 10;
                let tooltipY = midpoint.y - 30;

                if (tooltipX + 150 > containerRect.width) {
                    tooltipX = midpoint.x - 160;
                }
                if (tooltipY < 10) {
                    tooltipY = midpoint.y + 10;
                }

                edgeTooltip.style.left = tooltipX + 'px';
                edgeTooltip.style.top = tooltipY + 'px';
            });

            cy.on('mouseout', 'edge', function() {
                edgeTooltip.classList.remove('visible');
            });
        }
    }

    // Graph-to-Chat integration: populate chat with suggestion when clicking a node
    function populateChatWithIngredient(ingredientName) {
        const chatInput = document.getElementById('chatInput');
        const chatSection = document.getElementById('culinary-chat-demo');

        // Scroll to chat section first
        if (chatSection) {
            chatSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Then populate and focus the input
        if (chatInput) {
            chatInput.value = `What can I substitute for ${ingredientName}?`;
            // Small delay to ensure scroll completes before focus
            setTimeout(() => {
                chatInput.focus();
                chatInput.dispatchEvent(new Event('input', { bubbles: true }));
            }, 300);
        }
    }

    // Expose for use by showEnhancedNodeInfo
    window.populateChatWithIngredient = populateChatWithIngredient;

    // ============================================
    // ENHANCED NODE INFO DISPLAY
    // ============================================

    // Helper function to render strength dots
    function renderStrengthDots(strength) {
        const level = strength || 'medium';
        return `<span class="strength-dots" data-strength="${level}" title="${level} pairing">
            <span class="strength-dot"></span>
            <span class="strength-dot"></span>
            <span class="strength-dot"></span>
        </span>`;
    }

    // Helper function to render dietary badges with data attributes
    function renderDietaryBadges(dietary) {
        if (!dietary || dietary.length === 0) {
            return '<span class="dietary-badge" data-dietary="none">No restrictions</span>';
        }
        return dietary.map(d => `<span class="dietary-badge" data-dietary="${d}">${d}</span>`).join(' ');
    }

    function showEnhancedNodeInfo(node) {
        const nodeData = node.data();
        const detailsPanel = document.getElementById('ingredientDetailsPanel');
        const detailsContent = document.getElementById('ingredientDetailsContent');

        if (!detailsContent) {
            // Fall back to original showNodeInfo
            showNodeInfo(node);
            return;
        }

        const dietary = renderDietaryBadges(nodeData.dietary);
        const cuisines = nodeData.cuisine && nodeData.cuisine.length > 0
            ? nodeData.cuisine.join(', ')
            : 'All cuisines';

        // Get connected nodes
        const connectedEdges = node.connectedEdges();
        const substitutes = [];
        const pairsWith = [];
        const usedWith = [];

        connectedEdges.forEach(edge => {
            const edgeData = edge.data();
            const otherNode = edge.source().id() === nodeData.id ? edge.target() : edge.source();
            const otherLabel = otherNode.data('label');

            if (edgeData.type === 'substitutes') {
                if (edge.source().id() === nodeData.id) {
                    substitutes.push({ label: otherLabel, ratio: edgeData.ratio, context: edgeData.context });
                }
            } else if (edgeData.type === 'pairs-with') {
                pairsWith.push({ label: otherLabel, strength: edgeData.strength });
            } else if (edgeData.type === 'used-with') {
                usedWith.push({ label: otherLabel, context: edgeData.context });
            }
        });

        let html = `
            <div class="node-info-enhanced">
                <div class="node-info-header-row">
                    <div class="node-info-title-group">
                        <h4 class="node-info-title">${nodeData.label}</h4>
                        <p class="node-info-category">${formatCategory(nodeData.category)}</p>
                    </div>
                    <div class="node-info-meta-inline">
                        <span><strong>Common in:</strong> ${cuisines}</span>
                        <span><strong>Dietary:</strong> ${dietary}</span>
                    </div>
                </div>
        `;

        if (substitutes.length > 0) {
            html += `
                <div class="node-info-section">
                    <h5>Substitutions</h5>
                    <ul>
                        ${substitutes.map(s => `
                            <li>
                                <strong>${s.label}</strong>
                                ${s.ratio ? `<span class="ratio">(${s.ratio})</span>` : ''}
                                ${s.context ? `<span class="context">for ${s.context}</span>` : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        if (pairsWith.length > 0) {
            html += `
                <div class="node-info-section-inline">
                    <span class="section-label">Pairs Well With:</span>
                    <span class="section-items">${pairsWith.map(p => `${p.label}${renderStrengthDots(p.strength)}`).join(', ')}</span>
                </div>
            `;
        }

        if (usedWith.length > 0) {
            html += `
                <div class="node-info-section-inline">
                    <span class="section-label">Used Together:</span>
                    <span class="section-items">${usedWith.map(u => `${u.label}${u.context ? ` <span class="context">in ${u.context}</span>` : ''}`).join(', ')}</span>
                </div>
            `;
        }

        // Add "Ask about this" button for chat integration
        html += `
            <div class="node-info-actions">
                <button class="ask-chat-btn" onclick="window.populateChatWithIngredient('${nodeData.label.replace(/'/g, "\\'")}')">
                    Ask about ${nodeData.label}
                </button>
            </div>
        `;

        html += '</div>';
        detailsContent.innerHTML = html;

        // Scroll to details panel so users can see the ingredient info
        setTimeout(() => {
            detailsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 150);
    }

    // Override the original showNodeInfo to use enhanced version
    const originalShowNodeInfo = showNodeInfo;

})();
