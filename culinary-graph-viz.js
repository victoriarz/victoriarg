// Culinary Graph Visualization using Cytoscape.js
// Renders the interactive knowledge graph

(function() {
    'use strict';

    // Wait for DOM to load
    document.addEventListener('DOMContentLoaded', function() {
        initializeGraph();
        setupEventListeners();
    });

    let cy; // Cytoscape instance

    function initializeGraph() {
        // Transform data into Cytoscape format
        const elements = transformDataToCytoscape(culinaryGraphData);

        // Initialize Cytoscape
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
                        'label': 'data(label)',
                        'color': '#3d2e1f',
                        'font-size': '12px',
                        'font-weight': '600',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '60px',
                        'height': '60px',
                        'border-width': '2px',
                        'border-color': '#e8dfd2',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': '4px',
                        'border-color': '#d97642',
                        'background-color': '#f5ede3'
                    }
                },
                {
                    selector: 'node.highlighted',
                    style: {
                        'border-width': '3px',
                        'border-color': '#7fa563'
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
                    selector: 'edge[type="substitutes"]',
                    style: {
                        'line-style': 'dashed'
                    }
                }
            ],

            layout: {
                name: 'cose',
                idealEdgeLength: 100,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: 400000,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            },

            minZoom: 0.5,
            maxZoom: 3,
            wheelSensitivity: 0.2
        });

        // Node click event
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            showNodeInfo(node);
            highlightConnections(node);
        });

        // Background click event (deselect)
        cy.on('tap', function(evt) {
            if (evt.target === cy) {
                hideNodeInfo();
                removeHighlights();
            }
        });
    }

    function transformDataToCytoscape(data) {
        const elements = [];

        // Add nodes
        data.nodes.forEach(node => {
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

        // Add edges
        data.edges.forEach(edge => {
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
                <strong>Cuisines:</strong> ${nodeData.cuisine.join(', ')}
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

    function setupEventListeners() {
        // Filter by cuisine
        document.getElementById('filterCuisine').addEventListener('change', function(e) {
            filterByCuisine(e.target.value);
        });

        // Filter by dietary
        document.getElementById('filterDietary').addEventListener('change', function(e) {
            filterByDietary(e.target.value);
        });

        // Reset graph
        document.getElementById('resetGraph').addEventListener('click', function() {
            resetFilters();
            hideNodeInfo();
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

    function resetFilters() {
        document.getElementById('filterCuisine').value = 'all';
        document.getElementById('filterDietary').value = 'all';
        cy.nodes().show();
        cy.edges().show();
        removeHighlights();
        cy.fit();
    }

})();
