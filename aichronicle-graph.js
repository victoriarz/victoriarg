// ================================================
// THE AI CHRONICLE - Graph Visualization Engine
// Interactive force-directed graph using Canvas
// ================================================

class AIChronicleGraph {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById('graphCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.data = data;
        
        // Graph state
        this.nodes = [];
        this.edges = [];
        this.visibleNodes = new Set();
        this.selectedNode = null;
        this.hoveredNode = null;
        
        // Interaction state
        this.isDragging = false;
        this.dragNode = null;
        this.panOffset = { x: 0, y: 0 };
        this.lastPan = { x: 0, y: 0 };
        this.isPanning = false;
        this.zoom = 1;
        
        // Display settings
        this.showLabels = true;
        this.filterTrending = false;
        this.maxDisplayNodes = 50;
        
        // Physics settings
        this.physics = {
            repulsion: 5000,
            attraction: 0.01,
            damping: 0.85,
            centerGravity: 0.02
        };
        
        // Colors
        this.colors = {
            article: '#e07b53',
            topic: '#5b8a72',
            organization: '#6b8cae',
            model: '#9b7bb8',
            edge: 'rgba(139, 115, 85, 0.3)',
            edgeHighlight: 'rgba(193, 127, 89, 0.8)',
            text: '#4a4a4a',
            background: '#f5f0e1'
        };
        
        // Initialize
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.processData();
        this.setupEventListeners();
        this.startSimulation();
    }
    
    setupCanvas() {
        const resize = () => {
            const rect = this.container.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            
            this.canvas.width = rect.width * dpr;
            this.canvas.height = rect.height * dpr;
            this.canvas.style.width = rect.width + 'px';
            this.canvas.style.height = rect.height + 'px';
            
            this.ctx.scale(dpr, dpr);
            this.width = rect.width;
            this.height = rect.height;
            this.centerX = this.width / 2;
            this.centerY = this.height / 2;
        };
        
        resize();
        window.addEventListener('resize', resize);
    }
    
    processData() {
        // Create node objects with positions
        const nodeMap = new Map();
        
        this.data.nodes.forEach((node, index) => {
            const angle = (index / this.data.nodes.length) * Math.PI * 2;
            const radius = 150 + Math.random() * 100;
            
            const graphNode = {
                ...node,
                x: this.centerX + Math.cos(angle) * radius,
                y: this.centerY + Math.sin(angle) * radius,
                vx: 0,
                vy: 0,
                radius: this.getNodeRadius(node),
                color: this.colors[node.type] || this.colors.topic
            };
            
            nodeMap.set(node.id, graphNode);
            this.nodes.push(graphNode);
        });
        
        // Create edge objects
        this.data.edges.forEach(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            
            if (source && target) {
                this.edges.push({
                    source,
                    target,
                    relationship: edge.relationship
                });
            }
        });
        
        // Initial visible set - most connected and recent
        this.updateVisibleNodes();
    }
    
    getNodeRadius(node) {
        switch (node.type) {
            case 'article':
                return 12 + (node.trendingScore || 50) / 20;
            case 'topic':
                return 18 + (node.connectionCount || 0) * 1.5;
            case 'organization':
                return 16 + (node.connectionCount || 0);
            case 'model':
                return 14 + (node.connectionCount || 0);
            default:
                return 12;
        }
    }
    
    updateVisibleNodes() {
        this.visibleNodes.clear();
        
        // Score nodes by importance
        const scoredNodes = this.nodes.map(node => {
            let score = 0;
            
            // Recency bonus for articles
            if (node.type === 'article' && node.date) {
                const daysSince = (Date.now() - new Date(node.date)) / (1000 * 60 * 60 * 24);
                score += Math.max(0, 10 - daysSince) * 5;
            }
            
            // Trending score
            if (node.trendingScore) {
                score += node.trendingScore;
            }
            
            // Connection count
            const connections = this.edges.filter(
                e => e.source.id === node.id || e.target.id === node.id
            ).length;
            score += connections * 10;
            
            return { node, score };
        });
        
        // Sort and take top N
        scoredNodes.sort((a, b) => b.score - a.score);
        
        const topNodes = scoredNodes.slice(0, this.maxDisplayNodes);
        topNodes.forEach(({ node }) => this.visibleNodes.add(node.id));
        
        // Also add connected nodes
        const initialVisible = new Set(this.visibleNodes);
        initialVisible.forEach(nodeId => {
            this.edges.forEach(edge => {
                if (edge.source.id === nodeId) {
                    this.visibleNodes.add(edge.target.id);
                }
                if (edge.target.id === nodeId) {
                    this.visibleNodes.add(edge.source.id);
                }
            });
        });
        
        // Trim if too many
        if (this.visibleNodes.size > this.maxDisplayNodes) {
            const visible = Array.from(this.visibleNodes);
            this.visibleNodes = new Set(visible.slice(0, this.maxDisplayNodes));
        }
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('click', this.onClick.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseLeave.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left - this.panOffset.x) / this.zoom,
            y: (e.clientY - rect.top - this.panOffset.y) / this.zoom
        };
    }
    
    findNodeAt(pos) {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i];
            if (!this.visibleNodes.has(node.id)) continue;
            
            const dx = pos.x - node.x;
            const dy = pos.y - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < node.radius + 5) {
                return node;
            }
        }
        return null;
    }
    
    onMouseDown(e) {
        const pos = this.getMousePos(e);
        const node = this.findNodeAt(pos);
        
        if (node) {
            this.isDragging = true;
            this.dragNode = node;
            node.fx = node.x;
            node.fy = node.y;
        } else {
            this.isPanning = true;
            this.lastPan = { x: e.clientX, y: e.clientY };
        }
    }
    
    onMouseMove(e) {
        const pos = this.getMousePos(e);
        
        if (this.isDragging && this.dragNode) {
            this.dragNode.x = pos.x;
            this.dragNode.y = pos.y;
            this.dragNode.fx = pos.x;
            this.dragNode.fy = pos.y;
        } else if (this.isPanning) {
            this.panOffset.x += e.clientX - this.lastPan.x;
            this.panOffset.y += e.clientY - this.lastPan.y;
            this.lastPan = { x: e.clientX, y: e.clientY };
        } else {
            const node = this.findNodeAt(pos);
            if (node !== this.hoveredNode) {
                this.hoveredNode = node;
                this.canvas.style.cursor = node ? 'pointer' : 'grab';
            }
        }
    }
    
    onMouseUp(e) {
        if (this.dragNode) {
            delete this.dragNode.fx;
            delete this.dragNode.fy;
        }
        this.isDragging = false;
        this.dragNode = null;
        this.isPanning = false;
    }
    
    onClick(e) {
        if (this.isDragging) return;
        
        const pos = this.getMousePos(e);
        const node = this.findNodeAt(pos);
        
        if (node) {
            this.selectNode(node);
        } else {
            this.selectNode(null);
        }
    }
    
    onWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.3, Math.min(3, this.zoom * delta));
        
        // Zoom toward mouse position
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        this.panOffset.x = mouseX - (mouseX - this.panOffset.x) * (newZoom / this.zoom);
        this.panOffset.y = mouseY - (mouseY - this.panOffset.y) * (newZoom / this.zoom);
        
        this.zoom = newZoom;
    }
    
    onMouseLeave() {
        this.hoveredNode = null;
        this.canvas.style.cursor = 'grab';
    }
    
    onTouchStart(e) {
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            this.onMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }
    
    onTouchMove(e) {
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            this.onMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }
    
    onTouchEnd(e) {
        this.onMouseUp(e);
    }
    
    selectNode(node) {
        this.selectedNode = node;
        
        // Dispatch custom event for UI updates
        const event = new CustomEvent('nodeSelected', { detail: node });
        document.dispatchEvent(event);
        
        // Expand connections if node selected
        if (node) {
            this.expandNode(node);
        }
    }
    
    expandNode(node) {
        // Add connected nodes to visible set
        this.edges.forEach(edge => {
            if (edge.source.id === node.id) {
                this.visibleNodes.add(edge.target.id);
            }
            if (edge.target.id === node.id) {
                this.visibleNodes.add(edge.source.id);
            }
        });
    }
    
    startSimulation() {
        const simulate = () => {
            this.updatePhysics();
            this.render();
            requestAnimationFrame(simulate);
        };
        simulate();
    }
    
    updatePhysics() {
        const visibleNodesList = this.nodes.filter(n => this.visibleNodes.has(n.id));
        
        // Apply forces
        visibleNodesList.forEach(node => {
            if (node.fx !== undefined) {
                node.x = node.fx;
                node.y = node.fy;
                return;
            }
            
            // Repulsion from other nodes
            visibleNodesList.forEach(other => {
                if (node === other) return;
                
                const dx = node.x - other.x;
                const dy = node.y - other.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const force = this.physics.repulsion / (dist * dist);
                
                node.vx += (dx / dist) * force * 0.01;
                node.vy += (dy / dist) * force * 0.01;
            });
            
            // Attraction along edges
            this.edges.forEach(edge => {
                if (!this.visibleNodes.has(edge.source.id) || !this.visibleNodes.has(edge.target.id)) return;
                
                let other = null;
                if (edge.source === node) other = edge.target;
                if (edge.target === node) other = edge.source;
                if (!other) return;
                
                const dx = other.x - node.x;
                const dy = other.y - node.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                
                node.vx += dx * this.physics.attraction;
                node.vy += dy * this.physics.attraction;
            });
            
            // Center gravity
            const dx = this.centerX - node.x;
            const dy = this.centerY - node.y;
            node.vx += dx * this.physics.centerGravity * 0.01;
            node.vy += dy * this.physics.centerGravity * 0.01;
            
            // Apply velocity with damping
            node.vx *= this.physics.damping;
            node.vy *= this.physics.damping;
            
            node.x += node.vx;
            node.y += node.vy;
            
            // Keep in bounds
            const padding = 50;
            node.x = Math.max(padding, Math.min(this.width - padding, node.x));
            node.y = Math.max(padding, Math.min(this.height - padding, node.y));
        });
    }
    
    render() {
        // Clear canvas
        this.ctx.save();
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        const dpr = window.devicePixelRatio || 1;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.restore();
        
        // Apply transforms
        this.ctx.save();
        this.ctx.translate(this.panOffset.x, this.panOffset.y);
        this.ctx.scale(this.zoom, this.zoom);
        
        // Draw edges
        this.renderEdges();
        
        // Draw nodes
        this.renderNodes();
        
        // Draw labels
        if (this.showLabels) {
            this.renderLabels();
        }
        
        this.ctx.restore();
    }
    
    renderEdges() {
        this.edges.forEach(edge => {
            if (!this.visibleNodes.has(edge.source.id) || !this.visibleNodes.has(edge.target.id)) return;
            
            const isHighlighted = 
                this.selectedNode && 
                (edge.source === this.selectedNode || edge.target === this.selectedNode);
            
            this.ctx.beginPath();
            this.ctx.moveTo(edge.source.x, edge.source.y);
            this.ctx.lineTo(edge.target.x, edge.target.y);
            
            this.ctx.strokeStyle = isHighlighted ? this.colors.edgeHighlight : this.colors.edge;
            this.ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
            this.ctx.stroke();
        });
    }
    
    renderNodes() {
        this.nodes.forEach(node => {
            if (!this.visibleNodes.has(node.id)) return;
            
            const isSelected = node === this.selectedNode;
            const isHovered = node === this.hoveredNode;
            const isConnected = this.selectedNode && this.edges.some(
                e => (e.source === this.selectedNode && e.target === node) ||
                     (e.target === this.selectedNode && e.source === node)
            );
            
            // Node shadow
            if (isSelected || isHovered) {
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
                this.ctx.fillStyle = 'rgba(193, 127, 89, 0.3)';
                this.ctx.fill();
            }
            
            // Node circle
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            
            // Gradient fill
            const gradient = this.ctx.createRadialGradient(
                node.x - node.radius * 0.3, node.y - node.radius * 0.3, 0,
                node.x, node.y, node.radius
            );
            gradient.addColorStop(0, this.lightenColor(node.color, 30));
            gradient.addColorStop(1, node.color);
            
            this.ctx.fillStyle = gradient;
            this.ctx.fill();
            
            // Border
            this.ctx.strokeStyle = isSelected ? '#c17f59' : this.darkenColor(node.color, 20);
            this.ctx.lineWidth = isSelected ? 3 : 2;
            this.ctx.stroke();
            
            // Node icon
            this.renderNodeIcon(node);
        });
    }
    
    renderNodeIcon(node) {
        const icons = {
            article: '📰',
            topic: '🏷️',
            organization: '🏢',
            model: '🤖'
        };
        
        const icon = icons[node.type] || '●';
        
        this.ctx.font = `${node.radius * 0.8}px sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(icon, node.x, node.y);
    }
    
    renderLabels() {
        this.ctx.font = '11px "VT323", monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        
        this.nodes.forEach(node => {
            if (!this.visibleNodes.has(node.id)) return;
            
            const isSelected = node === this.selectedNode;
            const isHovered = node === this.hoveredNode;
            
            if (!isSelected && !isHovered && this.zoom < 0.8) return;
            
            const label = this.truncateText(node.title, 20);
            const y = node.y + node.radius + 5;
            
            // Background
            const metrics = this.ctx.measureText(label);
            this.ctx.fillStyle = 'rgba(245, 240, 225, 0.9)';
            this.ctx.fillRect(
                node.x - metrics.width / 2 - 3,
                y - 2,
                metrics.width + 6,
                14
            );
            
            // Text
            this.ctx.fillStyle = this.colors.text;
            this.ctx.fillText(label, node.x, y);
        });
    }
    
    truncateText(text, maxLen) {
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen - 3) + '...';
    }
    
    lightenColor(hex, percent) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = Math.min(255, (num >> 16) + amt);
        const G = Math.min(255, ((num >> 8) & 0x00FF) + amt);
        const B = Math.min(255, (num & 0x0000FF) + amt);
        return `rgb(${R}, ${G}, ${B})`;
    }
    
    darkenColor(hex, percent) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = Math.max(0, (num >> 16) - amt);
        const G = Math.max(0, ((num >> 8) & 0x00FF) - amt);
        const B = Math.max(0, (num & 0x0000FF) - amt);
        return `rgb(${R}, ${G}, ${B})`;
    }
    
    // Public methods for controls
    resetView() {
        this.panOffset = { x: 0, y: 0 };
        this.zoom = 1;
        this.updateVisibleNodes();
    }
    
    toggleLabels() {
        this.showLabels = !this.showLabels;
        return this.showLabels;
    }
    
    setFilter(options) {
        // Apply filters and update visible nodes
        this.updateVisibleNodes();
    }
    
    getStats() {
        return {
            totalNodes: this.nodes.length,
            visibleNodes: this.visibleNodes.size,
            totalEdges: this.edges.length,
            articles: this.nodes.filter(n => n.type === 'article').length
        };
    }
    
    getConnectedNodes(node) {
        const connected = [];
        this.edges.forEach(edge => {
            if (edge.source.id === node.id) {
                connected.push({ node: edge.target, relationship: edge.relationship });
            }
            if (edge.target.id === node.id) {
                connected.push({ node: edge.source, relationship: edge.relationship });
            }
        });
        return connected;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIChronicleGraph;
}
