// ================================================
// THE AI CHRONICLE - Main Application Controller
// Handles UI interactions and data binding
// ================================================

class AIChronicleApp {
    constructor() {
        this.graph = null;
        this.data = AIChronicleData;
        
        this.init();
    }
    
    init() {
        // Show loading screen
        this.showLoading();
        
        // Initialize after DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.onReady());
        } else {
            this.onReady();
        }
    }
    
    showLoading() {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.classList.remove('hidden');
        }
    }
    
    hideLoading() {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            setTimeout(() => {
                loadingScreen.classList.add('hidden');
            }, 2000); // Wait for loading animation
        }
    }
    
    onReady() {
        // Initialize graph
        this.graph = new AIChronicleGraph('graphContainer', this.data);
        
        // Setup UI
        this.setupClock();
        this.updateStats();
        this.populateTrending();
        this.populateRecentArticles();
        this.setupControls();
        this.setupFilters();
        this.setupNodeSelection();
        
        // Hide loading
        this.hideLoading();
    }
    
    setupClock() {
        const updateTime = () => {
            const now = new Date();
            const hours = now.getHours();
            const minutes = now.getMinutes().toString().padStart(2, '0');
            const ampm = hours >= 12 ? 'PM' : 'AM';
            const displayHours = (hours % 12 || 12).toString().padStart(2, '0');
            
            const timeDisplay = document.getElementById('timeDisplay');
            if (timeDisplay) {
                timeDisplay.textContent = `${displayHours}:${minutes} ${ampm}`;
            }
        };
        
        updateTime();
        setInterval(updateTime, 1000);
    }
    
    updateStats() {
        const stats = this.graph.getStats();
        
        document.getElementById('nodeCount').textContent = stats.visibleNodes;
        document.getElementById('edgeCount').textContent = this.data.edges.length;
        document.getElementById('articleCount').textContent = stats.articles;
        
        // Format last updated with date and time
        const lastUpdated = new Date(this.data.metadata.lastUpdated);
        const dateOptions = { month: 'short', day: 'numeric' };
        const timeOptions = { hour: 'numeric', minute: '2-digit', hour12: true };
        const dateStr = lastUpdated.toLocaleDateString('en-US', dateOptions);
        const timeStr = lastUpdated.toLocaleTimeString('en-US', timeOptions);
        document.getElementById('lastUpdated').textContent = `${dateStr}, ${timeStr}`;
    }
    
    populateTrending() {
        const container = document.getElementById('trendingTopics');
        if (!container) return;
        
        // Get topics sorted by connection count
        const topics = this.data.nodes
            .filter(n => n.type === 'topic')
            .sort((a, b) => (b.connectionCount || 0) - (a.connectionCount || 0))
            .slice(0, 5);
        
        const html = topics.map((topic, index) => `
            <div class="trending-item" data-node-id="${topic.id}">
                <span class="trending-rank">#${index + 1}</span>
                <span class="trending-name">${topic.title}</span>
                <span class="trending-count">${topic.connectionCount || 0}</span>
            </div>
        `).join('');
        
        container.innerHTML = `<div class="trending-list">${html}</div>`;
        
        // Add click handlers
        container.querySelectorAll('.trending-item').forEach(item => {
            item.addEventListener('click', () => {
                const nodeId = item.dataset.nodeId;
                const node = this.graph.nodes.find(n => n.id === nodeId);
                if (node) {
                    this.graph.selectNode(node);
                }
            });
        });
    }
    
    populateRecentArticles() {
        const container = document.getElementById('recentArticles');
        if (!container) return;
        
        // Get recent articles
        const articles = this.data.nodes
            .filter(n => n.type === 'article')
            .sort((a, b) => new Date(b.date) - new Date(a.date))
            .slice(0, 5);
        
        const html = articles.map(article => {
            const date = new Date(article.date);
            const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            
            return `
                <div class="article-item" data-node-id="${article.id}">
                    <h4 class="article-item-title">${this.truncate(article.title, 50)}</h4>
                    <span class="article-item-meta">${article.source} • ${formattedDate}</span>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `<div class="article-list">${html}</div>`;
        
        // Add click handlers
        container.querySelectorAll('.article-item').forEach(item => {
            item.addEventListener('click', () => {
                const nodeId = item.dataset.nodeId;
                const node = this.graph.nodes.find(n => n.id === nodeId);
                if (node) {
                    this.graph.selectNode(node);
                }
            });
        });
    }
    
    setupControls() {
        // Reset view button
        const resetBtn = document.getElementById('resetView');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.graph.resetView();
            });
        }
        
        // Toggle labels button
        const labelsBtn = document.getElementById('toggleLabels');
        if (labelsBtn) {
            labelsBtn.addEventListener('click', () => {
                const isOn = this.graph.toggleLabels();
                labelsBtn.classList.toggle('active', isOn);
            });
        }
        
        // Filter trending button
        const trendingBtn = document.getElementById('filterTrending');
        if (trendingBtn) {
            trendingBtn.addEventListener('click', () => {
                const isActive = trendingBtn.classList.toggle('active');
                this.graph.filterTrending = isActive;
                this.graph.updateVisibleNodes();
            });
        }
    }
    
    setupFilters() {
        // Time filter
        const timeFilter = document.getElementById('timeFilter');
        if (timeFilter) {
            timeFilter.addEventListener('change', () => {
                this.applyFilters();
            });
        }
        
        // Source filter
        const sourceFilter = document.getElementById('sourceFilter');
        if (sourceFilter) {
            sourceFilter.addEventListener('change', () => {
                this.applyFilters();
            });
        }
        
        // Node type checkboxes
        ['showArticles', 'showTopics', 'showOrgs', 'showModels'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    this.applyFilters();
                });
            }
        });
    }
    
    applyFilters() {
        const filters = {
            timeRange: document.getElementById('timeFilter')?.value || 'all',
            source: document.getElementById('sourceFilter')?.value || 'all',
            showArticles: document.getElementById('showArticles')?.checked ?? true,
            showTopics: document.getElementById('showTopics')?.checked ?? true,
            showOrgs: document.getElementById('showOrgs')?.checked ?? true,
            showModels: document.getElementById('showModels')?.checked ?? true
        };
        
        this.graph.setFilter(filters);
        this.updateStats();
    }
    
    setupNodeSelection() {
        document.addEventListener('nodeSelected', (e) => {
            const node = e.detail;
            this.displayNodeInfo(node);
        });
    }
    
    displayNodeInfo(node) {
        const container = document.getElementById('nodeInfoContent');
        if (!container) return;
        
        if (!node) {
            container.innerHTML = '<p class="placeholder-text">👆 Click any node to explore its connections and details. Hover to preview, drag to rearrange.</p>';
            return;
        }
        
        // Get connected nodes
        const connections = this.graph.getConnectedNodes(node);
        
        // Build HTML based on node type
        let html = `
            <div class="node-info fade-in">
                <span class="node-info-type ${node.type}">${node.type.toUpperCase()}</span>
                <h4 class="node-info-title">${node.title}</h4>
        `;
        
        if (node.summary) {
            html += `<p class="node-info-summary">${node.summary}</p>`;
        }
        
        if (node.type === 'article') {
            const date = new Date(node.date);
            const formattedDate = date.toLocaleDateString('en-US', { 
                weekday: 'short', 
                month: 'short', 
                day: 'numeric' 
            });
            
            html += `
                <p class="node-info-meta">
                    ${formattedDate} • ${node.source}
                    ${node.trendingScore ? ` • Trending: ${node.trendingScore}` : ''}
                </p>
            `;
            
            if (node.url) {
                html += `<a href="${node.url}" target="_blank" class="node-info-link">Read Article →</a>`;
            }
        }
        
        // Connected nodes
        if (connections.length > 0) {
            html += `
                <div class="node-connections">
                    <p class="node-connections-title">Connected to:</p>
                    <div class="connection-tags">
                        ${connections.map(c => `
                            <span class="connection-tag" data-node-id="${c.node.id}">
                                ${this.getNodeIcon(c.node.type)} ${this.truncate(c.node.title, 20)}
                            </span>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;

        // Scroll to details panel only for articles (calmer scroll)
        if (node.type === 'article') {
            const panel = document.getElementById('nodeInfoCard');
            if (panel) {
                setTimeout(() => {
                    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 150);
            }
        }

        // Add click handlers for connection tags
        container.querySelectorAll('.connection-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                const nodeId = tag.dataset.nodeId;
                const targetNode = this.graph.nodes.find(n => n.id === nodeId);
                if (targetNode) {
                    this.graph.selectNode(targetNode);
                }
            });
        });
    }
    
    getNodeIcon(type) {
        const colors = {
            article: '#e07b53',
            topic: '#5b8a72',
            organization: '#6b8cae',
            model: '#9b7bb8'
        };
        const color = colors[type] || '#888';
        return `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:4px;vertical-align:middle;"></span>`;
    }
    
    truncate(text, maxLen) {
        if (!text) return '';
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen - 3) + '...';
    }
}

// Initialize app
const app = new AIChronicleApp();
