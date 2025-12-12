// Recipe UI Manager - Handle recipe saving/loading user interface
// Integrates RecipeStorage with the chat interface

let recipeStorage = null;
let currentFilteredRecipes = [];

// Initialize recipe storage on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeRecipeStorage();
});

// Ensure recipe storage is initialized (can be called multiple times safely)
function initializeRecipeStorage() {
    if (!recipeStorage && typeof RecipeStorage !== 'undefined') {
        recipeStorage = new RecipeStorage();
        console.log('✅ RecipeStorage initialized');
        return true;
    } else if (!recipeStorage) {
        console.warn('⚠️ RecipeStorage not loaded');
        return false;
    }
    return true;
}

// ============================================
// SAVE RECIPE MODAL FUNCTIONS
// ============================================

/**
 * Open save recipe modal
 */
function openSaveRecipeModal() {
    // Check if there's a recipe to save
    if (!lastCalculatedRecipe) {
        showCopyFeedback('⚠️ No recipe to save. Calculate a recipe first!');
        return;
    }

    // Open modal
    const modal = document.getElementById('saveRecipeModal');
    modal.classList.add('show');

    // Clear form
    document.getElementById('recipeName').value = '';
    document.getElementById('recipeNotes').value = '';

    // Focus on name input
    setTimeout(() => {
        document.getElementById('recipeName').focus();
    }, 100);
}

/**
 * Close save recipe modal
 */
function closeSaveRecipeModal() {
    const modal = document.getElementById('saveRecipeModal');
    modal.classList.remove('show');
}

/**
 * Handle save recipe form submission
 */
function handleSaveRecipe(event) {
    event.preventDefault();

    if (!recipeStorage || !lastCalculatedRecipe) {
        showCopyFeedback('❌ Cannot save recipe');
        return;
    }

    const name = document.getElementById('recipeName').value.trim();
    const notes = document.getElementById('recipeNotes').value.trim();

    // Save the recipe
    const result = recipeStorage.saveRecipe(name, lastCalculatedRecipe.recipe, notes);

    if (result.success) {
        showCopyFeedback(`✅ ${result.message}`);
        closeSaveRecipeModal();
    } else {
        showCopyFeedback(`❌ ${result.error}`);
    }
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    const saveModal = document.getElementById('saveRecipeModal');
    const libraryModal = document.getElementById('recipeLibraryModal');

    if (e.target === saveModal) {
        closeSaveRecipeModal();
    }
    if (e.target === libraryModal) {
        closeRecipeLibrary();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSaveRecipeModal();
        closeRecipeLibrary();
    }
});

// ============================================
// RECIPE LIBRARY MODAL FUNCTIONS
// ============================================

/**
 * Open recipe library modal
 */
function openRecipeLibrary() {
    // Try to initialize if not already done
    if (!recipeStorage) {
        console.log('⚠️ Recipe storage not initialized, attempting to initialize...');
        if (!initializeRecipeStorage()) {
            showCopyFeedback('❌ Recipe storage not available');
            alert('Recipe storage could not be initialized. Please refresh the page.');
            return;
        }
    }

    const modal = document.getElementById('recipeLibraryModal');
    if (!modal) {
        console.error('❌ Recipe library modal not found in DOM');
        alert('Recipe library modal not found. Please refresh the page.');
        return;
    }

    modal.classList.add('show');

    // Load and display recipes
    refreshRecipeLibrary();
}

/**
 * Close recipe library modal
 */
function closeRecipeLibrary() {
    const modal = document.getElementById('recipeLibraryModal');
    modal.classList.remove('show');

    // Clear search
    document.getElementById('librarySearchInput').value = '';
}

/**
 * Refresh recipe library display
 */
function refreshRecipeLibrary() {
    if (!recipeStorage) return;

    const recipes = recipeStorage.getRecipesSortedByDate(false); // newest first
    currentFilteredRecipes = recipes;

    // Update stats
    const stats = recipeStorage.getStats();
    document.getElementById('recipeCount').textContent = `${stats.total} recipe${stats.total === 1 ? '' : 's'}`;

    const capacityBadge = document.getElementById('storageCapacity');
    capacityBadge.textContent = `${stats.total}/${stats.maxCapacity}`;
    capacityBadge.classList.toggle('warning', stats.percentFull >= 80);

    // Render recipes
    renderRecipes(recipes);
}

/**
 * Render recipes in library
 */
function renderRecipes(recipes) {
    const container = document.getElementById('recipesLibrary');

    if (recipes.length === 0) {
        container.innerHTML = `
            <div class="empty-library">
                <div class="empty-library-icon">📭</div>
                <p><strong>No recipes saved yet</strong></p>
                <p>Calculate a recipe and click "💾 Save Recipe" to save it here!</p>
            </div>
        `;
        return;
    }

    container.innerHTML = recipes.map(record => createRecipeCard(record)).join('');
}

/**
 * Create HTML for recipe card
 */
function createRecipeCard(record) {
    const recipe = record.recipe;
    const date = recipeStorage.formatDate(record.createdAt);

    // Get key info
    const totalOils = recipe.oils.length;
    const batchSize = recipe.totalBatchSize.grams;
    const superfat = recipe.superfat;

    // Get primary oils (top 3 by percentage)
    const topOils = [...recipe.oils]
        .sort((a, b) => b.percent - a.percent)
        .slice(0, 3)
        .map(oil => oil.name.split(' ')[0]) // Just first word
        .join(', ');

    return `
        <div class="recipe-card" data-recipe-id="${record.id}">
            <div class="recipe-card-header">
                <h3 class="recipe-card-title">${escapeHtml(record.name)}</h3>
                <span class="recipe-card-date">${date}</span>
            </div>
            <div class="recipe-card-info">
                <span class="recipe-card-detail"><strong>${batchSize}g</strong> batch</span>
                <span class="recipe-card-detail"><strong>${totalOils}</strong> oils</span>
                <span class="recipe-card-detail"><strong>${superfat}%</strong> superfat</span>
            </div>
            <div class="recipe-card-detail">
                <strong>Oils:</strong> ${topOils}${totalOils > 3 ? '...' : ''}
            </div>
            ${record.notes ? `<p class="recipe-card-notes">"${escapeHtml(record.notes)}"</p>` : ''}
            <div class="recipe-card-actions" onclick="event.stopPropagation()">
                <button class="btn-small btn-load" onclick="loadRecipe('${record.id}')">
                    📂 Load Recipe
                </button>
                <button class="btn-small btn-delete" onclick="confirmDeleteRecipe('${record.id}', '${escapeHtml(record.name)}')">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Filter recipes by search query
 */
function filterRecipes(query) {
    if (!recipeStorage) return;

    const filtered = recipeStorage.searchRecipes(query);
    currentFilteredRecipes = filtered;
    renderRecipes(filtered);

    // Update count
    const stats = recipeStorage.getStats();
    if (query.trim()) {
        document.getElementById('recipeCount').textContent =
            `${filtered.length} of ${stats.total} recipe${stats.total === 1 ? '' : 's'}`;
    } else {
        document.getElementById('recipeCount').textContent =
            `${stats.total} recipe${stats.total === 1 ? '' : 's'}`;
    }
}

// ============================================
// LOAD RECIPE FUNCTIONALITY
// ============================================

/**
 * Load a saved recipe and display it in the chat
 */
function loadRecipe(recipeId) {
    if (!recipeStorage) return;

    const record = recipeStorage.getRecipe(recipeId);
    if (!record) {
        showCopyFeedback('❌ Recipe not found');
        return;
    }

    // Format and display the recipe
    const recipe = record.recipe;
    const loadedMessage = `## 📂 Loaded Recipe: "${record.name}"\n\n` +
                         `*Saved on ${recipeStorage.formatDate(record.createdAt)}*\n\n` +
                         (record.notes ? `**Notes:** ${record.notes}\n\n---\n\n` : '---\n\n');

    const recipeOutput = formatCalculatedRecipe(recipe);
    const fullOutput = loadedMessage + recipeOutput;

    // Add to chat
    addMessage(fullOutput, true);

    // Update lastCalculatedRecipe so user can copy/scale
    lastCalculatedRecipe = {
        recipe: recipe,
        id: generateRecipeId(),
        timestamp: Date.now()
    };
    lastCalculatedRecipeTime = Date.now();

    // Close library modal
    closeRecipeLibrary();

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    showCopyFeedback(`✅ Loaded "${record.name}"`);
}

// ============================================
// DELETE RECIPE FUNCTIONALITY
// ============================================

/**
 * Confirm and delete recipe
 */
function confirmDeleteRecipe(recipeId, recipeName) {
    if (!recipeStorage) return;

    const confirmMessage = `Are you sure you want to delete "${recipeName}"? This cannot be undone.`;

    if (confirm(confirmMessage)) {
        const result = recipeStorage.deleteRecipe(recipeId);

        if (result.success) {
            showCopyFeedback(`✅ Recipe deleted`);
            refreshRecipeLibrary();
        } else {
            showCopyFeedback(`❌ ${result.error}`);
        }
    }
}

// ============================================
// UPDATE RECIPE DISPLAY WITH SAVE BUTTON
// ============================================

// Override formatCalculatedRecipe to add save button
// Store original function
const originalFormatCalculatedRecipe = formatCalculatedRecipe;

// Replace with enhanced version
formatCalculatedRecipe = function(result) {
    const output = originalFormatCalculatedRecipe(result);

    // Add save button to recipe actions
    const saveButton = `<button onclick="openSaveRecipeModal()" class="recipe-action-btn">💾 Save Recipe</button>\n`;

    // Insert save button after "Copy Recipe" button
    return output.replace(
        '<button onclick="copyRecipeToClipboard()" class="recipe-action-btn">📋 Copy Recipe</button>',
        '<button onclick="copyRecipeToClipboard()" class="recipe-action-btn">📋 Copy Recipe</button>\n' + saveButton
    );
};
