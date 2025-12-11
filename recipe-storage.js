// Recipe Storage Manager - Handle saving/loading recipes from localStorage
// Provides persistent storage for user's soap recipes

class RecipeStorage {
    constructor() {
        this.storageKey = 'saponifyai_saved_recipes';
        this.maxRecipes = 50; // Limit to prevent localStorage overflow
    }

    /**
     * Save a recipe to localStorage
     * @param {string} name - Recipe name
     * @param {Object} recipe - Recipe object from SoapCalculator
     * @param {string} notes - Optional user notes
     * @returns {Object} Save result
     */
    saveRecipe(name, recipe, notes = '') {
        try {
            const recipes = this.getAllRecipes();

            // Generate unique ID
            const id = this.generateId();

            // Create recipe record
            const recipeRecord = {
                id: id,
                name: name.trim(),
                recipe: recipe,
                notes: notes.trim(),
                createdAt: Date.now(),
                updatedAt: Date.now()
            };

            // Check if we're at max capacity
            if (recipes.length >= this.maxRecipes) {
                return {
                    success: false,
                    error: `Maximum ${this.maxRecipes} recipes reached. Please delete some recipes first.`
                };
            }

            // Add to recipes array
            recipes.push(recipeRecord);

            // Save to localStorage
            localStorage.setItem(this.storageKey, JSON.stringify(recipes));

            return {
                success: true,
                id: id,
                message: `Recipe "${name}" saved successfully!`
            };
        } catch (error) {
            console.error('Error saving recipe:', error);
            return {
                success: false,
                error: 'Failed to save recipe. Storage may be full.'
            };
        }
    }

    /**
     * Update an existing recipe
     * @param {string} id - Recipe ID
     * @param {string} name - Recipe name
     * @param {Object} recipe - Recipe object
     * @param {string} notes - User notes
     * @returns {Object} Update result
     */
    updateRecipe(id, name, recipe, notes = '') {
        try {
            const recipes = this.getAllRecipes();
            const index = recipes.findIndex(r => r.id === id);

            if (index === -1) {
                return {
                    success: false,
                    error: 'Recipe not found'
                };
            }

            // Update recipe
            recipes[index] = {
                ...recipes[index],
                name: name.trim(),
                recipe: recipe,
                notes: notes.trim(),
                updatedAt: Date.now()
            };

            // Save to localStorage
            localStorage.setItem(this.storageKey, JSON.stringify(recipes));

            return {
                success: true,
                message: `Recipe "${name}" updated successfully!`
            };
        } catch (error) {
            console.error('Error updating recipe:', error);
            return {
                success: false,
                error: 'Failed to update recipe'
            };
        }
    }

    /**
     * Delete a recipe
     * @param {string} id - Recipe ID
     * @returns {Object} Delete result
     */
    deleteRecipe(id) {
        try {
            const recipes = this.getAllRecipes();
            const filtered = recipes.filter(r => r.id !== id);

            if (filtered.length === recipes.length) {
                return {
                    success: false,
                    error: 'Recipe not found'
                };
            }

            localStorage.setItem(this.storageKey, JSON.stringify(filtered));

            return {
                success: true,
                message: 'Recipe deleted successfully'
            };
        } catch (error) {
            console.error('Error deleting recipe:', error);
            return {
                success: false,
                error: 'Failed to delete recipe'
            };
        }
    }

    /**
     * Get a specific recipe by ID
     * @param {string} id - Recipe ID
     * @returns {Object|null} Recipe record or null
     */
    getRecipe(id) {
        const recipes = this.getAllRecipes();
        return recipes.find(r => r.id === id) || null;
    }

    /**
     * Get all saved recipes
     * @returns {Array} Array of recipe records
     */
    getAllRecipes() {
        try {
            const data = localStorage.getItem(this.storageKey);
            if (!data) return [];
            return JSON.parse(data);
        } catch (error) {
            console.error('Error reading recipes:', error);
            return [];
        }
    }

    /**
     * Search recipes by name
     * @param {string} query - Search query
     * @returns {Array} Matching recipes
     */
    searchRecipes(query) {
        const recipes = this.getAllRecipes();
        const lowerQuery = query.toLowerCase().trim();

        if (!lowerQuery) return recipes;

        return recipes.filter(r =>
            r.name.toLowerCase().includes(lowerQuery) ||
            r.notes.toLowerCase().includes(lowerQuery)
        );
    }

    /**
     * Get recipes sorted by date
     * @param {boolean} ascending - Sort order
     * @returns {Array} Sorted recipes
     */
    getRecipesSortedByDate(ascending = false) {
        const recipes = this.getAllRecipes();
        return recipes.sort((a, b) => {
            return ascending
                ? a.createdAt - b.createdAt
                : b.createdAt - a.createdAt;
        });
    }

    /**
     * Get storage statistics
     * @returns {Object} Storage stats
     */
    getStats() {
        const recipes = this.getAllRecipes();
        return {
            total: recipes.length,
            maxCapacity: this.maxRecipes,
            percentFull: Math.round((recipes.length / this.maxRecipes) * 100),
            oldestRecipe: recipes.length > 0
                ? Math.min(...recipes.map(r => r.createdAt))
                : null,
            newestRecipe: recipes.length > 0
                ? Math.max(...recipes.map(r => r.createdAt))
                : null
        };
    }

    /**
     * Export all recipes as JSON
     * @returns {string} JSON string of all recipes
     */
    exportRecipes() {
        const recipes = this.getAllRecipes();
        return JSON.stringify(recipes, null, 2);
    }

    /**
     * Import recipes from JSON
     * @param {string} jsonString - JSON string of recipes
     * @returns {Object} Import result
     */
    importRecipes(jsonString) {
        try {
            const importedRecipes = JSON.parse(jsonString);

            if (!Array.isArray(importedRecipes)) {
                return {
                    success: false,
                    error: 'Invalid format: expected array of recipes'
                };
            }

            const currentRecipes = this.getAllRecipes();
            const combined = [...currentRecipes];

            let imported = 0;
            let skipped = 0;

            for (const recipe of importedRecipes) {
                // Check if recipe already exists (by name and creation date)
                const exists = combined.some(r =>
                    r.name === recipe.name && r.createdAt === recipe.createdAt
                );

                if (!exists && combined.length < this.maxRecipes) {
                    combined.push(recipe);
                    imported++;
                } else {
                    skipped++;
                }
            }

            localStorage.setItem(this.storageKey, JSON.stringify(combined));

            return {
                success: true,
                imported: imported,
                skipped: skipped,
                message: `Imported ${imported} recipe(s). ${skipped} skipped (duplicates or capacity limit).`
            };
        } catch (error) {
            console.error('Error importing recipes:', error);
            return {
                success: false,
                error: 'Failed to import recipes. Invalid JSON format.'
            };
        }
    }

    /**
     * Clear all saved recipes
     * @returns {Object} Clear result
     */
    clearAll() {
        try {
            localStorage.removeItem(this.storageKey);
            return {
                success: true,
                message: 'All recipes deleted successfully'
            };
        } catch (error) {
            console.error('Error clearing recipes:', error);
            return {
                success: false,
                error: 'Failed to clear recipes'
            };
        }
    }

    /**
     * Generate unique ID for recipe
     * @returns {string} Unique ID
     */
    generateId() {
        return 'recipe_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Format timestamp for display
     * @param {number} timestamp - Timestamp in milliseconds
     * @returns {string} Formatted date
     */
    formatDate(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RecipeStorage;
} else {
    window.RecipeStorage = RecipeStorage;
}
