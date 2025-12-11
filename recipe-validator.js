// Recipe Validator - Safety checks for soap recipes
// Prevents dangerous or problematic soap formulations

class RecipeValidator {
    constructor() {
        // Define safe ranges for soap properties
        this.safeRanges = {
            batchSize: { min: 100, max: 5000 }, // grams
            superfat: { min: 0, max: 20 }, // percent
            coconutOilMax: 50, // percent - can be drying above this
            hardOilsMin: 25, // percent - too soft otherwise
            iodineSafeMax: 70, // shelf life concern above this
            cleansingMax: 30 // too harsh above this
        };

        // Common problematic combinations
        this.warnings = {
            highCoconut: {
                threshold: 35,
                message: '⚠️ High coconut oil (>35%) can be drying to skin. Consider reducing to 25-30%.',
                severity: 'warning'
            },
            lowHardness: {
                threshold: 25,
                message: '⚠️ Low hardness - soap may be too soft. Add palm oil, cocoa butter, or shea butter.',
                severity: 'warning'
            },
            highIodine: {
                threshold: 70,
                message: '⚠️ High iodine value (>70) - short shelf life. Reduce soft oils or add vitamin E.',
                severity: 'warning'
            },
            highCleansing: {
                threshold: 25,
                message: '⚠️ Very cleansing - may strip skin oils. Reduce coconut oil for gentler soap.',
                severity: 'warning'
            },
            extremeSuperfat: {
                low: 2,
                high: 15,
                message: '⚠️ Unusual superfat percentage. Standard range is 5-8%. Are you sure?',
                severity: 'warning'
            },
            noCastor: {
                message: '💡 Tip: Adding 5-10% castor oil boosts lather and bubbles!',
                severity: 'tip'
            },
            allSoftOils: {
                message: '⚠️ No hard oils detected - soap will be very soft. Add coconut, palm, or butters.',
                severity: 'error'
            }
        };
    }

    /**
     * Validate batch size
     * @param {number} grams - Batch size in grams
     * @returns {Object} Validation result
     */
    validateBatchSize(grams) {
        const errors = [];
        const warnings = [];

        if (grams < this.safeRanges.batchSize.min) {
            errors.push(`Batch size too small (${grams}g). Minimum is ${this.safeRanges.batchSize.min}g for accurate measurements.`);
        }

        if (grams > this.safeRanges.batchSize.max) {
            errors.push(`Batch size too large (${grams}g). Maximum is ${this.safeRanges.batchSize.max}g for safety. Make multiple batches instead.`);
        }

        return { valid: errors.length === 0, errors, warnings };
    }

    /**
     * Validate superfat percentage
     * @param {number} percent - Superfat percentage
     * @returns {Object} Validation result
     */
    validateSuperfat(percent) {
        const errors = [];
        const warnings = [];

        if (percent < this.safeRanges.superfat.min) {
            errors.push(`Superfat cannot be negative or zero. Use 5-8% for skin-safe soap.`);
        }

        if (percent > this.safeRanges.superfat.max) {
            errors.push(`Superfat too high (${percent}%). Maximum is ${this.safeRanges.superfat.max}% to prevent rancidity.`);
        }

        if (percent < this.warnings.extremeSuperfat.low || percent > this.warnings.extremeSuperfat.high) {
            warnings.push(this.warnings.extremeSuperfat.message);
        }

        return { valid: errors.length === 0, errors, warnings };
    }

    /**
     * Validate oil percentages
     * @param {Array} oils - Array of oil objects with name and percent
     * @returns {Object} Validation result
     */
    validateOilPercentages(oils) {
        const errors = [];
        const warnings = [];

        // Calculate total percentage
        const totalPercent = oils.reduce((sum, oil) => sum + oil.percent, 0);

        if (Math.abs(totalPercent - 100) > 0.5) {
            errors.push(`Oil percentages must total 100%. Currently: ${totalPercent.toFixed(1)}%`);
        }

        // Check for coconut oil percentage
        const coconutOils = oils.filter(oil =>
            oil.name.toLowerCase().includes('coconut') ||
            oil.name.toLowerCase().includes('babassu')
        );

        const totalCoconut = coconutOils.reduce((sum, oil) => sum + oil.percent, 0);

        if (totalCoconut > this.safeRanges.coconutOilMax) {
            errors.push(`Coconut/Babassu oil too high (${totalCoconut.toFixed(1)}%). Maximum recommended is ${this.safeRanges.coconutOilMax}% to avoid skin dryness.`);
        } else if (totalCoconut > this.warnings.highCoconut.threshold) {
            warnings.push(this.warnings.highCoconut.message);
        }

        // Check for castor oil (helpful but not required)
        const hasCastor = oils.some(oil => oil.name.toLowerCase().includes('castor'));
        if (!hasCastor) {
            warnings.push(this.warnings.noCastor.message);
        }

        // Check for hard oils
        const hardOilNames = ['coconut', 'palm', 'babassu', 'cocoa butter', 'shea butter',
                             'mango butter', 'tallow', 'lard'];
        const hardOils = oils.filter(oil =>
            hardOilNames.some(name => oil.name.toLowerCase().includes(name))
        );
        const totalHardOils = hardOils.reduce((sum, oil) => sum + oil.percent, 0);

        if (totalHardOils < this.safeRanges.hardOilsMin) {
            if (totalHardOils < 10) {
                warnings.push(this.warnings.allSoftOils.message);
            } else {
                warnings.push(this.warnings.lowHardness.message);
            }
        }

        return { valid: errors.length === 0, errors, warnings };
    }

    /**
     * Validate soap properties (from calculator results)
     * @param {Object} properties - Soap properties object
     * @returns {Object} Validation result
     */
    validateSoapProperties(properties) {
        const errors = [];
        const warnings = [];

        // Check hardness
        if (properties.hardness.value < 25) {
            warnings.push(this.warnings.lowHardness.message);
        }

        // Check cleansing
        if (properties.cleansing.value > this.warnings.highCleansing.threshold) {
            warnings.push(this.warnings.highCleansing.message);
        }

        // Check iodine value (shelf life)
        if (properties.iodine.value > this.warnings.highIodine.threshold) {
            warnings.push(this.warnings.highIodine.message);
        }

        // Check INS value (overall quality indicator)
        if (properties.ins.value < 120) {
            warnings.push('⚠️ Low INS value - soap may be soft and have short shelf life.');
        } else if (properties.ins.value > 180) {
            warnings.push('⚠️ High INS value - soap may be very hard and less conditioning.');
        }

        return { valid: errors.length === 0, errors, warnings };
    }

    /**
     * Validate complete recipe
     * @param {Object} recipe - Complete recipe from SoapCalculator
     * @returns {Object} Complete validation results
     */
    validateRecipe(recipe) {
        const results = {
            valid: true,
            errors: [],
            warnings: [],
            tips: []
        };

        // Validate oils
        const oilValidation = this.validateOilPercentages(recipe.oils);
        results.errors.push(...oilValidation.errors);
        results.warnings.push(...oilValidation.warnings);

        // Validate superfat
        const superfatValidation = this.validateSuperfat(recipe.superfat);
        results.errors.push(...superfatValidation.errors);
        results.warnings.push(...superfatValidation.warnings);

        // Validate soap properties
        const propertyValidation = this.validateSoapProperties(recipe.properties);
        results.errors.push(...propertyValidation.errors);
        results.warnings.push(...propertyValidation.warnings);

        // Overall validity
        results.valid = results.errors.length === 0;

        return results;
    }

    /**
     * Format validation results for display
     * @param {Object} validation - Validation results
     * @returns {string} Formatted message
     */
    formatValidationMessage(validation) {
        let message = '';

        if (validation.errors.length > 0) {
            message += '## ❌ Recipe Errors\n\n';
            validation.errors.forEach(error => {
                message += `- ${error}\n`;
            });
            message += '\n**This recipe is not safe to make. Please adjust the formulation.**\n\n';
        }

        if (validation.warnings.length > 0) {
            message += '## ⚠️ Recipe Warnings\n\n';
            validation.warnings.forEach(warning => {
                message += `- ${warning}\n`;
            });
            message += '\n';
        }

        if (validation.tips.length > 0) {
            message += '## 💡 Tips\n\n';
            validation.tips.forEach(tip => {
                message += `- ${tip}\n`;
            });
            message += '\n';
        }

        if (validation.valid && validation.warnings.length === 0) {
            message += '✅ **Recipe looks great! Safe to make.**\n\n';
        }

        return message;
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RecipeValidator;
} else {
    window.RecipeValidator = RecipeValidator;
}
