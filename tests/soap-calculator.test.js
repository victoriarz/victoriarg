// Unit Tests for SoapCalculator
// Critical safety tests for lye calculations

// Simple test framework (can be replaced with Jest/Mocha later)
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    assertEquals(actual, expected, tolerance = 0.1) {
        if (Math.abs(actual - expected) > tolerance) {
            throw new Error(`Expected ${expected}, got ${actual} (tolerance: ${tolerance})`);
        }
    }

    async run() {
        console.log('🧪 Running SoapCalculator Tests...\n');

        for (const test of this.tests) {
            try {
                await test.fn();
                console.log(`✅ ${test.name}`);
                this.passed++;
            } catch (error) {
                console.error(`❌ ${test.name}`);
                console.error(`   Error: ${error.message}`);
                this.failed++;
            }
        }

        console.log(`\n📊 Test Results: ${this.passed} passed, ${this.failed} failed`);
        return this.failed === 0;
    }
}

// Initialize test runner
const runner = new TestRunner();

// ============================================
// CRITICAL LYE CALCULATION TESTS
// ============================================

runner.test('Basic olive oil recipe - lye calculation', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');
    calc.setSuperfat(5);

    const result = calc.calculate();

    // Expected lye: 1000g × 0.1353 SAP × 0.95 (5% superfat) = 128.5g
    runner.assertEquals(result.lye.grams, 128.5, 0.5);
});

runner.test('Basic coconut oil recipe - lye calculation', () => {
    const calc = new SoapCalculator();
    calc.addOil('coconut', 1000, 'grams');
    calc.setSuperfat(5);

    const result = calc.calculate();

    // Expected lye: 1000g × 0.1908 SAP × 0.95 = 181.3g
    runner.assertEquals(result.lye.grams, 181.3, 0.5);
});

runner.test('Mixed oils recipe - lye calculation', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 500, 'grams');
    calc.addOil('coconut', 300, 'grams');
    calc.addOil('palm', 200, 'grams');
    calc.setSuperfat(8);

    const result = calc.calculate();

    // Olive: 500 × 0.1353 = 67.65
    // Coconut: 300 × 0.1908 = 57.24
    // Palm: 200 × 0.1410 = 28.2
    // Total: 153.09 × 0.92 (8% superfat) = 140.8g
    runner.assertEquals(result.lye.grams, 140.8, 1.0);
});

runner.test('Water calculation - 2.5:1 ratio', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');
    calc.setSuperfat(5);
    calc.setWaterRatio(2.5);

    const result = calc.calculate();

    // Lye: 128.5g, Water: 128.5 × 2.5 = 321.25g
    runner.assertEquals(result.water.grams, 321.3, 2.0);
});

runner.test('Water calculation - 33% lye concentration', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');
    calc.setSuperfat(5);
    calc.setLyeConcentration(33);

    const result = calc.calculate();

    // Lye: 128.5g at 33% concentration
    // Total solution: 128.5 / 0.33 = 389.4g
    // Water: 389.4 - 128.5 = 260.9g
    runner.assertEquals(result.water.grams, 260.9, 2.0);
});

runner.test('Superfat percentage validation - too low', () => {
    const calc = new SoapCalculator();

    try {
        calc.setSuperfat(-5);
        runner.assert(false, 'Should throw error for negative superfat');
    } catch (error) {
        runner.assert(error.message.includes('Superfat'), 'Error message should mention superfat');
    }
});

runner.test('Superfat percentage validation - too high', () => {
    const calc = new SoapCalculator();

    try {
        calc.setSuperfat(25);
        runner.assert(false, 'Should throw error for superfat > 20%');
    } catch (error) {
        runner.assert(error.message.includes('Superfat'), 'Error message should mention superfat');
    }
});

runner.test('Oil percentages sum to 100%', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 600, 'grams');
    calc.addOil('coconut', 300, 'grams');
    calc.addOil('castor', 100, 'grams');

    const result = calc.calculate();

    const totalPercent = result.oils.reduce((sum, oil) => sum + oil.percent, 0);
    runner.assertEquals(totalPercent, 100, 0.1);
});

// ============================================
// SOAP PROPERTIES TESTS
// ============================================

runner.test('High coconut oil - high cleansing value', () => {
    const calc = new SoapCalculator();
    calc.addOil('coconut', 700, 'grams');
    calc.addOil('olive', 300, 'grams');

    const result = calc.calculate();

    // 70% coconut should give high cleansing (>45)
    runner.assert(result.properties.cleansing.value > 45, 'High coconut should give high cleansing');
});

runner.test('Castile soap - high conditioning', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');

    const result = calc.calculate();

    // 100% olive should give high conditioning (~82)
    runner.assertEquals(result.properties.conditioning.value, 82, 3);
});

runner.test('Recipe with castor - high bubbly lather', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 700, 'grams');
    calc.addOil('coconut', 200, 'grams');
    calc.addOil('castor', 100, 'grams');

    const result = calc.calculate();

    // Castor boosts bubbly lather
    runner.assert(result.properties.bubbly.value > 20, 'Castor should boost bubbly lather');
});

// ============================================
// FATTY ACID PROFILE TESTS
// ============================================

runner.test('Olive oil fatty acid profile', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');

    const result = calc.calculate();

    // Olive oil is high in oleic acid (~71%)
    runner.assertEquals(result.fattyAcids.oleic, 71, 2);
    // Low in lauric (0%)
    runner.assertEquals(result.fattyAcids.lauric, 0, 0.1);
});

runner.test('Coconut oil fatty acid profile', () => {
    const calc = new SoapCalculator();
    calc.addOil('coconut', 1000, 'grams');

    const result = calc.calculate();

    // Coconut oil is high in lauric acid (~48%)
    runner.assertEquals(result.fattyAcids.lauric, 48, 2);
    // High in myristic (~19%)
    runner.assertEquals(result.fattyAcids.myristic, 19, 2);
});

// ============================================
// EDGE CASES & ERROR HANDLING
// ============================================

runner.test('Empty recipe throws error', () => {
    const calc = new SoapCalculator();

    try {
        calc.calculate();
        runner.assert(false, 'Should throw error for empty recipe');
    } catch (error) {
        runner.assert(error.message.includes('No oils'), 'Error should mention no oils');
    }
});

runner.test('Unknown oil throws error', () => {
    const calc = new SoapCalculator();

    try {
        calc.addOil('unknownoil123', 500, 'grams');
        runner.assert(false, 'Should throw error for unknown oil');
    } catch (error) {
        runner.assert(error.message.includes('not found'), 'Error should mention oil not found');
    }
});

runner.test('Case insensitive oil lookup', () => {
    const calc = new SoapCalculator();

    // Should work with different cases
    calc.addOil('OLIVE', 500, 'grams');
    calc.addOil('CoCoNuT', 300, 'grams');
    calc.addOil('palm oil', 200, 'grams');

    const result = calc.calculate();
    runner.assert(result.oils.length === 3, 'Should accept case-insensitive oil names');
});

runner.test('Small batch size - precision test', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 100, 'grams');
    calc.setSuperfat(5);

    const result = calc.calculate();

    // Lye: 100 × 0.1353 × 0.95 = 12.85g
    runner.assertEquals(result.lye.grams, 12.9, 0.3);
    runner.assert(result.lye.grams > 0, 'Lye amount should be positive');
});

runner.test('Large batch size - precision test', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 5000, 'grams');
    calc.setSuperfat(5);

    const result = calc.calculate();

    // Lye: 5000 × 0.1353 × 0.95 = 642.675g
    runner.assertEquals(result.lye.grams, 642.7, 2.0);
});

// ============================================
// SAFETY VALIDATION TESTS
// ============================================

runner.test('Zero superfat warning test', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 1000, 'grams');

    try {
        calc.setSuperfat(0);
        const result = calc.calculate();
        // Should calculate but validator should flag it
        runner.assert(result.lye.grams > 0, 'Should calculate with 0% superfat');
    } catch (error) {
        // Acceptable if calculator prevents 0% superfat
        runner.assert(true);
    }
});

runner.test('INS value calculation', () => {
    const calc = new SoapCalculator();
    calc.addOil('olive', 500, 'grams');
    calc.addOil('coconut', 300, 'grams');
    calc.addOil('palm', 200, 'grams');

    const result = calc.calculate();

    // INS should be in reasonable range (120-180)
    runner.assert(result.properties.ins.value >= 100, 'INS should be >= 100');
    runner.assert(result.properties.ins.value <= 200, 'INS should be <= 200');
});

// ============================================
// RUN ALL TESTS
// ============================================

// Export test runner for browser or Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runner, TestRunner };
}

// Auto-run in browser console
if (typeof window !== 'undefined' && typeof SoapCalculator !== 'undefined') {
    console.log('SoapCalculator tests loaded. Run tests with: runner.run()');
}
