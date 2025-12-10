// Comprehensive Soap Calculator Engine
// Based on SoapCalc.net functionality with accurate SAP values and fatty acid profiles

// ============================================
// 1. OIL DATABASE WITH FATTY ACID PROFILES
// ============================================

const oilsDatabase = {
    'olive': {
        name: 'Olive Oil',
        sapNaOH: 0.1353,  // SAP value for NaOH (per gram)
        sapKOH: 0.1899,   // SAP value for KOH (per gram)
        // Fatty acid composition (%)
        lauric: 0,
        myristic: 0,
        palmitic: 11,
        stearic: 4,
        ricinoleic: 0,
        oleic: 71,
        linoleic: 10,
        linolenic: 1,
        properties: {
            hardness: 15,
            cleansing: 0,
            conditioning: 82,
            bubbly: 0,
            creamy: 15
        },
        description: 'Gentle, conditioning oil that creates mild soap. Makes creamy lather but can be slimy if used alone.'
    },
    'coconut': {
        name: 'Coconut Oil',
        sapNaOH: 0.1908,
        sapKOH: 0.2677,
        lauric: 48,
        myristic: 19,
        palmitic: 9,
        stearic: 3,
        ricinoleic: 0,
        oleic: 8,
        linoleic: 2,
        linolenic: 0,
        properties: {
            hardness: 79,
            cleansing: 67,
            conditioning: 10,
            bubbly: 67,
            creamy: 12
        },
        description: 'Creates hard bars with fluffy lather. Can be drying if over 30% of recipe.'
    },
    'palm': {
        name: 'Palm Oil',
        sapNaOH: 0.1410,
        sapKOH: 0.1978,
        lauric: 0,
        myristic: 1,
        palmitic: 44,
        stearic: 5,
        ricinoleic: 0,
        oleic: 39,
        linoleic: 10,
        linolenic: 0,
        properties: {
            hardness: 50,
            cleansing: 1,
            conditioning: 49,
            bubbly: 1,
            creamy: 49
        },
        description: 'Hard bar with creamy lather. Often used as sustainable alternative to palm.'
    },
    'castor': {
        name: 'Castor Oil',
        sapNaOH: 0.1286,
        sapKOH: 0.1803,
        lauric: 0,
        myristic: 0,
        palmitic: 2,
        stearic: 1,
        ricinoleic: 90,
        oleic: 4,
        linoleic: 4,
        linolenic: 0,
        properties: {
            hardness: 3,
            cleansing: 0,
            conditioning: 98,
            bubbly: 90,
            creamy: 93
        },
        description: 'Thick oil that boosts lather and creates creamy, stable bubbles. Use 5-10% in recipes.'
    },
    'sweet almond': {
        name: 'Sweet Almond Oil',
        sapNaOH: 0.1387,
        sapKOH: 0.1946,
        lauric: 0,
        myristic: 0,
        palmitic: 7,
        stearic: 2,
        ricinoleic: 0,
        oleic: 69,
        linoleic: 17,
        linolenic: 0,
        properties: {
            hardness: 9,
            cleansing: 0,
            conditioning: 86,
            bubbly: 0,
            creamy: 9
        },
        description: 'Luxurious conditioning oil that makes mild, moisturizing soap.'
    },
    'avocado': {
        name: 'Avocado Oil',
        sapNaOH: 0.1339,
        sapKOH: 0.1879,
        lauric: 0,
        myristic: 0,
        palmitic: 20,
        stearic: 2,
        ricinoleic: 0,
        oleic: 58,
        linoleic: 12,
        linolenic: 1,
        properties: {
            hardness: 22,
            cleansing: 0,
            conditioning: 71,
            bubbly: 0,
            creamy: 22
        },
        description: 'Rich, moisturizing oil great for dry or mature skin.'
    },
    'shea butter': {
        name: 'Shea Butter',
        sapNaOH: 0.1283,
        sapKOH: 0.1800,
        lauric: 0,
        myristic: 0,
        palmitic: 5,
        stearic: 41,
        ricinoleic: 0,
        oleic: 46,
        linoleic: 6,
        linolenic: 0,
        properties: {
            hardness: 46,
            cleansing: 0,
            conditioning: 52,
            bubbly: 0,
            creamy: 46
        },
        description: 'Conditioning butter that creates hard, creamy bars. Excellent for dry skin.'
    },
    'cocoa butter': {
        name: 'Cocoa Butter',
        sapNaOH: 0.1376,
        sapKOH: 0.1930,
        lauric: 0,
        myristic: 0,
        palmitic: 28,
        stearic: 34,
        ricinoleic: 0,
        oleic: 34,
        linoleic: 3,
        linolenic: 0,
        properties: {
            hardness: 62,
            cleansing: 0,
            conditioning: 37,
            bubbly: 0,
            creamy: 62
        },
        description: 'Very hard bar with rich, creamy lather. Chocolate scent may come through.'
    },
    'sunflower': {
        name: 'Sunflower Oil',
        sapNaOH: 0.1358,
        sapKOH: 0.1905,
        lauric: 0,
        myristic: 0,
        palmitic: 6,
        stearic: 5,
        ricinoleic: 0,
        oleic: 19,
        linoleic: 68,
        linolenic: 0,
        properties: {
            hardness: 11,
            cleansing: 0,
            conditioning: 87,
            bubbly: 0,
            creamy: 11
        },
        description: 'Light, conditioning oil. High linoleic content means shorter shelf life.'
    },
    'grapeseed': {
        name: 'Grapeseed Oil',
        sapNaOH: 0.1323,
        sapKOH: 0.1856,
        lauric: 0,
        myristic: 0,
        palmitic: 7,
        stearic: 4,
        ricinoleic: 0,
        oleic: 16,
        linoleic: 70,
        linolenic: 0,
        properties: {
            hardness: 11,
            cleansing: 0,
            conditioning: 86,
            bubbly: 0,
            creamy: 11
        },
        description: 'Light oil that absorbs quickly. Short shelf life due to high linoleic acid.'
    },
    'jojoba': {
        name: 'Jojoba Oil',
        sapNaOH: 0.0696,
        sapKOH: 0.0976,
        lauric: 0,
        myristic: 0,
        palmitic: 2,
        stearic: 0,
        ricinoleic: 0,
        oleic: 12,
        linoleic: 0,
        linolenic: 0,
        properties: {
            hardness: 2,
            cleansing: 0,
            conditioning: 12,
            bubbly: 0,
            creamy: 2
        },
        description: 'Actually a liquid wax. Excellent for skin but expensive. Use in small amounts.'
    },
    'hemp': {
        name: 'Hemp Seed Oil',
        sapNaOH: 0.1357,
        sapKOH: 0.1904,
        lauric: 0,
        myristic: 0,
        palmitic: 6,
        stearic: 2,
        ricinoleic: 0,
        oleic: 12,
        linoleic: 57,
        linolenic: 21,
        properties: {
            hardness: 8,
            cleansing: 0,
            conditioning: 90,
            bubbly: 0,
            creamy: 8
        },
        description: 'Conditioning oil rich in essential fatty acids. Short shelf life.'
    },
    'apricot kernel': {
        name: 'Apricot Kernel Oil',
        sapNaOH: 0.1390,
        sapKOH: 0.1950,
        lauric: 0,
        myristic: 0,
        palmitic: 6,
        stearic: 2,
        ricinoleic: 0,
        oleic: 60,
        linoleic: 29,
        linolenic: 0,
        properties: {
            hardness: 8,
            cleansing: 0,
            conditioning: 89,
            bubbly: 0,
            creamy: 8
        },
        description: 'Light, easily absorbed oil similar to sweet almond oil.'
    },
    'canola': {
        name: 'Canola Oil',
        sapNaOH: 0.1329,
        sapKOH: 0.1864,
        lauric: 0,
        myristic: 0,
        palmitic: 4,
        stearic: 2,
        ricinoleic: 0,
        oleic: 61,
        linoleic: 21,
        linolenic: 9,
        properties: {
            hardness: 6,
            cleansing: 0,
            conditioning: 91,
            bubbly: 0,
            creamy: 6
        },
        description: 'Inexpensive conditioning oil. Can produce soft soap if used in large amounts.'
    },
    'lard': {
        name: 'Lard',
        sapNaOH: 0.1410,
        sapKOH: 0.1978,
        lauric: 0,
        myristic: 1,
        palmitic: 26,
        stearic: 14,
        ricinoleic: 0,
        oleic: 44,
        linoleic: 10,
        linolenic: 0,
        properties: {
            hardness: 41,
            cleansing: 1,
            conditioning: 54,
            bubbly: 1,
            creamy: 40
        },
        description: 'Traditional soap making fat. Creates hard, mild bars with creamy lather.'
    },
    'tallow': {
        name: 'Tallow',
        sapNaOH: 0.1428,
        sapKOH: 0.2003,
        lauric: 0,
        myristic: 3,
        palmitic: 27,
        stearic: 22,
        ricinoleic: 0,
        oleic: 40,
        linoleic: 3,
        linolenic: 0,
        properties: {
            hardness: 52,
            cleansing: 3,
            conditioning: 43,
            bubbly: 3,
            creamy: 49
        },
        description: 'Traditional soap making fat from beef. Hard, long-lasting bars.'
    },
    'babassu': {
        name: 'Babassu Oil',
        sapNaOH: 0.1751,
        sapKOH: 0.2456,
        lauric: 50,
        myristic: 20,
        palmitic: 11,
        stearic: 3,
        ricinoleic: 0,
        oleic: 12,
        linoleic: 2,
        linolenic: 0,
        properties: {
            hardness: 84,
            cleansing: 70,
            conditioning: 14,
            bubbly: 70,
            creamy: 14
        },
        description: 'Similar to coconut oil. Great coconut substitute for those with allergies.'
    },
    'mango butter': {
        name: 'Mango Butter',
        sapNaOH: 0.1375,
        sapKOH: 0.1929,
        lauric: 0,
        myristic: 0,
        palmitic: 8,
        stearic: 42,
        ricinoleic: 0,
        oleic: 45,
        linoleic: 4,
        linolenic: 0,
        properties: {
            hardness: 50,
            cleansing: 0,
            conditioning: 49,
            bubbly: 0,
            creamy: 50
        },
        description: 'Conditioning butter similar to shea. Creates hard, moisturizing bars.'
    },
    'rice bran': {
        name: 'Rice Bran Oil',
        sapNaOH: 0.1350,
        sapKOH: 0.1894,
        lauric: 0,
        myristic: 0,
        palmitic: 15,
        stearic: 2,
        ricinoleic: 0,
        oleic: 42,
        linoleic: 37,
        linolenic: 0,
        properties: {
            hardness: 17,
            cleansing: 0,
            conditioning: 79,
            bubbly: 0,
            creamy: 17
        },
        description: 'Similar to olive oil. Rich in vitamins and antioxidants.'
    }
};

// ============================================
// 2. SOAP CALCULATOR CLASS
// ============================================

class SoapCalculator {
    constructor() {
        this.recipe = {
            lyeType: 'NaOH',  // 'NaOH' for bar soap, 'KOH' for liquid soap
            batchSize: 0,
            oils: [],
            waterRatio: 2.5,  // Water to lye ratio (default 2.5:1)
            lyeConcentration: 33,  // Default 33% lye concentration
            superfat: 5,  // Default 5% superfat
            useWaterRatio: false,  // false = use lye concentration, true = use water:lye ratio
            fragrance: {
                percent: 0,  // % of oil weight
                ounces: 0
            }
        };
    }

    // Find oil in database (case-insensitive, partial match)
    findOil(searchTerm) {
        const normalized = searchTerm.toLowerCase().trim();
        for (const [key, data] of Object.entries(oilsDatabase)) {
            if (key === normalized ||
                data.name.toLowerCase().includes(normalized) ||
                normalized.includes(key)) {
                return { key, ...data };
            }
        }
        return null;
    }

    // Get list of all available oils
    getAvailableOils() {
        return Object.entries(oilsDatabase).map(([key, data]) => ({
            key,
            name: data.name,
            description: data.description
        }));
    }

    // Add oil to recipe
    addOil(oilName, amount, unit = 'grams') {
        const oilData = this.findOil(oilName);
        if (!oilData) {
            throw new Error(`Oil "${oilName}" not found in database. Use getAvailableOils() to see available oils.`);
        }

        // Convert to grams if needed
        let grams = amount;
        if (unit === 'oz' || unit === 'ounces') {
            grams = amount * 28.3495;
        } else if (unit === 'lbs' || unit === 'pounds') {
            grams = amount * 453.592;
        }

        this.recipe.oils.push({
            key: oilData.key,
            name: oilData.name,
            grams: grams,
            percent: 0,  // Will be calculated
            sapNaOH: oilData.sapNaOH,
            sapKOH: oilData.sapKOH,
            fatty: {
                lauric: oilData.lauric,
                myristic: oilData.myristic,
                palmitic: oilData.palmitic,
                stearic: oilData.stearic,
                ricinoleic: oilData.ricinoleic,
                oleic: oilData.oleic,
                linoleic: oilData.linoleic,
                linolenic: oilData.linolenic
            }
        });

        this.calculatePercentages();
        return this;
    }

    // Calculate percentages based on oil weights
    calculatePercentages() {
        const total = this.recipe.oils.reduce((sum, oil) => sum + oil.grams, 0);
        this.recipe.batchSize = total;
        this.recipe.oils.forEach(oil => {
            oil.percent = (oil.grams / total) * 100;
        });
    }

    // Set superfat percentage
    setSuperfat(percent) {
        if (percent < 0 || percent > 20) {
            throw new Error('Superfat should be between 0-20%. Typical range is 5-8%.');
        }
        this.recipe.superfat = percent;
        return this;
    }

    // Set lye concentration (%)
    setLyeConcentration(percent) {
        if (percent < 25 || percent > 40) {
            throw new Error('Lye concentration should be between 25-40%. Typical is 33%.');
        }
        this.recipe.lyeConcentration = percent;
        this.recipe.useWaterRatio = false;
        return this;
    }

    // Set water to lye ratio
    setWaterRatio(ratio) {
        if (ratio < 1.5 || ratio > 4) {
            throw new Error('Water:Lye ratio should be between 1.5:1 and 4:1. Typical is 2.5:1.');
        }
        this.recipe.waterRatio = ratio;
        this.recipe.useWaterRatio = true;
        return this;
    }

    // Set lye type
    setLyeType(type) {
        if (type !== 'NaOH' && type !== 'KOH') {
            throw new Error('Lye type must be "NaOH" (bar soap) or "KOH" (liquid soap).');
        }
        this.recipe.lyeType = type;
        return this;
    }

    // Calculate complete recipe
    calculate() {
        if (this.recipe.oils.length === 0) {
            throw new Error('No oils added to recipe. Add oils before calculating.');
        }

        const sapKey = this.recipe.lyeType === 'NaOH' ? 'sapNaOH' : 'sapKOH';

        // Calculate lye needed for each oil
        let totalLyeNeeded = 0;
        this.recipe.oils.forEach(oil => {
            oil.lyeNeeded = oil.grams * oil[sapKey];
            totalLyeNeeded += oil.lyeNeeded;
        });

        // Apply superfat discount
        const superfatMultiplier = 1 - (this.recipe.superfat / 100);
        const lyeAmount = totalLyeNeeded * superfatMultiplier;

        // Calculate water
        let waterAmount;
        if (this.recipe.useWaterRatio) {
            // Water = Lye × Ratio
            waterAmount = lyeAmount * this.recipe.waterRatio;
        } else {
            // Water = (Lye / Concentration) - Lye
            waterAmount = (lyeAmount / (this.recipe.lyeConcentration / 100)) - lyeAmount;
        }

        // Calculate fatty acid profile
        const fattyAcidProfile = this.calculateFattyAcidProfile();

        // Calculate soap properties
        const soapProperties = this.calculateSoapProperties(fattyAcidProfile);

        // Calculate fragrance amount if specified
        let fragranceAmount = 0;
        if (this.recipe.fragrance.percent > 0) {
            fragranceAmount = this.recipe.batchSize * (this.recipe.fragrance.percent / 100);
        } else if (this.recipe.fragrance.ounces > 0) {
            fragranceAmount = this.recipe.fragrance.ounces * 28.3495;  // Convert to grams
        }

        return {
            oils: this.recipe.oils.map(oil => ({
                name: oil.name,
                grams: Math.round(oil.grams * 10) / 10,
                ounces: Math.round(oil.grams * 0.035274 * 100) / 100,
                percent: Math.round(oil.percent * 10) / 10
            })),
            lye: {
                type: this.recipe.lyeType,
                grams: Math.round(lyeAmount * 10) / 10,
                ounces: Math.round(lyeAmount * 0.035274 * 100) / 100
            },
            water: {
                grams: Math.round(waterAmount * 10) / 10,
                ounces: Math.round(waterAmount * 0.035274 * 100) / 100,
                ratio: this.recipe.useWaterRatio ? this.recipe.waterRatio : null,
                concentration: !this.recipe.useWaterRatio ? this.recipe.lyeConcentration : null
            },
            fragrance: fragranceAmount > 0 ? {
                grams: Math.round(fragranceAmount * 10) / 10,
                ounces: Math.round(fragranceAmount * 0.035274 * 100) / 100
            } : null,
            superfat: this.recipe.superfat,
            totalBatchSize: {
                grams: Math.round(this.recipe.batchSize),
                ounces: Math.round(this.recipe.batchSize * 0.035274 * 100) / 100
            },
            fattyAcids: fattyAcidProfile,
            properties: soapProperties
        };
    }

    // Calculate weighted fatty acid profile
    calculateFattyAcidProfile() {
        const profile = {
            lauric: 0,
            myristic: 0,
            palmitic: 0,
            stearic: 0,
            ricinoleic: 0,
            oleic: 0,
            linoleic: 0,
            linolenic: 0
        };

        this.recipe.oils.forEach(oil => {
            const weight = oil.percent / 100;
            Object.keys(profile).forEach(acid => {
                profile[acid] += oil.fatty[acid] * weight;
            });
        });

        // Round to 1 decimal
        Object.keys(profile).forEach(acid => {
            profile[acid] = Math.round(profile[acid] * 10) / 10;
        });

        return profile;
    }

    // Calculate soap properties based on fatty acids
    calculateSoapProperties(fattyAcids) {
        // Hardness = Lauric + Myristic + Palmitic + Stearic
        const hardness = fattyAcids.lauric + fattyAcids.myristic +
                        fattyAcids.palmitic + fattyAcids.stearic;

        // Cleansing = Lauric + Myristic
        const cleansing = fattyAcids.lauric + fattyAcids.myristic;

        // Conditioning = Oleic + Ricinoleic + Linoleic + Linolenic
        const conditioning = fattyAcids.oleic + fattyAcids.ricinoleic +
                           fattyAcids.linoleic + fattyAcids.linolenic;

        // Bubbly = Lauric + Myristic + Ricinoleic
        const bubbly = fattyAcids.lauric + fattyAcids.myristic + fattyAcids.ricinoleic;

        // Creamy = Palmitic + Stearic + Ricinoleic
        const creamy = fattyAcids.palmitic + fattyAcids.stearic + fattyAcids.ricinoleic;

        // Iodine value (unsaturation - affects shelf life, lower is better)
        const iodine = (fattyAcids.oleic * 0.899) + (fattyAcids.linoleic * 1.814) +
                      (fattyAcids.linolenic * 2.737);

        // INS value (hardness-unsaturation, typical range 136-170)
        const ins = hardness - iodine;

        return {
            hardness: {
                value: Math.round(hardness),
                range: '29-54',
                status: this.getRangeStatus(hardness, 29, 54)
            },
            cleansing: {
                value: Math.round(cleansing),
                range: '12-22',
                status: this.getRangeStatus(cleansing, 12, 22)
            },
            conditioning: {
                value: Math.round(conditioning),
                range: '44-69',
                status: this.getRangeStatus(conditioning, 44, 69)
            },
            bubbly: {
                value: Math.round(bubbly),
                range: '14-46',
                status: this.getRangeStatus(bubbly, 14, 46)
            },
            creamy: {
                value: Math.round(creamy),
                range: '16-48',
                status: this.getRangeStatus(creamy, 16, 48)
            },
            iodine: {
                value: Math.round(iodine),
                range: '41-70',
                status: this.getRangeStatus(iodine, 41, 70),
                note: 'Lower = longer shelf life'
            },
            ins: {
                value: Math.round(ins),
                range: '136-170',
                status: this.getRangeStatus(ins, 136, 170),
                note: 'Hardness minus unsaturation'
            }
        };
    }

    // Determine if value is in range, low, or high
    getRangeStatus(value, min, max) {
        if (value < min) return 'low';
        if (value > max) return 'high';
        return 'good';
    }

    // Reset recipe
    reset() {
        this.recipe = {
            lyeType: 'NaOH',
            batchSize: 0,
            oils: [],
            waterRatio: 2.5,
            lyeConcentration: 33,
            superfat: 5,
            useWaterRatio: false,
            fragrance: { percent: 0, ounces: 0 }
        };
        return this;
    }
}

// ============================================
// 3. EXPORT FOR USE IN OTHER FILES
// ============================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SoapCalculator, oilsDatabase };
} else {
    window.SoapCalculator = SoapCalculator;
    window.oilsDatabase = oilsDatabase;
}
