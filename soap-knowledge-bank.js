/**
 * Saponify AI - Comprehensive Soap Making Knowledge Bank
 * A deep knowledge base covering all aspects of soap making
 * Last Updated: December 2025
 *
 * Sources compiled from:
 * - learnsoapmaking.com, lovelygreens.com, soapqueen.com
 * - brambleberry.com, lovinsoap.com, modernsoapmaking.com
 * - Chemistry LibreTexts, FDA, CPSC guidelines
 */

const SOAP_KNOWLEDGE_BANK = {

    // ===========================================
    // 1. SAPONIFICATION CHEMISTRY & FUNDAMENTALS
    // ===========================================
    saponification: {
        definition: `Saponification is the chemical reaction that occurs when a fat or oil (triglyceride) combines with a strong alkali (sodium hydroxide or potassium hydroxide). This process converts fats/oils into glycerol and soap.`,

        chemicalReaction: {
            equation: "Triglyceride (Fat/Oil) + Strong Base (Lye) → Soap + Glycerin",
            explanation: `Triglycerides are esters derived from glycerol and three fatty acids. When the strong base is added, it breaks the bonds holding the fatty acids and glycerol together. The fatty acids then react with the base to form soap (a salt of a fatty acid), and glycerol is released as a by-product.`,
            byproducts: ["Glycerin (moisturizing component)", "Heat (exothermic reaction)"]
        },

        sapValues: {
            definition: `The SAP (saponification) value is the number that tells you how much lye is required to saponify a specific fat or oil. Each oil has a unique SAP value.`,
            calculation: `Multiply the weight of each oil in your formula with its respective NaOH saponification coefficient to get the amount of NaOH needed.`,
            example: `For olive oil with SAP value 0.1353: 32 oz olive oil × 0.1353 = 4.33 oz lye needed`,
            importance: `If you swap oils without adjusting lye calculations, you may end up with lye-heavy (dangerous) or superfat-heavy (soft/oily) soap.`
        },

        lyeTypes: {
            sodiumHydroxide: {
                chemicalFormula: "NaOH",
                use: "Bar soap (solid soap)",
                purity: "Typically 97-99% pure for soap making"
            },
            potassiumHydroxide: {
                chemicalFormula: "KOH",
                use: "Liquid soap",
                purity: "Typically 90% pure",
                conversion: "To convert KOH SAP to NaOH SAP, multiply by 0.713"
            }
        },

        timeline: {
            mixing: "0-30 minutes - mixing oils and lye solution",
            trace: "When soap batter thickens enough to leave a trail",
            initialSaponification: "24-48 hours - majority of chemical reaction occurs",
            curing: "4-6 weeks - water evaporates, crystalline structure forms, soap becomes milder"
        }
    },

    // ===========================================
    // 2. SOAP MAKING METHODS
    // ===========================================
    methods: {
        coldProcess: {
            description: `Traditional soap-making method where lye and oils are mixed and poured into molds. Saponification takes place at room temperature (80-120°F) over time.`,

            steps: [
                "1. Measure out all ingredients by weight using a digital scale",
                "2. Prepare the lye solution: slowly add lye to water (NEVER water to lye), stir until dissolved",
                "3. Gently melt any solid oils, then add liquid oils",
                "4. Allow lye solution and oils to cool to 100-120°F",
                "5. Pour lye solution into oils",
                "6. Blend with stick blender until trace is reached",
                "7. Add fragrances, colorants, and additives at trace",
                "8. Pour into mold",
                "9. Insulate and let saponify for 24-48 hours",
                "10. Unmold and cut into bars",
                "11. Cure for 4-6 weeks"
            ],

            traceDefinition: `Trace is when the soap batter thickens to the point where drizzled soap leaves a visible trail on the surface. Light trace = thin pudding; Medium trace = custard; Heavy trace = thick pudding.`,

            cureTime: "4-6 weeks minimum (Castile soap may need 6-12 months)",

            advantages: [
                "Complete control over ingredients",
                "Can create intricate designs and swirls",
                "Full customization of recipe",
                "Longer working time than hot process"
            ],

            disadvantages: [
                "Long cure time required",
                "Must handle lye",
                "Some fragrance oils may accelerate trace"
            ]
        },

        hotProcess: {
            description: `Uses external heat (crockpot/slow cooker) to accelerate saponification. The soap is cooked at low temperatures for 1-3 hours until fully saponified.`,

            steps: [
                "1. Prepare lye solution and melt oils (same as cold process)",
                "2. Combine and blend to trace",
                "3. Cook in slow cooker on LOW for 1-3 hours",
                "4. Stir every 15-20 minutes, watching for 'volcano'",
                "5. Soap will go through stages: separation, gel phase, mashed potato texture",
                "6. Test for doneness (pH test or zap test)",
                "7. Add fragrance when cooled to ~180°F",
                "8. Scoop/pour into mold",
                "9. Allow to harden 24 hours",
                "10. Cut and cure 1-2 weeks for best results"
            ],

            stages: [
                "Separation - oils may separate initially",
                "Applesauce/Custard - batter thickens",
                "Gel Phase - becomes translucent and gel-like",
                "Vaseline Stage - glossy, waxy appearance (done!)"
            ],

            cureTime: "Usable within 24 hours, but 1-2 weeks improves quality",

            advantages: [
                "Soap is ready to use much sooner",
                "Less fragrance loss during saponification",
                "No risk of lye-heavy bars",
                "Good for rebatching problem batches"
            ],

            disadvantages: [
                "Rustic, textured appearance",
                "Limited design options",
                "Must monitor to prevent 'volcano'",
                "Requires dedicated crockpot"
            ],

            volcanoWarning: `Never leave hot process soap unattended. If it starts to rise rapidly ('volcano'), stir immediately to prevent overflow.`
        },

        meltAndPour: {
            description: `Pre-made soap base is melted, customized with additives, and poured into molds. No lye handling required as saponification is already complete.`,

            steps: [
                "1. Cut soap base into small cubes",
                "2. Melt in double boiler or microwave (30-second intervals)",
                "3. Do not exceed 160°F to prevent rubbery texture",
                "4. Add fragrance (max 3% of total weight)",
                "5. Add colorants and stir thoroughly",
                "6. Pour into mold",
                "7. Spray surface with isopropyl alcohol to eliminate bubbles",
                "8. Allow to cool and harden (2-4 hours)",
                "9. Unmold and wrap to prevent sweating"
            ],

            baseTypes: [
                "Clear glycerin - transparent, shows embeds well",
                "White/opaque - classic white appearance",
                "Goat milk - moisturizing, creamy",
                "Shea butter - extra conditioning",
                "Oatmeal - gentle exfoliation",
                "Aloe vera - soothing properties",
                "Honey - natural humectant"
            ],

            cureTime: "Ready to use immediately after hardening",

            advantages: [
                "No lye handling - safest method",
                "Quick and easy",
                "Great for beginners and kids",
                "Professional-looking results",
                "Good for embedding objects"
            ],

            disadvantages: [
                "Less control over base ingredients",
                "Can 'sweat' in humid conditions",
                "Lower fragrance tolerance than cold process",
                "More expensive per bar"
            ],

            tips: [
                "Wrap finished bars tightly to prevent glycerin sweating",
                "Use suspension bases for heavy additives",
                "Spray alcohol between layers to help them adhere"
            ]
        },

        liquidSoap: {
            description: `Made using potassium hydroxide (KOH) instead of sodium hydroxide. Creates a paste that is diluted with water to form liquid soap.`,

            process: [
                "1. Calculate recipe using KOH SAP values",
                "2. Dissolve KOH in water (caution: very exothermic)",
                "3. Combine with oils and cook in slow cooker",
                "4. Cook 3-5+ hours until fully saponified",
                "5. Perform clarity test (dissolve in water, should be clear)",
                "6. Dilute paste with distilled water (1:1 to 3:1 water:paste)",
                "7. Add preservatives if desired",
                "8. Sequester (let sit) to allow clarity"
            ],

            kohNote: `KOH is typically only 90% pure. Always verify purity and adjust calculations accordingly. KOH is hygroscopic - keep sealed to prevent moisture absorption.`
        }
    },

    // ===========================================
    // 3. OILS & FATS - PROPERTIES AND USAGE
    // ===========================================
    oils: {
        categories: {
            hardOils: {
                definition: "Oils that are solid or scoopable at room temperature. They contribute to bar hardness and quick unmolding.",
                examples: ["Coconut oil", "Palm oil", "Cocoa butter", "Shea butter", "Lard", "Tallow", "Mango butter"]
            },
            softOils: {
                definition: "Oils that are liquid at room temperature. They contribute to conditioning and moisturizing properties, but may make softer bars.",
                examples: ["Olive oil", "Sweet almond oil", "Avocado oil", "Sunflower oil", "Rice bran oil", "Castor oil", "Canola oil"]
            }
        },

        commonOils: {
            coconutOil: {
                sapValueNaOH: 0.1910,
                properties: {
                    hardness: "High",
                    cleansing: "Very High",
                    conditioning: "Low",
                    lather: "Big, fluffy bubbles"
                },
                recommendedPercentage: "20-30%",
                notes: `Coconut oil is excellent for lather and hardness but can be drying if used over 30%. High lauric acid content provides strong cleansing. For 100% coconut oil soap, use 20% superfat.`
            },

            oliveOil: {
                sapValueNaOH: 0.1353,
                properties: {
                    hardness: "Low initially, hardens over time",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Creamy, small bubbles"
                },
                recommendedPercentage: "20-100%",
                notes: `Olive oil creates mild, moisturizing soap suitable for sensitive skin. Can be used up to 100% (Castile soap) but requires extended cure time (6-12 months). Pomace olive oil traces faster than pure olive oil.`
            },

            palmOil: {
                sapValueNaOH: 0.1410,
                properties: {
                    hardness: "High",
                    cleansing: "Medium",
                    conditioning: "Medium",
                    lather: "Stable, creamy"
                },
                recommendedPercentage: "25-33%",
                notes: `Palm oil contributes to hard, long-lasting bars with stable lather. Look for RSPO-certified sustainable palm oil. Palm-free alternatives include lard, tallow, or combinations of hard butters.`
            },

            castorOil: {
                sapValueNaOH: 0.1286,
                properties: {
                    hardness: "High when cured",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Boosts and stabilizes lather"
                },
                recommendedPercentage: "5-10%",
                notes: `Castor oil is a lather booster - it doesn't create bubbles on its own but enhances and stabilizes the lather from other oils. More than 10% can make bars sticky. Excellent humectant.`
            },

            sheaButter: {
                sapValueNaOH: 0.1280,
                properties: {
                    hardness: "Medium-High",
                    cleansing: "Low",
                    conditioning: "Very High",
                    lather: "Creamy, stable"
                },
                recommendedPercentage: "5-15%",
                notes: `Shea butter is rich in vitamins A, E, and F. Highly moisturizing and creates a luxurious feel. Can accelerate trace at higher percentages.`
            },

            cocoaButter: {
                sapValueNaOH: 0.1378,
                properties: {
                    hardness: "Very High",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Minimal contribution"
                },
                recommendedPercentage: "5-15%",
                notes: `Cocoa butter creates very hard bars and adds a subtle chocolate scent. Good for balancing soft oils. Can cause acceleration and brittle bars if overused.`
            },

            sweetAlmondOil: {
                sapValueNaOH: 0.1360,
                properties: {
                    hardness: "Low",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Creamy"
                },
                recommendedPercentage: "5-20%",
                notes: `Mild and moisturizing, good for sensitive skin. Can be substituted for a portion of olive oil for slightly different feel.`
            },

            avocadoOil: {
                sapValueNaOH: 0.1330,
                properties: {
                    hardness: "Low",
                    cleansing: "Low",
                    conditioning: "Very High",
                    lather: "Creamy, stable"
                },
                recommendedPercentage: "5-15%",
                notes: `Rich in vitamins and excellent for dry/mature skin. Slow-tracing oil good for swirl designs. Green color may tint soap.`
            },

            sunflowerOil: {
                sapValueNaOH: 0.1360,
                properties: {
                    hardness: "Low",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Minimal"
                },
                recommendedPercentage: "5-15%",
                notes: `Budget-friendly conditioning oil. High linoleic version preferred (prone to less rancidity). Good olive oil substitute.`
            },

            riceBranOil: {
                sapValueNaOH: 0.1280,
                properties: {
                    hardness: "Low-Medium",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Creamy"
                },
                recommendedPercentage: "10-20%",
                notes: `Contains natural antioxidants (gamma oryzanol). Good all-purpose oil with nice skin feel. Slow tracing.`
            },

            lard: {
                sapValueNaOH: 0.1399,
                properties: {
                    hardness: "High",
                    cleansing: "Low",
                    conditioning: "High",
                    lather: "Creamy, stable"
                },
                recommendedPercentage: "25-50%",
                notes: `Traditional soap-making fat. Creates hard, conditioning bars with creamy lather. Excellent palm oil alternative.`
            },

            tallow: {
                sapValueNaOH: 0.1405,
                properties: {
                    hardness: "Very High",
                    cleansing: "Low",
                    conditioning: "Medium-High",
                    lather: "Creamy, stable"
                },
                recommendedPercentage: "25-50%",
                notes: `Beef fat that creates very hard, long-lasting bars. Traditional soap ingredient. Good palm oil substitute.`
            }
        },

        oilSubstitutions: {
            forOliveOil: ["Rice bran oil", "Sweet almond oil", "Sunflower oil", "Canola oil"],
            forPalmOil: ["Lard", "Tallow", "Cocoa butter + shea butter combination", "Mango butter"],
            forCoconutOil: ["Babassu oil", "Palm kernel oil"],
            forSheaButter: ["Mango butter", "Kokum butter", "Cocoa butter"]
        }
    },

    // ===========================================
    // 4. RECIPE FORMULATION
    // ===========================================
    formulation: {
        classicRatios: {
            beginner343333: {
                name: "Classic Beginner Recipe (34/33/33)",
                recipe: {
                    oliveOil: "34%",
                    coconutOil: "33%",
                    palmOil: "33%"
                },
                notes: "Well-balanced soap with good lather, hardness, and conditioning"
            },

            beginner30301010: {
                name: "Enhanced Beginner Recipe (30/30/30/10)",
                recipe: {
                    oliveOil: "30%",
                    coconutOil: "30%",
                    palmOil: "30%",
                    specialOil: "10% (shea butter, sweet almond, or castor oil)"
                },
                notes: "Adds extra skin benefits while maintaining balance"
            },

            palmFree: {
                name: "Palm-Free Recipe",
                recipe: {
                    oliveOil: "44%",
                    coconutOil: "24%",
                    sheaButter: "8%",
                    cocoaButter: "8%",
                    canola: "12%",
                    castorOil: "4%"
                },
                notes: "Sustainable option without palm oil"
            },

            castileSoap: {
                name: "Castile Soap (100% Olive Oil)",
                recipe: {
                    oliveOil: "100%"
                },
                notes: "Extremely mild and moisturizing. Requires 6-12 months cure time. Low lather initially but improves with age."
            },

            bastileSoap: {
                name: "Bastile Soap (High Olive Oil)",
                recipe: {
                    oliveOil: "70%",
                    coconutOil: "20%",
                    castorOil: "10%"
                },
                notes: "Gentler than basic recipes with good lather. Faster to cure than pure Castile."
            }
        },

        superfat: {
            definition: `Superfatting (or lye discount) means using less lye than required to fully saponify all oils, leaving unsaponified oils in the finished soap for extra moisturizing.`,

            recommendations: {
                standard: "5% - Good balance for most recipes",
                sensitive: "6-8% - More moisturizing for dry/sensitive skin",
                shampooBar: "3-5% - Less residue for hair",
                laundry: "0-2% - Minimum superfat to prevent residue",
                coconut100: "15-20% - High superfat needed due to very cleansing nature"
            },

            calculation: `To calculate: Take the total lye needed for 0% superfat, multiply by (1 - desired superfat percentage). Example: 100g lye × (1 - 0.05) = 95g lye for 5% superfat.`
        },

        waterCalculations: {
            fullWater: {
                ratio: "1:2.5 (lye:water)",
                percentage: "28.6% lye concentration",
                use: "Maximum working time, good for beginners"
            },
            moderateDiscount: {
                ratio: "1:2 (lye:water)",
                percentage: "33.3% lye concentration",
                use: "Good balance of working time and faster unmolding"
            },
            strongDiscount: {
                ratio: "1:1.5 (lye:water)",
                percentage: "40% lye concentration",
                use: "Faster unmolding, reduced soda ash risk"
            },
            safeRange: "Water:lye ratios between 3:1 (25% lye) and 1:1 (50% lye) are workable. Never go below 1:1 as lye won't fully dissolve."
        },

        soapProperties: {
            hardness: {
                description: "How firm/hard the bar will be",
                idealRange: "29-54",
                increasedBy: ["Palm oil", "Tallow", "Lard", "Cocoa butter", "Sodium lactate"]
            },
            cleansing: {
                description: "How well soap removes oils (too high = stripping/drying)",
                idealRange: "12-22",
                increasedBy: ["Coconut oil", "Babassu oil", "Palm kernel oil"]
            },
            conditioning: {
                description: "How moisturizing/emollient the soap feels",
                idealRange: "44-69",
                increasedBy: ["Olive oil", "Shea butter", "Avocado oil", "Sweet almond oil", "Higher superfat"]
            },
            bubblyLather: {
                description: "Big, fluffy bubbles",
                idealRange: "14-46",
                increasedBy: ["Coconut oil", "Castor oil (as booster)", "Sugar"]
            },
            creamyLather: {
                description: "Dense, creamy lather",
                idealRange: "16-48",
                increasedBy: ["Olive oil", "Palm oil", "Lard", "Tallow"]
            },
            longevity: {
                description: "How long the bar lasts in use",
                idealRange: "25-55",
                increasedBy: ["Hard oils", "Proper curing", "Lower water content"]
            }
        }
    },

    // ===========================================
    // 5. ADDITIVES - COLORANTS, SCENTS, BOTANICALS
    // ===========================================
    additives: {
        naturalColorants: {
            purples: {
                alkanetRoot: {
                    color: "Purple to blue-purple",
                    method: "Oil infusion (oil-soluble)",
                    usage: "1-2 tsp infused oil per pound of oils",
                    notes: "Must be infused in oil; water-based use creates gray specks"
                },
                ratanjot: {
                    color: "Purple",
                    method: "Oil infusion",
                    usage: "Similar to alkanet"
                }
            },
            blues: {
                indigo: {
                    color: "Denim blue to navy",
                    method: "Add powder at trace",
                    usage: "1/8 - 1/2 tsp per pound of oils",
                    notes: "A little goes a VERY long way. Mix with oil first."
                },
                woad: {
                    color: "Lighter blue",
                    method: "Add powder at trace",
                    usage: "1/4 - 1 tsp per pound of oils"
                }
            },
            greens: {
                spirulina: {
                    color: "Blue-green",
                    usage: "1/2 - 2 tsp per pound",
                    notes: "May fade over time"
                },
                frenchGreenClay: {
                    color: "Sage green",
                    usage: "1 tsp per pound",
                    notes: "Mix with 3x water first to prevent cracking"
                },
                chlorophyll: {
                    color: "Bright green",
                    notes: "Can fade; not as stable as clays"
                }
            },
            pinks: {
                madderRoot: {
                    color: "Pink to deep red (depending on amount)",
                    method: "Powder at trace or oil infusion",
                    usage: "1/2 - 2 tsp per pound",
                    notes: "Gelling intensifies color"
                },
                roseClay: {
                    color: "Soft pink",
                    usage: "1 tsp per pound"
                }
            },
            yellows: {
                turmeric: {
                    color: "Bright yellow to gold",
                    usage: "1/4 - 1 tsp per pound",
                    notes: "Very potent; start small. May stain skin initially."
                },
                annatto: {
                    color: "Yellow to orange",
                    method: "Oil infusion or powder",
                    usage: "1 tsp infused oil per pound"
                },
                calendula: {
                    color: "Yellow-gold",
                    notes: "One of the only flowers that holds color in soap"
                }
            },
            browns: {
                cocoaPowder: {
                    color: "Brown",
                    usage: "1-2 tsp per pound"
                },
                cinnamonPowder: {
                    color: "Brown with speckles",
                    notes: "Can be sensitizing; use caution",
                    usage: "1/2 - 1 tsp per pound"
                },
                coffee: {
                    color: "Tan to brown",
                    method: "Replace water with strong brewed coffee"
                }
            },
            blackGray: {
                activatedCharcoal: {
                    color: "Gray to black",
                    usage: "1/4 - 1 tsp per pound",
                    notes: "Very messy; can make black lather initially"
                }
            },
            clays: {
                types: [
                    "Kaolin (white) - mild, good for sensitive skin",
                    "Bentonite (gray-green) - detoxifying",
                    "French Green - mineral-rich",
                    "Rose (pink) - gentle coloring",
                    "Rhassoul (brown) - cleansing"
                ],
                usage: "1 tsp per pound of oils, mixed with 3x water to prevent cracking"
            }
        },

        fragrance: {
            essentialOils: {
                usageRate: "0.5-0.7 oz per pound of oils (3-5% of oil weight)",
                maxRate: "Follow IFRA guidelines for each oil",
                notes: `Calculate based on oil weight, not total soap weight. Some EOs (citrus) fade quickly; others (patchouli, cedarwood) are more stable.`,

                categories: {
                    topNotes: {
                        description: "Light, fresh, fade fastest (1-2 hours)",
                        examples: ["Citrus oils", "Peppermint", "Eucalyptus", "Bergamot"]
                    },
                    middleNotes: {
                        description: "Heart of the blend (2-4 hours)",
                        examples: ["Lavender", "Rosemary", "Geranium", "Chamomile"]
                    },
                    baseNotes: {
                        description: "Deep, grounding, last longest (4+ hours)",
                        examples: ["Patchouli", "Cedarwood", "Sandalwood", "Vetiver", "Vanilla"]
                    }
                },

                anchoring: "Add base notes (patchouli, sandalwood) to help fix lighter scents",

                safetyNotes: {
                    photosensitizing: ["Bergamot", "Lime (cold-pressed)", "Lemon", "Grapefruit"],
                    skinSensitizing: ["Cinnamon bark", "Clove", "Oregano", "Thyme"],
                    pregnancyAvoid: ["Clary sage", "Rosemary", "Sage"]
                }
            },

            fragranceOils: {
                usageRate: "0.7 oz per pound of oils for cold process",
                meltAndPour: "0.3 oz per pound (max 3% of total weight)",
                notes: `Fragrance oils are synthetic blends designed for soap. They generally have better scent retention than essential oils. Always check that FO is soap-safe.`,

                behaviors: {
                    accelerating: "Florals, spices, and some bakery scents may thicken soap quickly",
                    ricing: "Some FOs cause oils to form small lumps",
                    discoloring: "Vanilla content causes browning over time"
                }
            }
        },

        exfoliants: {
            gentle: ["Ground oatmeal", "Poppy seeds", "Cornmeal", "Ground lavender"],
            medium: ["Coffee grounds (used)", "Coconut flakes", "Loofah pieces"],
            heavy: ["Pumice", "Ground walnut shells", "Apricot kernel"],
            usage: "1-2 tbsp per pound of soap base",
            notes: "Exfoliants can sink in thin batter; add at medium trace in cold process"
        },

        botanicals: {
            holdColor: ["Calendula petals", "Chamomile flowers"],
            turnBrown: ["Lavender buds", "Rose petals", "Most dried flowers"],
            whenToAdd: {
                lyeInfusion: "Steep herbs in water, strain, use liquid for lye solution",
                oilInfusion: "Steep herbs in warm oil for weeks before using",
                atTrace: "Stir whole or ground botanicals into soap at trace",
                topDecoration: "Sprinkle on top of freshly poured soap"
            },
            allergenWarning: "Botanicals can increase allergen risk. Include on labels."
        },

        specialAdditives: {
            sodiumLactate: {
                use: "Harder bars, faster unmolding",
                rate: "1 tsp per pound of oils",
                addTo: "Cooled lye solution"
            },
            sugar: {
                use: "Increases bubbles/lather",
                rate: "1 tsp per pound of oils",
                addTo: "Lye water (dissolve completely)"
            },
            salt: {
                use: "Harder bars",
                rate: "1/2 - 1 tsp per pound",
                notes: "Add to lye water; too much inhibits lather"
            },
            honey: {
                use: "Moisturizing, boost bubbles",
                rate: "1 tsp per pound",
                notes: "Add at thin trace; causes heating/browning"
            },
            milks: {
                types: ["Goat milk", "Coconut milk", "Buttermilk", "Oat milk"],
                method: "Freeze milk, use in place of water for lye solution",
                notes: "Add lye slowly to prevent scorching. Milk sugars accelerate trace and cause natural browning."
            },
            yogurt: {
                use: "Creamier bars, adds lactic acid",
                rate: "1-2 tbsp per pound",
                notes: "Can help achieve fluid hot process"
            },
            oatmeal: {
                use: "Soothing, gentle exfoliation",
                method: "Colloidal oatmeal dissolves; whole oats provide texture",
                rate: "1-2 tbsp per pound"
            },
            vitaminE: {
                use: "Antioxidant, extends shelf life",
                rate: "1 tsp per pound of oils"
            },
            ROE: {
                fullName: "Rosemary Oleoresin Extract",
                use: "Prevents rancidity (DOS)",
                rate: "0.1% of oil weight"
            }
        }
    },

    // ===========================================
    // 6. SAFETY GUIDELINES
    // ===========================================
    safety: {
        personalProtection: {
            eyes: {
                equipment: "Safety goggles or wrap-around glasses with strap",
                importance: "Lye splash can cause blindness",
                notes: "Regular glasses are NOT sufficient"
            },
            hands: {
                equipment: "Long rubber gloves (not thin latex)",
                importance: "Protects from lye burns",
                notes: "Tuck sleeves into gloves"
            },
            body: {
                equipment: "Long sleeves, long pants, closed-toe shoes, apron",
                importance: "No exposed skin",
                notes: "Natural fibers (cotton) preferred - won't melt"
            },
            lungs: {
                equipment: "Work in well-ventilated area or use respirator",
                importance: "Lye fumes are caustic",
                notes: "Mix lye outdoors or by open window if possible"
            }
        },

        lyeHandling: {
            goldenRule: "ALWAYS add lye TO water, NEVER water to lye",
            reason: "Adding water to lye can cause violent boiling and splashing",

            steps: [
                "1. Wear all safety gear before handling lye",
                "2. Use room temperature or cool distilled water",
                "3. Place container in sink or on protected surface",
                "4. Slowly pour lye into water while stirring",
                "5. Stir until completely dissolved",
                "6. Allow to cool in safe location (reaches 200°F+)",
                "7. Never leave unattended where children/pets can reach"
            ],

            materials: {
                safe: ["Stainless steel", "Heat-resistant glass (Pyrex)", "High-density polyethylene (HDPE)", "Polypropylene (PP)"],
                avoid: ["Aluminum (reacts violently)", "Tin", "Non-tempered glass", "Thin plastics"]
            }
        },

        workspace: {
            requirements: [
                "Dedicated soap-making equipment (never use for food)",
                "Clear, clutter-free work surface",
                "Good ventilation",
                "Away from children and pets",
                "Access to running water",
                "Paper towels and white vinegar nearby"
            ],
            cleanup: "Raw soap batter is still caustic for 24-48 hours. Wear gloves during cleanup.",
            spillResponse: "Wipe up lye spills immediately with paper towels, dispose safely. Neutralize residue with vinegar."
        },

        firstAid: {
            skinExposure: [
                "1. Immediately flush with cool water for 15+ minutes",
                "2. Remove contaminated clothing",
                "3. If irritation persists, seek medical attention"
            ],
            eyeExposure: [
                "1. Immediately flush eyes with water for 15-20 minutes",
                "2. Hold eyelids open while flushing",
                "3. Seek medical attention IMMEDIATELY"
            ],
            ingestion: [
                "1. Do NOT induce vomiting",
                "2. Rinse mouth with water",
                "3. Drink small amounts of water or milk",
                "4. Call Poison Control immediately",
                "5. Seek emergency medical care"
            ],
            poisonControl: "1-800-222-1222 (US)"
        },

        lyeStorage: {
            container: "Keep in original container or clearly labeled airtight container",
            location: "Cool, dry place away from moisture",
            keepAway: "Children, pets, food items",
            note: "Lye is hygroscopic (absorbs moisture). Wet lye can damage container."
        }
    },

    // ===========================================
    // 7. TROUBLESHOOTING
    // ===========================================
    troubleshooting: {
        lyeHeavySoap: {
            symptoms: ["Burns or tingles on tongue (zap test)", "pH above 10", "Hard/crumbly texture", "White pockets in soap"],
            causes: ["Mismeasured lye", "Mismeasured oils", "Forgot an oil", "Faulty scale"],
            testing: {
                zapTest: "Touch soap to tongue - if it 'zaps' (sharp tingle), it's lye heavy",
                phTest: "pH strip test - should be 9-10. Over 11 indicates lye heavy",
                timing: "Wait 5 days before testing to allow saponification to complete"
            },
            solutions: ["Rebatch with additional oils", "Use for laundry soap (shred and add washing soda)"]
        },

        softSoap: {
            symptoms: ["Won't unmold after 2-3 days", "Remains squishy after weeks", "Dents easily"],
            causes: [
                "Too much water in recipe",
                "Not enough hard oils",
                "Not enough lye (under-lyed)",
                "Didn't reach trace",
                "High humidity environment"
            ],
            solutions: [
                "Wait longer - soft recipes may need 4-6 weeks",
                "Add sodium lactate (1 tsp/lb) to future batches",
                "Rebatch to remove excess water",
                "Use more hard oils in recipe",
                "Force gel phase to speed saponification"
            ]
        },

        seizing: {
            symptoms: ["Soap suddenly becomes solid/thick", "Can't pour from pot", "'Soap on a stick'"],
            causes: [
                "Fragrance oil reaction",
                "Temperatures too cold (oils solidifying)",
                "High percentage of hard oils/butters",
                "Stick blended too long"
            ],
            solutions: [
                "Work quickly - scoop into mold immediately",
                "Hot process rescue - cook in crockpot until fully saponified",
                "Test fragrances in small batches first"
            ],
            prevention: [
                "Use fragrances tested for cold process",
                "Work at proper temperatures (100-120°F)",
                "Don't over-blend",
                "Mix fragrance with equal part liquid oil, warm slightly before adding"
            ]
        },

        acceleration: {
            symptoms: ["Soap thickens very quickly", "Less extreme than seizing", "Limited time for design"],
            causes: ["Certain fragrance oils", "High temperatures", "High hard oil percentage", "Over-blending"],
            solutions: [
                "Stop stick blending, switch to whisk",
                "Work quickly with thick batter",
                "Use spoon-plop or simple designs",
                "Lower working temperatures next time"
            ]
        },

        ricing: {
            symptoms: ["Small, rice-like lumps in batter", "Curdled appearance"],
            causes: ["Fragrance oil reaction with hard oils", "Temperature issues"],
            solutions: [
                "Stick blend vigorously to smooth out",
                "Continue with thicker batter",
                "Hot process if severe"
            ]
        },

        separation: {
            symptoms: ["Oil pooling on top", "Layers forming", "Oily rivers in finished soap"],
            causes: ["Didn't reach true trace", "Fragrance reaction", "Temperature mismatch between lye and oils"],
            solutions: [
                "Stick blend more to re-emulsify",
                "Hot process if won't combine",
                "Ensure temps are within 10°F of each other"
            ]
        },

        sodaAsh: {
            symptoms: ["White powdery coating on soap surface", "Harmless but cosmetically undesirable"],
            causes: ["Unsaponified lye reacting with CO2 in air"],
            prevention: [
                "Soap at higher temperatures (120-140°F)",
                "Force gel phase",
                "Spray with 99% isopropyl alcohol",
                "Cover mold with plastic wrap",
                "Use water discount"
            ],
            removal: ["Steam with fabric steamer", "Wash off with water", "Wipe with alcohol"]
        },

        dreadedOrangeSpots: {
            abbreviation: "DOS",
            symptoms: ["Orange or rust-colored spots", "May appear weeks/months after making", "Rancid smell"],
            causes: ["Rancid oils", "Old oils", "Contaminated water", "Poor storage"],
            prevention: [
                "Use fresh oils from reputable suppliers",
                "Use distilled water only",
                "Add antioxidant (vitamin E or ROE)",
                "Store cured soap in cool, dry, dark place",
                "Ensure good airflow during curing"
            ],
            note: "Soap with DOS is still usable but may smell off"
        },

        glycerinRivers: {
            symptoms: ["Clear/translucent rivers or squiggles through soap"],
            causes: ["Overheating during gel phase", "Water discount with forced gel"],
            note: "Purely cosmetic - soap is still perfectly usable",
            prevention: ["Don't insulate too heavily", "Watch temperatures during gel"]
        },

        cracking: {
            symptoms: ["Cracks on surface or through soap"],
            causes: ["Overheating", "Too much hard oil/butter", "Clay not properly dispersed"],
            prevention: [
                "Don't over-insulate",
                "Balance hard and soft oils",
                "Mix clays with water before adding"
            ]
        },

        partialGel: {
            symptoms: ["Ring pattern - center darker/more translucent than edges"],
            causes: ["Uneven temperature during saponification"],
            note: "Cosmetic issue only - soap is fine to use",
            prevention: ["Force full gel (insulate) or prevent gel entirely (refrigerate)"]
        }
    },

    // ===========================================
    // 8. CURING & STORAGE
    // ===========================================
    curingAndStorage: {
        whyCure: {
            saponification: "Allows remaining 1-5% of lye to finish converting (cold process)",
            waterEvaporation: "Bars lose up to 50% of water content, becoming harder and longer-lasting",
            crystallineStructure: "Soap molecules organize into tighter crystalline structure",
            mildness: "pH drops and soap becomes gentler over time",
            lather: "Improves dramatically with cure time"
        },

        cureTime: {
            coldProcess: "4-6 weeks minimum",
            highOliveOil: "6-12 weeks (or longer for Castile)",
            hotProcess: "1-2 weeks (already saponified but benefits from drying)",
            meltAndPour: "Ready immediately after hardening"
        },

        curingMethod: {
            environment: "Cool, dry area with good airflow",
            setup: "Place bars on rack or cardboard, not touching each other",
            position: "Stand on smallest edge for maximum air exposure",
            rotation: "Turn bars occasionally for even drying",
            humidity: "Use dehumidifier in humid climates",
            avoid: ["Direct sunlight", "Aluminum surfaces", "Sealed containers", "High humidity"]
        },

        howToKnowWhenCured: {
            weightTest: "Weigh weekly - when weight stabilizes for 2 weeks, soap is cured",
            feelTest: "Should feel hard, not tacky or soft",
            zapTest: "Should not zap tongue (cold process)"
        },

        storage: {
            curedSoap: "Wrap in paper or breathable fabric. Store in cool, dry, dark place.",
            avoidPlastic: "Plastic traps moisture and can cause sweating",
            shelfLife: "Properly stored soap lasts 1-2 years",
            note: "Soap improves with age - many soapers cure 6+ months for premium bars"
        }
    },

    // ===========================================
    // 9. DESIGN TECHNIQUES
    // ===========================================
    designTechniques: {
        swirls: {
            dropSwirl: {
                difficulty: "Beginner",
                description: "Pour alternating colors into mold in drops, creating random pattern",
                trace: "Thin to medium",
                tips: "Rotate colors each pour; height of drop affects spread"
            },
            inThePotSwirl: {
                difficulty: "Beginner",
                description: "Add colors directly to pot and lightly swirl before pouring",
                trace: "Thin to medium",
                tips: "Don't over-stir; 2-3 swirls is enough"
            },
            hangerSwirl: {
                difficulty: "Beginner-Intermediate",
                description: "Pour layers, then drag hanger tool through to create swirls",
                trace: "Thin to medium",
                tips: "Slow, deliberate movements; wire hanger or chopstick works"
            },
            taiwanSwirl: {
                difficulty: "Intermediate",
                description: "Linear pour with zig-zag hanger swirl pattern",
                trace: "Thin",
                tips: "Creates feather-like pattern"
            },
            spinSwirl: {
                difficulty: "Advanced",
                description: "Spin filled mold to create spiral pattern",
                trace: "Very thin (emulsion)",
                tips: "Requires extremely thin batter; work quickly"
            },
            columnPour: {
                difficulty: "Intermediate",
                description: "Pour colors over a column (PVC pipe) into center of mold",
                trace: "Thin",
                tips: "Creates circular ring patterns"
            }
        },

        layers: {
            horizontalLayers: {
                description: "Pour one color, let set slightly, pour next",
                tips: "Allow thin skin to form between layers; spray alcohol to help adhesion"
            },
            verticalLayers: {
                description: "Use dividers or pour carefully to create vertical sections",
                tips: "Dividers must be removed carefully"
            },
            tiltedLayers: {
                description: "Prop mold at angle, pour, change angle, repeat",
                tips: "Creates diagonal line effects"
            }
        },

        tipsForSuccess: {
            oils: "Use 60%+ slow-tracing oils (olive, rice bran, avocado) for swirl designs",
            fragrance: "Avoid accelerating fragrances or soap at low temp without fragrance",
            temperature: "Lower temps (95-105°F) give more working time",
            blending: "Use stick blender in short bursts; switch to whisk for final mixing",
            colorMixing: "Disperse colorants in oil before adding to batter",
            trace: "Thinner trace = more movement/blending; thicker trace = more defined lines"
        }
    },

    // ===========================================
    // 10. BUSINESS & REGULATIONS
    // ===========================================
    businessRegulations: {
        productClassification: {
            trueSoap: {
                definition: "Product where bulk of non-volatile matter is alkali salts of fatty acids, detergent properties come from these compounds, and it's labeled/marketed ONLY as soap",
                regulatedBy: "Consumer Product Safety Commission (CPSC)",
                requirements: "Basic labeling: product name, net weight, business info"
            },
            cosmetic: {
                definition: "Product intended for cleansing, beautifying, promoting attractiveness, or altering appearance (moisturizing claims, deodorizing, fragrance)",
                regulatedBy: "FDA",
                requirements: "Full ingredient list in descending order, all cosmetic labeling requirements"
            },
            drug: {
                definition: "Product making claims to treat or prevent disease (acne, eczema, antibacterial)",
                regulatedBy: "FDA",
                requirements: "Pre-market approval, drug facts label, extensive testing"
            }
        },

        labelingRequirements: {
            trueSoap: {
                required: ["Product name/identity", "Net weight", "Business name and address"],
                optional: "Ingredients (recommended for transparency)"
            },
            cosmetic: {
                required: [
                    "Product identity/name",
                    "Net weight/contents",
                    "Business name and address (including ZIP)",
                    "Ingredient list in descending order by weight",
                    "Colorants listed by name at end"
                ],
                warnings: "Any required safety warnings"
            }
        },

        claimsToAvoid: {
            cosmeticClaims: ["Moisturizing", "Anti-aging", "Softening skin", "Making skin smell nice"],
            drugClaims: ["Treats acne", "Antibacterial", "Heals eczema", "Cures any condition"],
            note: "These claims change product classification and regulatory requirements"
        },

        insurance: {
            productLiability: "Essential for any soap seller",
            coverage: "Typically $1-2 million recommended",
            providers: "Specialty insurance for handmade products available (Indie Business Network, Handcrafted Soap & Cosmetic Guild)"
        },

        childrenProducts: {
            definition: "Products intended for children 12 and under",
            requirements: ["Third-party lead testing", "Permanent tracking label", "Additional CPSC requirements"]
        },

        resourceLinks: {
            fda: "FDA.gov - Cosmetics guidance",
            cpsc: "CPSC.gov - Soap guidance",
            ftc: "FTC.gov - Fair Packaging and Labeling Act"
        }
    },

    // ===========================================
    // 11. POPULAR RECIPES
    // ===========================================
    popularRecipes: {
        lavenderOatmeal: {
            name: "Lavender Oatmeal Soap",
            oils: {
                oliveOil: "40%",
                coconutOil: "25%",
                palmOil: "25%",
                castorOil: "5%",
                sheaButter: "5%"
            },
            additives: [
                "Lavender essential oil: 0.7 oz per pound",
                "Colloidal oatmeal: 1 tbsp per pound",
                "Dried lavender buds for top (decorative)"
            ],
            superfat: "5%",
            notes: "Soothing and gentle, great for sensitive skin"
        },

        goatMilkHoney: {
            name: "Goat Milk & Honey Soap",
            oils: {
                oliveOil: "45%",
                coconutOil: "30%",
                avocadoOil: "15%",
                castorOil: "10%"
            },
            additives: [
                "Frozen goat milk (replace water)",
                "Honey: 1 tsp per pound at trace",
                "Optional: oatmeal for texture"
            ],
            superfat: "5%",
            notes: "Freeze milk before adding lye. Add lye very slowly to prevent scorching. Expect caramel coloring."
        },

        coffeeScrub: {
            name: "Coffee Scrub Soap",
            oils: {
                oliveOil: "35%",
                coconutOil: "30%",
                palmOil: "25%",
                castorOil: "10%"
            },
            additives: [
                "Strong brewed coffee (replace water)",
                "Used coffee grounds: 1-2 tbsp per pound at trace",
                "Optional: coffee fragrance oil"
            ],
            superfat: "5%",
            notes: "Great exfoliating soap. Coffee grounds are less scratchy after brewing."
        },

        charcoalTea: {
            name: "Activated Charcoal & Tea Tree",
            oils: {
                oliveOil: "40%",
                coconutOil: "25%",
                palmOil: "20%",
                castorOil: "10%",
                sheaButter: "5%"
            },
            additives: [
                "Activated charcoal: 1/2 tsp per pound",
                "Tea tree essential oil: 0.6 oz per pound"
            ],
            superfat: "5%",
            notes: "Good for oily/acne-prone skin (don't make claims). Charcoal can make gray lather initially."
        },

        sheaButterLuxury: {
            name: "Shea Butter Luxury Bar",
            oils: {
                oliveOil: "35%",
                coconutOil: "25%",
                sheaButter: "15%",
                avocadoOil: "15%",
                castorOil: "10%"
            },
            additives: [
                "Fragrance of choice: 0.7 oz per pound"
            ],
            superfat: "6%",
            notes: "Highly moisturizing, creamy lather"
        },

        pineTarSoap: {
            name: "Traditional Pine Tar Soap",
            oils: {
                oliveOil: "40%",
                coconutOil: "25%",
                lard: "25%",
                castorOil: "10%"
            },
            additives: [
                "Pine tar: 1-2 tbsp per pound at trace"
            ],
            superfat: "5%",
            notes: "Traditional remedy soap. Pine tar accelerates trace significantly - work quickly."
        }
    },

    // ===========================================
    // 12. GLOSSARY OF TERMS
    // ===========================================
    glossary: {
        acceleration: "When soap batter thickens faster than expected, often caused by fragrance oils or high temperatures",
        bastile: "Soap made with high percentage of olive oil (typically 70%+) but not 100%",
        castile: "Soap made with 100% olive oil",
        coldProcess: "Soap making method where saponification occurs at room temperature over several weeks",
        cure: "The process of allowing soap to dry and harden over 4-6 weeks after making",
        DOS: "Dreaded Orange Spots - rancidity appearing as orange spots on soap",
        emulsion: "The initial stage of mixing lye solution and oils before reaching trace",
        gelPhase: "Stage during saponification where soap heats up and becomes translucent; results in brighter colors and faster saponification",
        hotProcess: "Soap making method using external heat to accelerate saponification",
        lye: "Sodium hydroxide (NaOH) for bar soap or potassium hydroxide (KOH) for liquid soap",
        lyeDiscount: "Same as superfat - using less lye than needed to convert all oils",
        meltAndPour: "Pre-made soap base that is melted and customized",
        rebatch: "Process of grating and re-melting soap to fix problems or add ingredients",
        ricing: "When fragrance causes small rice-like lumps in soap batter",
        saponification: "Chemical reaction converting fats/oils and lye into soap and glycerin",
        sapValue: "Saponification value - amount of lye needed to convert specific oil to soap",
        seizing: "When soap suddenly solidifies in pot, becoming unworkable",
        sodaAsh: "White powdery coating on soap surface from lye reacting with air",
        soaping: "The process of making soap",
        superfat: "Percentage of oils left unsaponified for extra moisturizing properties",
        trace: "When soap batter thickens enough that drizzled soap leaves a trail on surface",
        unmolding: "Removing soap from mold after it has hardened enough",
        waterDiscount: "Using less water than the standard amount to speed unmolding and reduce soda ash"
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SOAP_KNOWLEDGE_BANK;
}
