# Saponify AI Soap Calculator Guide

## Overview

Saponify AI now includes a comprehensive soap calculator engine modeled after SoapCalc.net functionality. This ensures users receive **accurate, safe soap recipes** with precise lye calculations and detailed soap property analysis.

## Features

### ✅ Complete Implementation

1. **19 Oils Database** with accurate SAP values and fatty acid profiles
2. **Automatic Property Calculation** (Hardness, Cleansing, Conditioning, Bubbly, Creamy, Iodine, INS)
3. **Flexible Input Options** (grams, ounces, pounds, percentages)
4. **Customizable Settings** (superfat %, lye concentration, water:lye ratio)
5. **Safety Warnings** included in all recipe outputs
6. **NaOH and KOH Support** (bar soap and liquid soap)

## How It Works

### Architecture

```
saponifyai.html
├── soap-calculator.js   (Calculator engine with oils database)
├── ai-config.js         (AI system prompt with calculator knowledge)
└── soap-chat.js         (Chat integration and formatting)
```

### User Workflow

1. **User requests a recipe** (e.g., "Create a beginner soap recipe for 500g")
2. **AI asks clarifying questions** (oils, batch size, superfat %, etc.)
3. **AI uses SoapCalculator** to calculate exact amounts
4. **Formatted recipe displayed** with:
   - All oils with amounts (grams, ounces, percentages)
   - Exact lye amount (NaOH or KOH)
   - Exact water amount
   - Complete soap properties with range indicators
   - Safety warnings

## Available Oils (19 Total)

### Common Oils
- Olive Oil - Gentle, conditioning
- Coconut Oil - Hard bars, fluffy lather
- Palm Oil - Balanced hardness/conditioning
- Castor Oil - Boosts lather (use 5-10%)
- Sweet Almond Oil - Luxurious, moisturizing

### Butters
- Shea Butter - Hard, conditioning
- Cocoa Butter - Very hard, creamy
- Mango Butter - Similar to shea

### Specialty Oils
- Avocado, Sunflower, Grapeseed, Jojoba
- Hemp Seed, Apricot Kernel, Rice Bran, Canola
- Lard, Tallow (animal fats)
- Babassu Oil (coconut substitute)

## Soap Properties Explained

Based on fatty acid composition, calculated automatically:

### Hardness (29-54)
- Formula: Lauric + Myristic + Palmitic + Stearic
- Physical bar hardness, ease of unmolding

### Cleansing (12-22)
- Formula: Lauric + Myristic
- **NOT cleaning power** - water solubility
- High values can dry skin (keep coconut ≤35%)

### Conditioning (44-69)
- Formula: Oleic + Ricinoleic + Linoleic + Linolenic
- Moisturizing, "anti-dry" property

### Bubbly Lather (14-46)
- Formula: Lauric + Myristic + Ricinoleic
- Big, fluffy bubbles

### Creamy Lather (16-48)
- Formula: Palmitic + Stearic + Ricinoleic
- Fine, stable, creamy bubbles

### Iodine Value (41-70)
- Unsaturation level
- Lower = longer shelf life

### INS Value (136-170)
- Hardness minus Iodine
- Overall bar quality indicator

## Example Recipes Programmed

### Beginner Recipe (Balanced)
- 35% Olive Oil
- 30% Coconut Oil
- 25% Palm Oil (or Shea Butter)
- 10% Castor Oil

### Conditioning Bar (Moisturizing)
- 40% Olive Oil
- 25% Coconut Oil
- 20% Shea Butter
- 10% Castor Oil
- 5% Sweet Almond Oil

### Hard Bar (Long-lasting)
- 30% Olive Oil
- 30% Coconut Oil
- 25% Palm Oil
- 10% Cocoa Butter
- 5% Castor Oil

### Gentle Bar (Sensitive Skin)
- 60% Olive Oil
- 20% Coconut Oil
- 10% Shea Butter
- 10% Castor Oil

## AI Integration

The AI has been trained with:
- Complete oils database knowledge
- SoapCalc methodology
- Recipe calculation best practices
- Safety protocols
- Property interpretation guidelines

### AI System Prompt Includes:
1. **Mandatory calculator use** for all recipes
2. **Safety-first approach** with warnings
3. **Property explanations** with typical ranges
4. **Recipe recommendations** for different soap types
5. **Formatting guidelines** for clear, readable output

## Safety Features

Every calculated recipe includes:
- ⚠️ "Always add lye to water" warning
- Safety equipment reminders (goggles, gloves)
- Ventilation requirements
- Measurement accuracy emphasis
- Curing time (4-6 weeks)
- Child/pet safety warnings

## Technical Details

### SAP Values
- Stored as NaOH and KOH values per gram
- Accurate to 4 decimal places
- Based on verified soapmaking references

### Fatty Acid Profiles
- Complete 8-acid breakdown for each oil
- Percentages used to calculate weighted averages
- Properties calculated from fatty acid composition

### Calculation Method
```javascript
1. Sum total oil weight
2. Calculate lye needed: oil_weight × SAP_value
3. Apply superfat discount: lye × (1 - superfat%)
4. Calculate water: Based on lye concentration or water:lye ratio
5. Calculate fatty acid profile: Weighted average by oil percentages
6. Calculate properties: Standard SoapCalc formulas
7. Format output with ranges and status indicators
```

## Future Enhancements

Potential additions:
- [ ] More oils (currently 19, can expand to 50+)
- [ ] Liquid soap (KOH) calculator workflows
- [ ] Fragrance calculator with usage rates
- [ ] Colorant suggestions
- [ ] Recipe saving/sharing
- [ ] Visual property charts
- [ ] Mobile-optimized calculator interface
- [ ] Batch size converter
- [ ] Cost calculator

## Usage Examples

### Simple Request
```
User: "Make me a basic soap recipe"
AI: Asks for batch size, confirms oils, calculates, shows recipe with properties
```

### Advanced Request
```
User: "I have olive, coconut, and shea butter. Make 800g with 7% superfat"
AI: Suggests percentages, calculates exact amounts, shows all properties
```

### Educational Request
```
User: "What does the cleansing number mean?"
AI: Explains it's NOT cleaning power, but water solubility based on lauric/myristic acids
```

## Testing Checklist

✅ Calculator loads properly
✅ Oils database accessible
✅ AI references calculator in recipes
✅ Properties calculated correctly
✅ Safety warnings included
✅ Formatting displays properly (markdown tables)
✅ Range indicators work (good/low/high)
✅ Multiple recipe types available

## Resources

This implementation is based on:
- SoapCalc.net methodology
- Modern Soapmaking property explanations
- Classic Bells soapmaking resources
- Verified SAP value tables
- Industry-standard calculation methods

---

**Last Updated**: 2025-12-10
**Version**: 1.0
**Author**: Claude Code (AI Engineering)
