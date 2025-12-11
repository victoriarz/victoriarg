# Recipe Saving Feature - User Guide

## Overview
The Saponify AI now includes a comprehensive recipe saving system that allows users to save, organize, and reload their favorite soap recipes.

## Features

### 💾 Save Recipes
- Save any calculated recipe with a custom name
- Add optional notes (fragrance, colorants, techniques)
- Store up to 50 recipes in browser localStorage
- Recipes persist across sessions

### 📚 Recipe Library
- View all saved recipes in an organized library
- Search recipes by name or notes
- Sort by date (newest first)
- See key details at a glance (batch size, oils, superfat)

### 📂 Load Recipes
- Load any saved recipe back into the chat
- Recipe appears with all original calculations
- Can copy, print, or scale loaded recipes
- Preserves all soap properties and measurements

### 🗑️ Delete Recipes
- Remove recipes you no longer need
- Confirmation dialog prevents accidents
- Free up space for new recipes

## How to Use

### Saving a Recipe

1. **Calculate a Recipe**
   - Use the AI chat to calculate any soap recipe
   - Recipe will display with properties and measurements

2. **Click "💾 Save Recipe"**
   - Button appears below each calculated recipe
   - Opens the save dialog

3. **Enter Recipe Details**
   - **Name** (required): Give your recipe a memorable name
     - Example: "Lavender Dream", "Summer Citrus Soap"
   - **Notes** (optional): Add details about the recipe
     - Example: "Used 30ml lavender EO, swirled with purple mica"

4. **Save**
   - Click "Save Recipe" button
   - Recipe is stored in your browser
   - Success message confirms save

### Viewing Saved Recipes

1. **Click "📚 My Recipes"**
   - Button in the chat header (next to "Start Over")
   - Opens the recipe library

2. **Browse Your Recipes**
   - Recipes sorted by date (newest first)
   - Each card shows:
     - Recipe name
     - Save date
     - Batch size
     - Number of oils
     - Superfat percentage
     - Top oils used
     - Your notes

3. **Search Recipes**
   - Use the search box at the top
   - Searches recipe names and notes
   - Results update as you type

### Loading a Recipe

1. **Find Your Recipe**
   - Open the recipe library
   - Browse or search for the recipe

2. **Click "📂 Load Recipe"**
   - Recipe appears in the chat
   - Shows original name and save date
   - All calculations preserved

3. **Use the Recipe**
   - Copy to clipboard
   - Print
   - Scale up/down
   - Modify and save as new recipe

### Deleting a Recipe

1. **Find the Recipe**
   - Open recipe library
   - Locate the recipe to delete

2. **Click "🗑️ Delete"**
   - Confirmation dialog appears
   - Prevents accidental deletion

3. **Confirm**
   - Recipe is permanently removed
   - Cannot be undone

## Storage Information

### Capacity
- **Maximum**: 50 recipes
- **Warning**: Badge turns yellow at 80% capacity (40+ recipes)
- **Full**: Must delete recipes before saving new ones

### Storage Location
- Recipes stored in browser localStorage
- Data persists across sessions
- Clearing browser data will delete recipes
- Not synced across devices/browsers

### Data Size
- Each recipe: ~2-5 KB
- 50 recipes: ~100-250 KB total
- Well within localStorage limits (5-10 MB)

## Tips & Best Practices

### Naming Conventions
✅ **Good Names:**
- "Lavender Oatmeal - Gentle"
- "Coffee Scrub Bar"
- "Baby's First Soap (Castile)"
- "Winter Peppermint 2025"

❌ **Poor Names:**
- "Recipe 1"
- "Test"
- "asdf"
- "Untitled"

### Writing Useful Notes
Include:
- **Fragrance**: "30ml lavender EO, 10ml rosemary"
- **Colorants**: "Spirulina for green, mica swirl"
- **Additives**: "1 tbsp oatmeal, 1 tsp honey"
- **Results**: "Trace in 5 min, cured beautifully"
- **Date Made**: "Batch made Jan 2025"

### Organization Strategies

**By Type:**
- "FACIAL - Gentle Castile"
- "BODY - Coffee Exfoliating"
- "GIFT - Holiday Peppermint"

**By Season:**
- "SUMMER - Citrus Fresh"
- "WINTER - Cozy Cinnamon"

**By Batch:**
- "Batch #12 - Perfect Recipe"
- "Experimental - Hemp Oil Test"

## Keyboard Shortcuts

- **Escape**: Close any open modal
- **Ctrl/Cmd + K**: Start over (clears chat)
- **Enter**: Send message in chat

## Troubleshooting

### "Maximum 50 recipes reached"
**Solution**: Delete old or test recipes you no longer need

### Recipe not saving
**Possible causes:**
- Browser in private/incognito mode (localStorage disabled)
- Browser storage full (unlikely)
- Browser doesn't support localStorage (very old browser)

**Solution**: Use regular browsing mode, clear browser cache

### Lost all recipes
**Possible causes:**
- Cleared browser data/cookies
- Different browser or device
- Private browsing mode

**Prevention**:
- Export recipes (feature coming soon)
- Screenshot important recipes
- Keep written backup of favorites

### Recipe loads but can't scale
**Issue**: This is normal - scaling only works on newly calculated recipes

**Solution**: Loaded recipes display-only; ask AI to recalculate for scaling

## Privacy & Security

### What's Stored
- Recipe name
- Recipe calculations (oils, lye, water, properties)
- Your notes
- Save/update timestamps

### What's NOT Stored
- Personal information
- Chat conversations
- AI interactions
- Your IP address

### Data Access
- Only you can see your recipes
- Stored locally in your browser
- Not sent to any server
- Not shared with anyone

## Technical Details

### Browser Compatibility
- **Chrome/Edge**: Full support ✅
- **Firefox**: Full support ✅
- **Safari**: Full support ✅
- **Mobile Browsers**: Full support ✅
- **IE 11**: Limited support (no localStorage)

### Storage Format
- JSON format in localStorage
- Key: `saponifyai_saved_recipes`
- Each recipe includes full calculation results
- Timestamps in UTC milliseconds

### Export/Import
Coming soon:
- Export all recipes as JSON
- Import recipes from file
- Share recipes via URL

## Future Enhancements

### Planned Features
1. **Recipe Export** - Download recipes as PDF/JSON
2. **Recipe Sharing** - Share via URL or QR code
3. **Recipe Tags** - Categorize recipes (facial, body, gift)
4. **Recipe Variants** - Save variations of same base recipe
5. **Batch Tracking** - Log when you made each recipe
6. **Rating System** - Rate your recipes (1-5 stars)
7. **Photos** - Attach photos to recipes
8. **Cloud Sync** - Optional cloud backup (with account)

### Community Requests
Have a feature idea? Let us know!
- Email: v-ruiz@outlook.com
- Subject: "Saponify AI - Recipe Feature Request"

## FAQ

**Q: Will my recipes be deleted if I close the browser?**
A: No, recipes are saved permanently until you delete them or clear browser data.

**Q: Can I access my recipes on my phone and computer?**
A: Not yet - recipes are device-specific. Cloud sync coming soon.

**Q: What happens if I reach 50 recipes?**
A: You must delete old recipes before saving new ones.

**Q: Can I export my recipes?**
A: Export feature coming soon. For now, use "Print" or "Copy" buttons.

**Q: Are my recipes private?**
A: Yes, stored locally in your browser only.

**Q: Can I edit a saved recipe?**
A: Not yet - load it, modify in AI, save as new recipe. Edit feature coming soon.

**Q: What if I delete a recipe by accident?**
A: No undo - recipe is permanently deleted. Be careful!

## Support

### Getting Help
- **In-app**: Ask the AI assistant
- **Email**: v-ruiz@outlook.com
- **Website**: victoriarg.com

### Reporting Bugs
Please include:
- Browser and version
- Steps to reproduce
- What you expected vs. what happened
- Screenshots if possible

---

**Last Updated**: December 11, 2025
**Version**: 1.0.0
**Feature**: Recipe Saving System
