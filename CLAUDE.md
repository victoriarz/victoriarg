# CLAUDE.md - AI Assistant Guide for Victoria Ruiz Griffith Portfolio

## Project Overview

This is a personal portfolio website for Victoria Ruiz Griffith, an AI Engineer. The site is hosted on GitHub Pages with a custom domain (victoriarg.com).

**Project Type**: Static portfolio website
**Tech Stack**: HTML5, CSS3
**Hosting**: GitHub Pages
**Domain**: victoriarg.com

## Repository Structure

```
victoriarg/
├── index.html          # Main landing page with hero section
├── style.css          # Global styles and component styling
├── CNAME              # Custom domain configuration for GitHub Pages
└── CLAUDE.md          # This file - AI assistant guidance
```

### File Descriptions

- **index.html**: The main entry point featuring a navigation bar and hero section introducing Victoria as an AI Engineer
- **style.css**: Organized CSS file with clearly commented sections for different components
- **CNAME**: Contains the custom domain `victoriarg.com` for GitHub Pages

## Code Style and Conventions

### HTML Conventions

1. **Semantic HTML**: Use semantic HTML5 elements appropriately
2. **Indentation**: 4 spaces per indentation level
3. **Meta tags**: Always include charset and viewport meta tags
4. **Accessibility**: Ensure proper heading hierarchy and semantic structure
5. **Structure**: Keep the document clean and well-organized with proper nesting

### CSS Conventions

1. **Organization**: CSS is organized into clearly commented sections:
   - Global Reset and Base Styles
   - Navigation Bar Styles
   - Component-specific styles

2. **Commenting**: Each major section has a clear header comment:
   ```css
   /* ------------------------------------------- */
   /* 1. Section Name */
   /* ------------------------------------------- */
   ```

3. **Inline Documentation**: Important properties have inline comments explaining their purpose
   ```css
   font-family: Arial, sans-serif; /* Sets a clean, standard font */
   ```

4. **Spacing**: Consistent spacing between selectors and properties
5. **Color Scheme** (Earthy Tones):
   - Primary brand color: `#d4a574` (Earthy orange-tan)
   - Hover state: `#c89563` (Darker orange)
   - Text color: `#4a3f35` (Warm brown)
   - Background: `#f5ebe0` (Beige) and `#faf8f5` (Off-white)
   - Accent green: `#8fa888` (Soft green for success indicators)

### Design Patterns

1. **Navigation**: Flexbox-based horizontal navigation with:
   - Name on the left (`h1`) with clickable link to home
   - Navigation links on the right (`ul li`)
   - Warm beige background with subtle border
   - Active state highlighting for current page

2. **CTA Buttons**: Standardized call-to-action button style with:
   - Inline-block display
   - Earthy orange-tan background color (#d4a574)
   - Smooth hover transitions with transform effect
   - Border radius for rounded corners
   - Subtle box shadows for depth

3. **Responsive Design**: Viewport meta tag included for mobile responsiveness

## Development Workflow

### Branch Strategy

- **Feature branches**: Use branches prefixed with `claude/` for AI-assisted development
- **Branch naming**: `claude/claude-md-<session-id>`
- **Main development**: Work should be developed on feature branches before merging

### Git Workflow

1. **Always develop on the designated branch** (never directly on main)
2. **Commit messages**: Use clear, descriptive commit messages in imperative mood:
   - ✅ "Update portfolio title and hero section content"
   - ✅ "Refactor global styles and navigation layout"
   - ✅ "Clean up and update global styles in style.css"

3. **Push strategy**: Always use `git push -u origin <branch-name>`
4. **Retry logic**: For network errors, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

### Making Changes

#### When Modifying Existing Files

1. **Always read the file first** before making changes
2. **Preserve existing structure and style** - maintain consistency with current conventions
3. **Keep inline comments** that explain important styling decisions
4. **Maintain CSS section organization** - don't break the commented section structure

#### When Adding New Features

1. **Add new CSS sections** with proper comment headers
2. **Follow the established color scheme** unless explicitly requested otherwise
3. **Ensure responsive design** considerations for mobile devices
4. **Update navigation links** if adding new pages

#### Common Tasks

**Adding a new page:**
1. Create the HTML file with the same structure as index.html
2. Link to it from the navigation in index.html
3. Add any page-specific styles to style.css in a new section
4. Ensure the nav links are updated across all pages

**Updating styles:**
1. Read style.css first to understand current organization
2. Add styles to the appropriate section
3. Include inline comments for complex or important properties
4. Test hover states and transitions

**Changing color scheme:**
1. Update the primary brand color (`#007bff`) and its hover state (`#0056b3`)
2. Check all uses of colors in style.css
3. Ensure sufficient contrast for accessibility

## Deployment

This site is deployed via **GitHub Pages**:
- Automatically deploys from the main branch
- Custom domain configured via CNAME file
- Changes to main branch trigger automatic deployment

## Project Context

### Current State (as of latest commits)

- Clean, minimal portfolio landing page
- Professional navigation bar with name and links
- Hero section introducing Victoria as an AI Engineer
- CTA button linking to future projects page
- Well-organized and documented CSS

### Planned Features (referenced in navigation)

Based on navigation links in index.html, these pages are planned but not yet created:
- `projects.html` - Portfolio projects showcase
- `about.html` - About Victoria page
- `contact.html` - Contact information/form

### Design Philosophy

- **Minimalist and professional**: Clean design focused on content
- **Well-documented code**: Extensive inline comments for maintainability
- **Organized structure**: Clear separation of concerns and logical organization
- **Standard technologies**: Vanilla HTML/CSS without frameworks for simplicity

## AI Assistant Guidelines

### DO:
- ✅ Read existing files before making modifications
- ✅ Maintain the established CSS commenting style
- ✅ Follow the existing color scheme unless asked to change it
- ✅ Keep code well-organized and documented
- ✅ Preserve the clean, minimal aesthetic
- ✅ Write clear, descriptive commit messages
- ✅ Test responsive behavior when making layout changes
- ✅ Use the TodoWrite tool for multi-step tasks

### DON'T:
- ❌ Remove or modify inline comments without good reason
- ❌ Add frameworks or libraries unless explicitly requested
- ❌ Make changes without reading the file first
- ❌ Break the CSS section organization
- ❌ Push directly to main branch
- ❌ Add unnecessary complexity to this simple portfolio site
- ❌ Over-engineer solutions - keep it simple

### When Uncertain:
- Ask the user for clarification before making significant structural changes
- Propose changes before implementing major design modifications
- Verify color scheme changes before applying them across the site

## Testing Checklist

When making changes, verify:
- [ ] HTML validates (proper nesting, closed tags, semantic structure)
- [ ] CSS formatting is consistent with existing style
- [ ] Navigation links work correctly
- [ ] Hover states function properly
- [ ] Comments are clear and helpful
- [ ] Code follows established conventions
- [ ] Changes are committed with clear messages

## Quick Reference

### Key Files by Purpose

| Purpose | File | Description |
|---------|------|-------------|
| Landing page | index.html | Main entry point with hero section |
| Projects showcase | projects.html | Portfolio projects with tech tags |
| About page | about.html | Bio, experience, education, skills |
| Contact page | contact.html | Contact info and social links |
| Styling | style.css | All CSS styles, organized by component |
| Domain config | CNAME | GitHub Pages custom domain |
| AI Guide | CLAUDE.md | Documentation for AI assistants |

### Color Palette (Earthy Tones)

| Color | Hex Code | Usage |
|-------|----------|-------|
| Warm Brown | #4a3f35 | Main text, headings |
| Dark Brown | #2c1f15 | Dark accents, button text |
| Beige | #f5ebe0 | Navigation background, cards |
| Light Beige | #e3d5ca | Borders, subtle dividers |
| Earthy Orange | #d4a574 | CTA buttons, primary accents, links |
| Hover Orange | #c89563 | Button/link hover states |
| Medium Brown | #6b5d52 | Secondary text, descriptions |
| Light Brown | #7a6a5a | Meta text, subtitles |
| Soft Green | #8fa888 | Certification checkmarks, success |
| Off-White | #faf8f5 | Body background |
| Pure White | #ffffff | Card backgrounds |

### Common Commands

```bash
# View current status
git status

# Create and switch to feature branch
git checkout -b claude/feature-name-session-id

# Add and commit changes
git add .
git commit -m "Descriptive message in imperative mood"

# Push to remote
git push -u origin claude/feature-name-session-id

# View recent commits
git log --oneline -10
```

---

**Last Updated**: 2025-12-08
**Maintained by**: AI assistants working on victoriarg portfolio
