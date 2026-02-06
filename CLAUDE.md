# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the production template for the "Algorithms in Finance" course at Utrecht University. It contains:
- A custom Beamer style package (`uu-aif-beamer.sty`) with UU branding
- Lecture slides (LaTeX)
- Python scripts for generating course figures
- Jupyter notebooks for student workshops

## Build Commands

### Compile LaTeX presentations
```bash
pdflatex latex/course_intro.tex
```

### Run Python figure generators
```bash
cd py && python generate_acf_plots.py
```
Output saved to `assets/figures/`.

### Install LaTeX packages (if missing)
```bash
sudo tlmgr install moloch tcolorbox fontawesome5 fira pgfopts
```

## Architecture

### Beamer Template (`uu-aif-beamer.sty`)
Custom Beamer theme built on Moloch with:
- UU brand colors (primary: `#FFCD00` yellow, `#1A1A2E` dark background)
- Auto-cycling section divider backgrounds (`yellow_green.png`, `purple_pink.png`, etc.)
- Custom commands: `\titleslide`, `\sectiondivider`, `\closingslide`, `\statemphasis`, `\circicon`
- Footline with/without logo (use `[nologo]` frame option for dense slides)
- Boxes: `formulabox`, `codebox`, `infobox`

### Asset Paths
The style uses `\uuaifassets` for relative paths. Helper commands:
- `\uulogo{filename}` → `assets/logos/filename`
- `\uubg{filename}` → `assets/backgrounds/filename`

### Python Scripts (`py/`)
Generate publication-quality figures for lectures:
- `generate_acf_plots.py` - Long memory/ACF visualization
- `generate_fat_tails.py` - Return distribution tails
- `generate_volatility_clustering.py` - Volatility clustering demo
- `generate_news_impact_curve_v4.py` - GJR-GARCH asymmetry
- `generate_var_chart.py` - Value at Risk illustration

Scripts output to `assets/figures/`.

## UU Brand Colors

| Name | Hex | Usage |
|------|-----|-------|
| Yellow | `#FFCD00` | Primary accent, bullets, frametitles |
| Dark BG | `#1A1A2E` | Title/section slide backgrounds |
| Cream | `#FAF9F6` | Content slide backgrounds |
| Purple | `#5B2182` | Secondary accent |
| Blue | `#5287C6` | Secondary accent |
| Green | `#24A793` | Secondary accent |
| Red | `#C00A35` | Warnings (use sparingly) |

## Common Patterns

### Add a new lecture
1. Copy `latex/course_intro.tex` as template
2. Update `\titleslide{Course}{Lecture Title}{Date}`
3. Use `\sectiondivider{background.png}{Section Title}` between major sections
4. End with `\closingslide{Questions?}{endpage_light_green.png}`

### Create emphasis slides (AI Now "$1T" style)
```latex
\statemphasis{Top text}{BIG STAT}{Bottom text}{softgreen}
\statemphasismed{Top text}{Medium content}{Bottom text}{softcoral}
```

### Icon badges for topic overviews
```latex
\circicon{uugreen}{\faChartLine}  % Creates circular badge with FontAwesome icon
```
