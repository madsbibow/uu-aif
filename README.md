# Algorithms in Finance

Course materials for the Algorithms in Finance course at Utrecht University.

## Repository Structure

```
uu-aif/
├── latex/              # Lecture slides (.tex)
├── py/                 # Python scripts for generating figures
├── ipynb/              # Jupyter notebooks for workshops
├── assets/             # Images, logos, backgrounds
├── output/             # Final PDFs for distribution (tracked in git)
└── uu-aif-beamer.sty   # Custom Beamer template
```

## Compiling Slides

### Option 1: Local compilation

```bash
pdflatex latex/course_intro.tex
```

Requires a LaTeX distribution with these packages:
```bash
sudo tlmgr install moloch tcolorbox fontawesome5 fira pgfopts
```

### Option 2: Hosted services (recommended)

> **Note:** The custom Beamer style (`uu-aif-beamer.sty`) uses recent LaTeX packages that may not be available in older or minimal local installations. If you encounter compilation errors, consider using a hosted service with an up-to-date LaTeX distribution:
>
> - [Overleaf](https://www.overleaf.com/)
> - [Prism](https://prism.run/) (Utrecht University)
>
> These services maintain current TeX Live installations and handle package dependencies automatically.

## Output Folder

PDFs are ignored by git except in `output/`. When a lecture is ready for distribution:

```bash
cp latex/course_intro.pdf output/
git add output/course_intro.pdf
```

## Using the Template

### Create a new lecture

1. Copy an existing `.tex` file from `latex/`
2. Update the title slide:
   ```latex
   \titleslide{Algorithms in Finance}{Lecture Title}{Date}
   ```
3. Use section dividers between major topics:
   ```latex
   \sectiondivider{yellow_green.png}{Section Title}
   ```
4. End with the closing slide:
   ```latex
   \closingslide{Questions?}{endpage_light_green.png}
   ```

### Available backgrounds for section dividers

- `yellow_green.png`
- `purple_pink.png`
- `blue_turq.png`
- `green_turq.png`
- `endpage_light_green.png` (for closing slides)

## Generating Figures

Python scripts in `py/` generate figures for lectures:

```bash
cd py
python generate_volatility_clustering.py
```

Output is saved to `assets/figures/`.

### Asset directories

- `assets/figures/` — Generated figures (from Python scripts)
- `assets/graphs/` — Graphics collected from external sources
