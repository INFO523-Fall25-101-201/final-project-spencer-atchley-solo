[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21346825)
# Video Game Recommendation Engine
INFO 523 final project: a video game recommendation engine using collaborative filtering plus sales trend exploration on the vgsales dataset.

## What’s in this repo
- writeup.qmd: Non-technical narrative of project goals, approach, tools, datasets, and findings.
- notebooks/final.ipynb: Technical writeup with EDA and the collaborative filtering model built with Surprise.
- index.qmd: Landing page/abstract for the Quarto site summarizing the project.
- about.qmd: Course and author context.
- proposal.qmd: Original project plan with early data exploration and synthetic data sketching.
- presentation.qmd: Reveal.js slide deck for the final presentation (uses data/customtheming.scss).

## Datasets and code
- Data lives under `data/` (vgsales plus synthetic player/rating data).
- Supporting scripts are under `scripts/`; rendered site output is in `_site/`.
- Quarto configuration is in `_quarto.yml`; project file is `project-final.Rproj`.

## Usage
- View the non-technical writeup via `writeup.qmd`; open `notebooks/final.ipynb` for the full technical analysis.
- Quarto site pages are built from the `.qmd` files; `_site/` holds the rendered output if present.

## Disclosure
Derived from the original data viz course by Mine Çetinkaya-Rundel @ Duke University.
