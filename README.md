# QS World University Rankings 2025

This repository contains the implementation and documentation of a project developed for the Visual Analytics course (2025/2026).

The project presents an interactive visual analytics system for exploring the QS World University Rankings 2025 dataset, with the goal of enabling profile-based comparison of universities beyond ordinal ranking positions.

## Repository Structure

```text
├── app.py                           # Dash application (interactive visual analytics system)
├── qs_2025_pca.csv                  # Preprocessed QS 2025 dataset with PCA components
├── va_2526_shokayeva_tskhe.pdf      # Final project report 
├── va_2526_ppt_shokayeva_tskhe.pdf  # PowerPoint presentation
├── va_2526_data_preprocessing.ipynb # Data preprocessing script
└── README.md                        # Project description and instructions
```

## Dataset

The dataset is based on the QS World University Rankings 2025, obtained from a publicly available Kaggle source (https://www.kaggle.com/datasets/darrylljk/worlds-best-universities-qs-rankings-2025). It includes information on 1,503 universities worldwide and multiple quantitative performance indicators used for ranking computation.

## How to Run the Application
Install dependencies: ```pip install dash plotly pandas numpy```

Run the application: ```python app.py```

Open the browser at: ```http://127.0.0.1:8050```

## Authors
Yelizaveta Tskhe & Dariga Shokayeva
