RAG-implementation
==============================

An implementation of a Retrieval-Augment-Generation (RAG) system where an LLM gives answers to a question based from a given database.

**Live Website**: https://lena-rag.vercel.app/

**Techstack**:
- LLM: Cohere Command-R
- Embedding Model: Cohere Embeddings
- LangChain
- LangGraph
- FastAPI
- NextJS

**Database/Corpus**
The corpus covers 2 topics.
Topic 1: Lang Yang Lamu Symbiosis
- Study of the symbiosis of three fictitious creatures
    - Lang. Mythic wolf. Hunts down Yang. Urine nutritionally enhances Lamu Plant.
    - Yang. Mythic sheep. Herbivore. When eating enhanced Lamu plant, its feces becomes great fertilizer
    - Lamu. Miracle bloom. When eaten by Yangs, they gain a poisonous property lethal to Langs.

Topic 2: Side effects of time travelling
- Temporal disorientation/displacement
    - Symptoms include “chrono-cultural shock”, stress and anxiety, and identity crises
    - Coping mechanisms include journaling, meditation, etc.
- Dr. Alexander Hayes
    - Pioneered time travel together with his team.
    - First to experience the side effects of time travelling

**Pipeline**
![image](https://github.com/Tiny-Banana/RAG-implementation/assets/71121771/d9fe6daa-92b6-4105-a43f-9059ce51e914)

**Running the application**
```console
npm run dev
uvicorn api.index:app --reload
```
Once running, go to address: localhost:3000


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
