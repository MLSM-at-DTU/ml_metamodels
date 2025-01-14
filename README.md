# Project Description

## Overall Goal

The overall goal of this project is to build a repository for metamodelling scientific simulators using Graph Neural Networks (GNNs). The repository expects the data to be provided in `train.pickle`, `validation.pickle`, and `test.pickle` files, each containing a list of PyTorch Geometric (PyG) objects.

## Framework

The primary framework used in this repository is PyTorch Geometric, which provides robust tools for handling graph-based data and implementing GNN architectures. This choice enables efficient experimentation with state-of-the-art GNN models.

## Data

The initial dataset will be generated from a scientific simulator designed to estimate the flow on edges of a transportation network. The input data includes node features and edge features, representing various attributes of the transportation network. The flow on edges is estimated by solving the stochastic user equilibrium, which forms the basis of an edge regression task.

The repository expects static graph data, meaning the number of nodes, edges, and adjacency matrix structure remain constant. However, variations in node features, edge features, and corresponding edge-level regression targets should be supported.

## Models

We plan to experiment with state-of-the-art GNN architectures, starting with a baseline implementation of a simple transformer-based encoder, GNN layer, and decoder model. Initially, a GCN layer will be used as the GNN component, however the repository must allow for iteratively adding more advanced architectures.


# Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
## Notes
To save new dependencies, use the following command:
Either use pipreqs or pip freeze (not recommended):
```bash
pipreqs .
```

For format and linting, use the following commands:
```bash
ruff check .
ruff format .
```


# Tests
```bash
coverage run -m pytest tests/
coverage report
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
