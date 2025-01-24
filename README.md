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
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── gnn_decoders.py
│   │   ├── gnn_layers.py
│   │   ├── models.py
│   │   ├── node_embeddings.py
│   │   └── train.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_data.py
│   └── test_train.py
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yml
├── fastapi_app.py
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
## Notes
To run training for example, start with:
```bash
pip install -r requirements.txt
pip install -e .
python src/ml_metamodels/train.py
```

Additionally, since the CLI is setup, you can run the following command to see the available options:
```bash
invoke --list
```

And since pyproject.toml is setup with scripts, you can ommit the invoke for some of the tasks, example:
```bash
train
```

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

# Docker
For the train dockerfile:
```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
docker run --name train --env WANDB_API_KEY=$(cat .env | grep WANDB_API_KEY | cut -d '=' -f2) train:latest
```

# Cloud
Provide the WANDB_API_KEY in the environment variables
```bash
gcloud secrets create wandb-api-key --replication-policy="automatic"
echo -n "your-wandb-api-key-value" | gcloud secrets versions add wandb-api-key --data-file=-

gcloud secrets add-iam-policy-binding wandb-api-key \
  --member="serviceAccount:test-service@ml-metamodels.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud builds submit --config=configs/cloudbuild.yaml .
```

You can pull the image that was pushed to the cloud and then run it by:
```bash
docker pull <region>-docker.pkg.dev/<project-id>/<registry-name>/<image-name>:<image-tag>
docker run --name train --env WANDB_API_KEY=$(cat .env | grep WANDB_API_KEY | cut -d '=' -f2) <region>-docker.pkg.dev/<project-id>/<registry-name>/<image-name>:<image-tag>
```

You can also create a VM instance and run the job on it:
```bash
gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=configs/config.yaml --command 'python src/ml_metamodels/train.py'
```

And to use VERTEX AI to run the job on the cloud, training the model and exposing everything to wandb:
```bash
gcloud builds submit --project=ml-metamodels --config=configs/vertex_ai_train.yaml
```

# Deployment
You can build the image:
```bash
docker build --platform=linux/amd64 \
  -f dockerfiles/fastapi_app.dockerfile \
  -t fastapi_app:latest \
  .
```

Then on Cloud Run, have a Service that uses that image from cloud registry.


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
