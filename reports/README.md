# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
  are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1

> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

93

### Question 2

> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

fmfsa: s250106, obola: s155827

### Question 3

> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

The members of this group are PhD students at DTU specializing in metamodeling of scientific simulators. We utilized the
third-party framework PyG, which builds upon the PyTorch library to simplify the design and training of Graph Neural
Networks (GNNs). This framework provides a streamlined framework for the development of metamodels for various
simulators by providing tools to implement cutting-edge GNN layer types within the PyTorch ecosystem. PyG allowed us to
efficiently test our research against state-of-the-art methodologies, enhancing both the robustness and scalability of
our metamodeling efforts. Our dataset consists of a list of PyG objects (i.e., PyG Graphs) that plays nicely together
with ML model pipeline. The PyG framework is essential for doing research in a fast changing environment as machine
learning as it reduces the time spent on coding significantly.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used Conda to manage our dependencies in the project, creating a dedicated Conda environment to ensure consistency
and reproducibility. The list of dependencies was auto-generated using pipreqs, which ensures that only the packages
explicitly used in the code are included in the requirements.txt file. A new team member can recreate the exact
environment by following these steps:

1. Run `conda env create -f environment.yml` to create the environment
2. Activate the environment using `conda activate ml_ops`
3. Install the project dependencies locally with `pip install -e .`

These steps ensure that the environment mirrors the original setup, allowing seamless collaboration and project
execution.

On a local computer it is great to have conda to control several virtual environments on the same machine. However, the
dependencies can also be installed using only pip and the `requirements.txt` file (this is for example how the VMs on
docker are configured). The file `requirements_dev.txt` contains the packages that are needed for further development of
the repository (in our case the two files are identical).

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

The cookiecutter template was used as the baseline structure for our project. We stayed very close to the template but
deviated slightly in the src folder where the model.py script is now taking encoding methods, GNN layers, and decoding
methods from three new scripts with PyTorch classes. The idea is that these scripts can house customized modules
developed during the PhD to improve the prediction error of the metamodels. We removed the `visualize.py` script since
full integration with W&B provides sufficient visualizations. Additionally, the `api.py` script was deleted as the cloud
configuration files were sufficient for setting up Google Cloud.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used typing throughout the code to ensure better readability, have better documentation, and make debugging easier.
Typing is a great tool to ensure that the reader of a function can see what type of argument the function expects (e.g.,
an argument `x` in a `torch.module` class is expected to be a `torch.tensor` which could then be written as
`x: torch.tensor` in the definition of the function).

We also implemented ruff as a pre_commit action to ensure that the code was always formatted properly. This helped us
removing whitespace after a finished line of code and generally identifying undesirable formatting patterns. Using Ruff
as a pre-commit action to ensure that any code merged with the main branch adhered to proper formatting and quality
standards.

At last, we tried to document our code with inline comments using `#` and with doc-strings using `'''`. This is good
practice and also supports efficient collaboration when building on each others code.

These measures are important in larger projects as it makes it easier for team members to maintain each others code and
prevent bugs.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 14 tests. We are testing the `data.py`, the `model.py` and the `train.py`.

* The `test_data.py` checks whether the data is available and has been formatted correctly (e.g., the existence of the
  data and the shape of x features and edge features).
* The `test_model.py` checks whether the shapes of the encoder, gnn_layer, and decoder works and also puts these three
  elements together into a transformer.
* The `train_model.py` checks is the `TrainModel` instance can be initialized with a given example of a configuration
  file.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The test coverage is around 42% of the code which is in the lower end (and quite far from 100%). However, the most
critical parts are covered (including testing the GCN and all elements used to build it, and the `TrainModel` class
which is by far the largest and most important class in the code). In addition, even if code coverage were close to
100%, it would not guarantee the absence of errors. Comprehensive tests can ensure critical functionality but cannot
account for all edge cases or potential unforeseen issues, making it essential to complement testing with other
validation practices (as code review in Github).

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

As a group, we consistently used branches and pull requests (PRs) in our workflow. All new work was developed on
separate branches, and several automations and requirements had to be satisfied before merging into the main branch.
This approach ensured proper version control, facilitated peer review, and minimized the risk of introducing bugs into
the main codebase.

In order to merge code, all tests had to pass on both ubuntu and macos, the pre-commit had to be passed, and another
member of the group needed to review all code changes and approve them. When approved the student that had submitted the
PR would then merge into main.

If there was conflicts the submitter of the PR would have to resolve these before a code review. Other student /
colleagues would then be there to support if it was needed.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

While we have not employed DVC specifically for data version control, we do maintain versioned models via Docker images in a cloud-based Artifact Registry. Each time we update our models, a new Docker image is pushed, effectively capturing and preserving the state of the code, dependencies, and any embedded data artifacts used by the model at that point in time. This approach ensures that we can easily roll back to previous versions of our models if needed and keeps our production deployments consistent.

Still, a tool like DVC could be beneficial in scenarios where large datasets themselves need explicit versioning and collaborative workflows. For example, if you need to track incremental changes to a massive dataset in a structured manner or experiment with multiple data subsets for training and evaluation, DVC would allow you to manage data revisions alongside code changes in Git. This can be especially helpful when working in larger teams or when reproducibility is a high priority.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We set up continuous integration with several GitHub automations. This included configuring the environment on GitHub
and running our tests on both ubuntu and macOS. Additionally, we implemented a pre-commit automation that ran Ruff to
check code quality and formatted the code automatically, ensuring consistency and reducing potential errors. We also
setup a codecheck that runs ```ruff check .``` and ```ruff format .``` but in the end do not use. Similar to our
pre_commit based on ```stefanzweifel``` git repo. The reason for this is because we also setup pre-commit CI that runs
the same commands and also we wanted to have a more consistent setup. The pre-commit CI is also faster than the
codecheck CI. For example our unittests workflow can be checked
on [tests.yaml](/Users/fmfsa/Repositories/ml_metamodels/.github/workflows/tests.yaml). Additionally, our pre-commit CI
can be checked on [PRE-COMMIT CI](https://results.pre-commit.ci/run/github/914763986/1737407950.3heoTpreQTaBC9l-VAaUXw).
For the unit tests workflow upon each push or pull request to the main branch, it checks out the code, sets up Python (
3.11), installs dependencies, and runs pytest with coverage. The workflow uses a matrix strategy to test on both Ubuntu
and macOS. For pushes to the main branch, it also authenticates with GCP and submits the build to Cloud Build for
continuous integration.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

The core of our ML pipeline revolves around conducting scientific experiments on the metamodeling of simulators. To
streamline this process, we extensively used Hydra for configuration management.

Hydra configuration setup consisted of a base configuration file (`hydra_config.yaml`) that linked to five subfolders
containing configuration files for `data`, `inference`, `model`, `train`, and `wandb`. For example, certain model types requiring
unique parameters could have their settings specified in dedicated .yaml files. Furthermore, the Hydra setup was
integrated with W&Bsweeps, enabling seamless formatting of configuration files into W&Bsweep dictionaries when
parameters were provided as lists.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure reproducibility of experiments, we relied on configuration files and set a random seed within the
configuration. By using the same configuration setup, any experiment could be replicated with identical results.

W&B further facilitated reproducibility by storing configuration details in the `hydra_config.yaml`, which was linked to
the Hydra-generated configuration file. Additionally, model weights were stored as artifacts in W&B and could be
retrieved using the evaluate.py script. By specifying the path to a particular run_id in the inference configuration
file, prediction results could be reproduced without retraining the model. This setup ensured that all information
pertaining to an experiment was logged and accessible, enabling easy replication of results.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

One of the cornerstones of this repository is the integration of Hydra and W&B. As shown in the first image, the W&B interface allows for easy inspection of training curves and comparison of different model runs:

![Wandb runs](figures/wandb_runs.png)

The Wandb interface allows the users to investigate the specific configurations of the different runs and the model weights are stored as an artifact in Wandb. The setup also allows for saving the data used to train the specific model as an artifact. This is as default set to false as the datasets can be quite large and Wandb is not build to store very large files.

The second image demonstrates the successful setup of W&B sweeps, enabling efficient hyperparameter searches:

![Wandb sweep](figures/wandb_sweeps.png)

This tool helps identify parameters with the most significant impact on validation error, ultimately optimizing the model for minimizing loss. The Wandb sweep interface is a amazing tool for researchers that want to test different models and easily evaulate the effect on the validation loss when doing smaller tweeks.

In general, Wandb is an amazing tool to manage experimentation with Machine Learning models in a strutured way and is a great alternative to store model weights and configurations of model runs on HDDs either on the local machine, on a local server or on a Google Cloud Bucket.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker was used to create containerized applications for this project. We developed two Docker files, both of which can
be built locally or on Google Cloud.

The training image can be built locally using the following command:

```docker build -f dockerfiles/train.dockerfile . -t train:latest```

This image relies on locally available configuration files, which are copied into the container.

The evaluation image can be built with the command:

```docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest```

This image also depends on configuration files and a specific run_id from W&B, which is used to download model weights
for evaluation.

Currently, the setup builds an image for a specific configuration, a limitation that could be addressed by providing
configuration files as input to the container when running the image.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging methods varied among group members:

* Oskar used VS Code and its built-in debugging tools.
* Francisco used PyCharm and its integrated debugging features.

These tools facilitated efficient identification and resolution of issues during experimentation. While we did not
explicitly profile the code, we believe the setup is efficient, though there is always room for improvement.
We also made use of breakpoint whenever we wanted to debug the code. Breakpoints allow us to pause the code at a
specific line and inspect the variables at that point in time. This was very useful when we wanted to understand the
output of a specific function or to understand the flow of the code.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used:

- IAM to manage access to our project. This service is used to control who/which service accounts can do what in the
  project.
- Artifact Registry to store our docker images. This service is used to store and manage docker images.
- Cloud Build to build our docker images. This service is used to build and push docker images.
- Compute Engine to run our docker images. This service is used to run virtual machines in the cloud.
- Cloud Storage to store our data. This service is used to store and manage data in the cloud.
- Vertex AI to train our models. This service is used to train machine learning models in the cloud.
- Secret Manager to store our secrets. This service is used to store and manage secrets in the cloud like WANDB_API_KEY.
- Cloud Run to showcase that we can setup a simple API, but no usecase for use so we did not extend this.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used Compute engine to run our docker images. We used instances with the following hardware: n1-standard-1.
Here are the key specifications for the n1-standard-1 machine type in Google Cloud:
- vCPUs: 1
- Memory (GB): 3.75
- Max number of Persistent Disks: 128
- Max total Persistent Disk size (TiB): 257
- Local SSD: Yes
- Default egress bandwidth (Gbps): Up to 2
- Tier_1 egress bandwidth (Gbps): N/A

Using this we would expect long training times, but for resource management we decided that this was good enough for our
project setup during this course. Additionally, the intention is to migrate to a local server in the future.
We created a VM instance called playground1 on Zone: europe-west1-b and then can connect to it via ssh.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![General cloud storage](figures/cloud_storage1.png)
![Raw and processed data](figures/cloud_storage2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Artifact Registry](figures/artifact_registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Cloud Build](figures/cloud_build.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model in the cloud using Vertex AI. We did this by creating a custom job and then selecting the
docker image we wanted to run that is on the artifact registry. We then selected the machine type and the number of
nodes we wanted to use. The reason we choose Vertex AI was because it is a managed service that makes it easy to train
machine learning models in the cloud. Additionally we inject WANDB_API_KEY on RUNTIME from GCP Secret Manager to log the
training process to Weights & Biases and to be able to do sweeps.
We also managed to do all of this using Compute Engine by running our docker images on a virtual machine, playground1.
Both ways of training are described on the README.md file.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We managed to write a Simple API and managed to deploy it on Cloud Run and used FastAPI to do this.
![Fast API](figures/fast_api.png)

We did not extend the API to expose our model and be able to do inference since this adds nothing to the intent of this
project. The goal of the repository is to support us in doing research and build custom GNNs for predicting the outcome
of scientific simulators.

Oh look, number 42, must mean something...
![42.png](figures/42.png)

As it can be seen from the above screenshot, the API was successfully deployed on the Google Cloud which shows that we
could have added model inference to it, if it had made sense for our purpose.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

As explained on Question 23, we did indeed manage to expose a simple API on the Cloud, also locally, but did not develop a
more complex API that would expose our model or be able to do model inference. The reason is that the use case of this
project is for experimentation on metamodelling of scientific simulators and for that reason, there's no point in
deploying the models. The API was deployed on Cloud Run and could be invoked by calling the URL of the API. The API
itself is a simple API that returns a JSON response with a message "Hello, World!" or an integer that is requested,
similar to the API developed during the class. If we would have extended the API to expose the model, we would have
needed to add a POST method that would take in the input data and return the prediction of the model, for example.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

Based on answers 23 and 24, there was no point in performing load test on the API. A load test would have been done by
using a tool like Locust to simulate a large number of users accessing the API at the same time. The goal of the load
test would be to see how the API behaves under heavy load and to identify potential bottlenecks. The results of the load
test would show how many requests the API can handle before it crashes and how the response time changes.
Since our API is a simple API that returns a JSON response, we did not perform this step.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Based on answers 23 and 24, there was no point in performing monitoring on the deployed model. Monitoring would have
been done by using a tool like Prometheus to collect metrics from the API and Grafana to visualize the metrics. The goal
of monitoring would be to track the performance of the API over time and to identify potential issues before they become
critical. The metrics that would be collected could include the response time of the API, the number of requests it
receives, and the error rate. Additionally, data drift detection could be implemented to monitor the performance of the
model over time and to identify when the model needs to be retrained.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

In total our total cost was $4.14. ![Cost breakdown](figures/cost.png). The most expensive was Compute because storage
is cheap, so Artifact and Cloud Storage will most of the times be reduced, and then we mostly used Compute Engine since
we only started using VertexAI by the end of the project. It's very good in a production environment, but for
development, it can be expensive and over-engineered. If we had used a machine with a GPU, the cost would have been
higher. The cost of the project was very low, but the cost of the cloud can be very high if not managed properly.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

No.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

In the figure below the overall architecture of the repository is outlined:
![Repository overview](figures/ml_ops_overview.png)

As it can be seen on the figure, the control and integrity of the repository is managed using github actions and code
reviews. The code can be run both to train a model and to evaulate a model. This is achieved using docker images and
containers that can be build and run in either Google Cloud or on the local desktop. The training image is linked to
wandb where all experiements are logged. The evaulate script is also linked to wandb where a run_id must be provided in
the config file and then the evaulate image can be build and executed.
The image is built and pushed to the Artifact Registry and then run on a Compute Engine or Vertex AI. The data is stored
in the Cloud Storage and the secrets are stored in the Secret Manager. The API is deployed on Cloud Run, even though it's
only a simple API that is not exposing anything related to the model since the purpose of the repository does not require
this. Github Actions is used to ensure that the code is always in a working state and that the code is always formatted
properly.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

While developing the project, we overall ran into two larger struggles:

1. Setting up wandb sweep
2. Setting up Google Cloud

When setting up wandb, there was issues in how to provide the API key to the docker containers in a safe way. In the beginning, we built the docker image and during this we provided the API key as an environment variable by copying the `.env` file. We ensured that the `.env` file was in the `.gitignore` and didn't push the docker image to the docker remote repositories. This had the limitation that users would need a `.env` file with the `WANDB_API_KEY` to build the image. We ended up being able to build the docker image without providing the `WANDB_API_KEY` which allowed us to make a VM without containing secret keys. When running the container, the user then needed to provide the `WANDB_API_KEY` for the image to run.

When setting up the project work in the cloud we experienced several issues. It was difficult to succeed with a connection directly on runtime to wandb via a `WANDB_API_KEY`. However, we solved that by providing the API key through the secret manager and inject it in VertexAI. Then when running the
docker image that exists on the Artifact Registry it would be provided with the `WANDB_API_KEY`.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Oskar s155827 was in charge of setting up the initial cookie cutter project, developing the project idea and setting up
wandb sweep.

Francisco s250106 was responsible for developing the docker images for training our models in the cloud.

Both members contributed with Typer, Hydra, logging, and testing. Both member were responsible for the maintenance of
the github repository as well as github Actions. Both members contributed to the code, the README.md file, the report
and the presentation.
