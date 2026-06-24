![](assets/mlflow-banner.png)
# MLflow Tutorial
In this repository, we will be covering the ins and outs of **MLflow**. MLflow is an open source AI/ML platform developed by the company **Databricks** and has been popular in the AI/ML community for a long while now. In fact, I did a series of blog posts about MLflow back in 2020 [starting with this one](https://medium.com/data-science/mlflow-part-1-getting-started-with-mlflow-8b45bfbbb334), and while some things have been consistent with MLflow since that time, there have been a lot of additions and changes in the recent years. Specifically MLflow used to hone in only on **traditional ML**, but in more recent times, there has been more of a turn toward things to support **generative AI**. Throughout this tutorial, we will cover both sides of MLflow, both traditional ML and generative AI.

Something to be aware of: I mentioned that MLflow was originally created by Databricks, and to this day, MLflow is a major part of the Databricks platform. This does not mean, however, that MLflow is exclusive to Databricks. Because MLflow itself is open source, we can leverage it on any infrastructure we please, including our own local laptops / desktops. For the purposes of this tutorial, we will not focus on anything with Databricks in particular. That said, much of this tutorial should still largely translate well if you're working directly on Databricks, but please be aware that there are some things that the Databricks implementation of MLflow does a little differently than the "pure" open source version.

> [!NOTE]
> This repository is currently a work in progress. I am currently making updates to it throughout April-May 2026. I will remove this note once I've soldified all the content!



## Accompanying Livestreams
This tutorial is intended to accompany some livestreams I conducted across April and May 2026. You can catch each of the respective livestreams on YouTube at the links below:

- **An Overview of MLflow for Traditional ML** ([Link](https://www.youtube.com/live/2OVU4DZLAvY?si=OiduwVFB46TR5nwL)): This stream introduces MLflow by covering the architecture and terms associated to MLflow in general and also covers all content associated to traditional ML.


## MLflow Architecture
<p align="center">
	<img src="/assets/mlflow-arch.png" alt="MLflow architecture" />
	<br />
	<em><a href="https://mlflow.org/docs/latest/self-hosting/architecture/overview/">Image taken from MLflow's website</a></em>
</p>

When it comes to using MLflow, the architecture diagram above generally references how MLflow is deployed. To be clear, this diagram is not the *only* way MLflow can be deployed, as providers such as Amazon Web Services (AWS) also offer their own flavor of a managed MLflow. For our purposes, we will be working with the pattern depicted by the left side of the image, but in terms of how we interact with MLflow, that is pretty standard regardless of how MLflow is deployed.

Conceptually speaking, MLflow manifests with the following backend mechanisms:

- **Tracking server**: The tracking server is the "entrypoint" for interacting with the other elements we'll discuss down below, and it leverages FastAPI under the hood. In a production setting (as dictated by the right side of the arch image), the tracking server is generally a literal, dedicated server. For our purposes, we will emulate the functioanlity of the tracking server on our local laptops / desktops.
- **Backend Store**: This is generally a database or something that emulates a database and stores all the metadata regarding experiments, runs, traces, and more. (Don't worry if you're not familiar with those terms I just mentioned there; we'll cover those further on down.) For our local use, we will not setup a database; however when we start up the tracking server for the first time, you will see that MLflow will automatically create a file called `mlflow.db` that holds this information as the backend store.
- **Artifact Store**: The artifact store is a backend object-based store intended to hold large artifacts, such as serialized models. Because it is an object-based store, a common backend used is AWS S3. For our purposes, we will not set up anything specifically to serve as the artifact store, but when we initiate the tracking server for the first time, MLflow will create a local directory called `mlruns/` that will use our local filesystem to store any artifacts.



## MLflow Terms / Concepts
In this section, we'll cover a variety of terms and concepts associated to various elements of MLflow. Please be aware, for better or worse, that I'm going to give "David's take" on these terms. The reason I'm giving my take is because some of these terms don't seem to align with the intention of the concept behind the term, so my take is to try to explain why these terms are named the way they are.

### Experiment
An experiment is essentially a logical container for a set of related runs. Now, you might be wondering, "Why is it called an experiment then?" The term "experiment" in general can be defined relatively loosely, but in the early days of MLflow, applying this term manifested more closely to the "pure" definition of an experiment. Consider hyperparameter tuning specifically. In that case, it makes complete sense to label an MLflow an experiment as we are naturally experimenting with a bunch of different hyperparameter combinations to find the ideal combo to produce the best results from a trained model. These days, MLflow still uses the "experiment" terminology, but is more encompassing than just a simple hyperparameter tuning experiment.

### Run
In MLflow, a **run** represents one specific execution of a set of code, in which we can also capture things like metrics, logs, and artifacts. If you've been following along with my recent livestreams, you might think this sounds a lot like [OpenTelemetry](https://github.com/dkhundley/otel-tutorial), and while the two are not equal, I personally like to think about runs like they are MLflow's version of OpenTelemetry. Runs are *always* associated to an experiment in a "one-to-many" fashion: one experiment may contain many runs.

### Model Signature
Because MLflow can directly serve out models for inference through the tracking server, it is important that the tracking server has an idea of what it should be expecting in terms of what it receives (inputs) and what it sends back to the calling client (outputs). We do this by applying a model signature, which specifically dictates the data elements / data element types for inputs and outputs. In addition, MLflow also allows users to register specific examples of input / output data to reinforce the model signature.



## Traditional ML
When it comes to traditional ML, MLflow has been around for almost a decade and has refined their process to make everything as seamless as possible for a data science practitioner. In this section, we'll touch on the ways in which MLflow supports traditional ML. If you want to see a tangible demonstration of MLflow with code, please see the notebook at `notebooks/traditional_ml.ipynb`.

### Hyperparameter Tuning Experiment Tracking
As we touched on with the definition of "experiment" above, MLflow supports experimentation by keeping track of a data scientist's hyperparameter tuning experiments by saving the parameters tested alongside the validation results. This way, a data scientist can go back to view the results to analyze and understand which hyperparamater combinations may be more ideal.

The other great thing about MLflow is that it supports **autologging**. Where it used to be that a user had to be more explicit in writing code to log things like the parameter combinations or validation metrics, MLflow supports a wide range of popular ML frameworks and will automatically log a bunch of different things associated to that respective library. Supported libraries include Scikit-Learn, XGBoost, PyTorch, and more. If a user wants to be a little more particular or is working with an outlier framework that doesn't support autologging, manual logging is always still available.

### Model Training / Registration
MLflow supports model training and registration into their own built in registry, but they do so in a very interesting way. Namely, it registers the model into an artifact package that can be seamlessly served out directly from the MLflow tracking server. Essentially, MLflow is very intentional about capturing your dependencies to ensure that your model works at runtime with all the right package installs.

The MLflow artifact package generally consists of the following files:

- `MLModel`: This file represents the core manifest of the artifact package. It is essentially a YAML file that contains all the information about how to properly run the trained model, including things like the model signature, which trained / serialized model files should be used, and more.
- `requirements.txt`: As most Python developers are familiar with, we save a list of our version pinned dependencies with a requirements.txt file. This file serves exactly that same purpose.
- `python_env.yaml`: This dictates how a Python virtual environment should be created, including with which version of Python.
- `conda.yaml`: This file dictates how to set up a Python virtual environment with `conda-forge`. For context, `conda-forge` spun off from the company Anaconda and is now its own open source library for installing dependencies to a Conda environment. Part of the reason this is important is because the company Anaconda semi-recently changed some of its licensing requirements for larger organizations, which requires larger organizations to pay that licensing fee. By leveraging `conda-forge`, we need not concern ourselves with the Anaconda licensing issue.
- **Serialized model artifacts**: These files are the traditional trained model files that data scientists are familiar working with, such as `.pkl` or `.pth` files.

### Model Serving
Once a model has been trained, it is possible to serve the model for inference directly from the tracking server. It makes use of the whole artifact package we covered in the section above to stand up an endpoint that can be called for inference use as a real-time API.

In addition to serving a single model endpoint, it is possible to customize the endpoint using **pyfunc**. pyfunc is essentially a universal interface for interacting with MLflow models. More specifically, this is how one may apply pre- or post-processing logic, or a user can even serve multiple models behind the same endpoint this way.



## GenAI
Because MLflow as a whole predates the whole generative AI (GenAI) revolution, there naturally did not previously exist features that support GenAI. In recent years, MLflow has been enhanced to 