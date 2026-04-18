![](assets/mlflow-banner.png)
# MLflow Tutorial
In this repository, we will be covering the ins and outs of **MLflow**. MLflow is an open source AI/ML platform developed by the company **Databricks** and has been popular in the AI/ML community for a long while now. In fact, I did a series of blog posts about MLflow back in 2020 [starting with this one](https://medium.com/data-science/mlflow-part-1-getting-started-with-mlflow-8b45bfbbb334), and while some things have been consistent with MLflow since that time, there have been a lot of additions and changes in the recent years. Specifically MLflow used to hone in only on **traditional ML**, but in more recent times, there has been more of a turn toward things to support **generative AI**. Throughout this tutorial, we will cover both sides of MLflow, both traditional ML and generative AI.

Something to be aware of: I mentioned that MLflow was originally created by Databricks, and to this day, MLflow is a major part of the Databricks platform. This does not mean, however, that MLflow is exclusive to Databricks. Because MLflow itself is open source, we can leverage it on any infrastructure we please, including our own local laptops / desktops. For the purposes of this tutorial, we will not focus on anything with Databricks in particular. That said, much of this tutorial should still largely translate well if you're working directly on Databricks, but please be aware that there are some things that the Databricks implementation of MLflow does a little differently than the "pure" open source version.




> [!NOTE]
> This repository is currently a work in progress.

## Traditional ML