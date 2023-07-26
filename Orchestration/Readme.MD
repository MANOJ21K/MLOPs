Orchestrating ML Pipeline with Prefect
# Orchestrating ML Pipeline with Prefect

This repository demonstrates how to orchestrate an ML pipeline using Prefect, an open-source workflow management system. The pipeline involves converting an IPython notebook (ipynb) file to a Python script (py), setting up a Prefect server, deploying the project, and scheduling a run.

## Steps in Orchestration

1. **Have the IPython notebook (ipynb) file**

   Once you have completed all the data science tasks and analysis in your IPython notebook, save the file as an `.ipynb` file.

2. **Convert to Python script (py)**

   Convert the IPython notebook file to a Python script (`*.py`) with functions and a main function/flow. This step helps modularize the code and make it suitable for Prefect workflows.

3. **Install Prefect**

   Ensure you have Prefect installed. You can install it using `pip`:

   ```bash
   pip install prefect



