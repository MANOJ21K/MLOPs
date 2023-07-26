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


4. **Start the Prefect Server**
Start the Prefect server to manage your workflow runs:


5. **Configure the Prefect Server**
Configure the Prefect server by setting the API URL. Replace the URL with the actual URL of your Prefect server:


6. **Prefect Project Initialization**
Run the command below to initialize the Prefect YAML file that will hold all the necessary files for deployment:


7. **Prefect Deployment**
Deploy the orchestration.py script containing the main flow to Prefect. Replace `'orchestration.py'` with the actual filename and function name of your Prefect flow:


8. **Start the Prefect Worker**
Start the Prefect worker to execute the scheduled runs. Replace `'zoomcamp1'` with the pool name or resource identifier:


9. **Schedule/Run the Workflow**
Schedule and run the Prefect deployment for your main flow with the following command:


## Contributing

Pull requests and contributions are welcome. For major changes, please open an issue first to discuss the changes.

## License

This project is licensed under the [MIT License](LICENSE).
