# Chat With Your Map

This project is a modular backend that interprets natural language queries for a GIS application.
It's built using the Haystack Framework to break down the query into its core components: Action, Layer, and Location.

## Project Structure

- master_agent.py: The main entry point. It builds the Haystack pipeline and orchestrates the different components to produce a final, structured JSON command.
- components/: This directory contains the individual, specialized agents (Haystack components).

  - action_extractor.py: Determines the primary action (e.g., "Zeigen", "Messen").
  - layer_extractor.py: Identifies the correct GIS layer based on the query and a list in data/layers.csv.
  - location_extractor.py: Extracts the location name (e.g., "Altona") and uses geopy to find its real-world coordinates.json_combiner.py: Merges the outputs from the other components into the final JSON structure.
- data/: Contains the CSV file with the list of available GIS layers.
- pyproject.toml: Defines all project dependencies for Poetry.
- .env: A file (that you will create) to store your secret API key.

## How to Run

Step 1: first create an conda environment with

```conda env create -f environment.yml -p .venv/ ```

Step 2: Install DependenciesThis project uses Poetry for dependency management.
Make sure you have it installed.In your terminal, navigate to the project root and run:

```bash poetry install ```

This will create a virtual environment and install all required packages, including haystack-ai, pandas, and geopy.

Step 3: Create API Key file .env. Open the .env file and replace the placeholder with your actual IONOS API key.

Step 4: Run the Master AgentActivate the Poetry virtual environment and run the main script.

## Activate the environment

```poetry shell ```

Run the master agent with a sample query

```python master_agent.py```

The script will print the final, consolidated JSON command to the console.
You can easily change the USER_QUERY variable in master_agent.py to test other inputs.
