# Depmap Experiment Creation

This is a README file for the depmapexp project.

The project is structured as follows:

- `depmapexp.py`: This file contains the main functions for data preprocessing, experiment creation, and data analysis.
- `test_depmapexp.py`: This file contains unit tests for the functions in `depmapexp.py`.


To configure and execute the main file (`depmapexp.py`), follow these steps:

1. Open the `config.toml` file and configure the necessary parameters.
2. Run the main file with the command `python depmapexp.py`.

To run the unit tests in `test_depmapexp.py`, install `pytest` and use the command `pytest test_depmapexp.py`.

The dataset used in this project can be downloaded from here: https://depmap.org/portal/download/all/.


## Configuration Values

The configuration file (`config.toml`) contains the following values:

- `file.file_path`: The path to the input CSV file.
- `params.n`: The number of experiments to create.
- `params.important_bucket`: The name of the bucket that should have an emphasized weight.
- `cell_line_buckets`: A dictionary of cell line buckets. Each key is a bucket name and the corresponding value is a list of cell lines.
- `gene_buckets`: A dictionary of gene buckets. Each key is a bucket name and the corresponding value is a list of genes.
