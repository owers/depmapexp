import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from typing import Any, Dict, List, Tuple
import toml

def read_data(file_path: str) -> pd.DataFrame:
    """
    This function reads the input data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """ 
    return pd.read_csv(file_path, index_col=0, nrows=1)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Preprocesses the input DataFrame by melting it into a long format and filtering out rows based on gene effect values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be preprocessed.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame in long format with filtered gene effect values.
    """
    df = df.melt(ignore_index=False, var_name='gene', value_name='gene_effect').reset_index()
    df.columns = ['cell_line', 'gene', 'gene_effect']
    df = df[(df['gene_effect'] <= -0.55) | (df['gene_effect'] >= -0.45)]
    return df

def create_experiment(cell_line: str, gene: str, gene_effect: float, buckets: List[str]) -> Dict[str, Any]:
    """
    Create an experiment dictionary based on gene effect.
    
    Args:
        cell_line (str): Name of the cell line.
        gene (str): Name of the gene.
        gene_effect (float): Effect of gene knockout.
        buckets (List[str]): List of buckets the experiment belongs to.
    
    Returns:
        Dict[str, Any]: Experiment dictionary.
    """
    viability = 0 if gene_effect < -0.5 else 1
    return {
        "cell_line": cell_line,
        "KO": gene,
        "viability": viability,
        "buckets": buckets,
        "weight": 1.0
    }

def assign_buckets(df: pd.DataFrame, cell_line_buckets: Dict[str, List[str]], gene_buckets: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Assign cell line and gene buckets to each experiment.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        cell_line_buckets (Dict[str, List[str]]): Dictionary of cell line buckets.
        gene_buckets (Dict[str, List[str]]): Dictionary of gene buckets.
    
    Returns:
        pd.DataFrame: DataFrame with assigned buckets.
    """
    def get_buckets(item, bucket_dict):
        return [bucket for bucket, items in bucket_dict.items() if item in items]

    df['cell_line_buckets'] = df['cell_line'].apply(lambda x: get_buckets(x, cell_line_buckets))
    df['gene_buckets'] = df['gene'].apply(lambda x: get_buckets(x, gene_buckets))
    return df

def determine_gene_selectivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine gene selectivity based on the percentage of dead cell lines.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with gene selectivity information.
    """
    gene_stats = df.groupby('gene').agg({'gene_effect': lambda x: (x < -0.5).mean()})
    gene_stats['selectivity'] = pd.cut(gene_stats['gene_effect'], 
                                       bins=[-float('inf'), 0.15, 0.85, float('inf')],
                                       labels=['common_non_essential', 'selective', 'common_essential'])
    return df.merge(gene_stats[['selectivity']], left_on='gene', right_index=True)

def select_top_n_samples(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select the N most sensitive and N most resistant samples per gene.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        n (int): Number of samples to select for each category.
    
    Returns:
        pd.DataFrame: DataFrame with selected samples.
    """
    return df.groupby('gene').apply(lambda x: pd.concat([
        x.nsmallest(n, 'gene_effect'),
        x.nlargest(n, 'gene_effect')
    ])).reset_index(drop=True)

def create_experiments(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create a list of experiments from the processed DataFrame.
    
    Args:
        df (pd.DataFrame): Processed DataFrame.
    
    Returns:
        List[Dict[str, Any]]: List of experiment dictionaries.
    """
    experiments = []
    for _, row in df.iterrows():
        buckets = (row.get('cell_line_buckets', []) + row.get('gene_buckets', []) + [row['selectivity']])
        experiment = create_experiment(row['cell_line'], row['gene'], row['gene_effect'], buckets)
        experiments.append(experiment)
    return experiments

def stratified_split(experiments: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform stratified sampling to split data into training and test sets.
    
    Args:
        experiments (List[Dict[str, Any]]): List of experiments.
        test_size (float): Proportion of the dataset to include in the test split.
    
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Training and test sets.
    """
    df = pd.DataFrame(experiments)
    genes = df['KO'].unique()
    train_genes, test_genes = train_test_split(genes, test_size=test_size, stratify=df.groupby('KO')['buckets'].first())
    
    train_set = [exp for exp in experiments if exp['KO'] in train_genes]
    test_set = [exp for exp in experiments if exp['KO'] in test_genes]
    
    return train_set, test_set

def assign_weights(experiments: List[Dict[str, Any]], important_bucket: str) -> List[Dict[str, Any]]:
    """
    Assign weights to experiments, with emphasis on the 'IMPORTANT' bucket.
    
    Args:
        experiments (List[Dict[str, Any]]): List of experiments.
        important_bucket (str): Name of the important bucket.
    
    Returns:
        List[Dict[str, Any]]: List of experiments with assigned weights.
    """
    total_weight = len(experiments)
    important_weight = total_weight * 0.25
    important_count = sum(1 for exp in experiments if important_bucket in exp['buckets'])
    
    if important_count == 0:
        return experiments
    
    weight_per_important = important_weight / important_count
    # modify this for upsampling (if w(exp) > 5  then split)
    weighted_experiments = []
    for exp in experiments:
        if important_bucket in exp['buckets']:
            if weight_per_important > 5:
                num_copies = important_weight // 5
                for _ in range(num_copies):
                    new_exp = exp.copy()
                    new_exp['weight'] = 5.0
                    weighted_experiments.append(new_exp)
            else:
                exp['weight'] = important_weight
                weighted_experiments.append(exp)
        else:
            weighted_experiments.append(exp)
    
    return weighted_experiments

def save_to_json(data: List[Dict[str, Any]], file_path: str):
    """
    Save data to a JSON file.
    
    Args:
        data (List[Dict[str, Any]]): Data to be saved.
        file_path (str): Path to save the JSON file.
    """
    with jsonlines.open(file_path, 'w') as w:
        w.write_all(data)

def print_statistics(experiments: List[Dict[str, Any]]):
    """
    Print statistics about the dataset.
    
    Args:
        experiments (List[Dict[str, Any]]): List of experiments.
    """
    df = pd.DataFrame(experiments)

    statistics = [
        ["Total number of experiments", len(df)],
        ["Number of unique cell lines", df['cell_line'].nunique()],
        ["Number of unique genes", df['KO'].nunique()],
    ]

    print("Statistics:")
    print(tabulate(statistics, tablefmt="pipe"))

    print("\nViability distribution:")
    print(tabulate(df['viability'].value_counts(normalize=True).to_frame().reset_index().rename(columns={0: "Count"}).values.tolist(), headers=["Viability", "Count"]))

    print("\nTop 10 most common buckets:")
    print(tabulate(df['buckets'].explode().value_counts().head(10).to_frame().reset_index().rename(columns={0: "Count"}).values.tolist(), headers=["Bucket", "Count"]))

def plot_viability_distribution(experiments: List[Dict[str, Any]]):

    df = pd.DataFrame(experiments)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='viability', data=df)
    plt.title('Distribution of Viability Scores')
    plt.xlabel('Viability')
    plt.ylabel('Count')
    plt.show()

def main(file_path: str, n: int, cell_line_buckets: Dict[str, List[str]], gene_buckets: Dict[str, List[str]], important_bucket: str):
    """
    Main function to create experiments from the input data.
    """
    df = read_data(file_path)
    df = preprocess_data(df)
    df = assign_buckets(df, cell_line_buckets, gene_buckets)
    df = determine_gene_selectivity(df)
    df = select_top_n_samples(df, n)
    
    experiments = create_experiments(df)
    train_set, test_set = stratified_split(experiments)
    train_set = assign_weights(train_set, important_bucket)
    
    save_to_json(train_set, 'training_set.jsonl')
    save_to_json(test_set, 'test_set.jsonl')

    print_statistics(experiments)
    plot_viability_distribution(experiments)

if __name__ == "__main__":
    with open('config.toml', 'r') as file:
        config = toml.load(file)

    file_path = config['file']['file_path']
    n = config['params']['n']
    cell_line_buckets = config['cell_line_buckets']
    gene_buckets = config['gene_buckets']
    important_bucket = config['params']['important_bucket'][0]
    
    main(file_path, n, cell_line_buckets, gene_buckets, important_bucket)