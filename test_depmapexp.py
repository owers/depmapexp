import depmapexp as dme
import pandas as pd
import pandas.testing as pdt
import unittest
import unittest.mock

def test_preprocess_data():
    # Arrange
    data = {
        'gene1': [0.1, -0.6, -0.5],
        'gene2': [-0.55, -0.45, -0.5]
    }
    df = pd.DataFrame(data, index=['cell1', 'cell2', 'cell3'])
    expected_data = {
        'cell_line': ['cell1', 'cell2', 'cell1', 'cell2'],
        'gene': ['gene1', 'gene1', 'gene2', 'gene2'],
        'gene_effect': [0.1, -0.6, -0.55, -0.45]
    }
    expected_df = pd.DataFrame(expected_data, index=[0, 1, 3, 4])
    
    # Act
    processed_df = dme.preprocess_data(df)
    
    # Assert
    pdt.assert_frame_equal(processed_df, expected_df)

def test_read_data():
    # Arrange
    with unittest.mock.patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
        
        # Act
        result = dme.read_data('dummy_path.csv')
        
        # Assert
        pdt.assert_frame_equal(result, expected)

def test_create_experiment():
    # Arrange
    expected = {
        'cell_line': 'cellA',
        'KO': 'geneX',
        'viability': 0,
        'buckets': ['bucket1', 'bucket2'],
        'weight': 1.0
    }
    
    # Act
    result = dme.create_experiment('cellA', 'geneX', -0.6, ['bucket1', 'bucket2'])
    
    # Assert
    assert result == expected

def test_assign_buckets():
    # Arrange
    df = pd.DataFrame({
        'cell_line': ['cellA', 'cellB'],
        'gene': ['geneX', 'geneY']
    })
    cell_line_buckets = {'bucket1': ['cellA'], 'bucket2': ['cellB']}
    gene_buckets = {'bucket3': ['geneX'], 'bucket4': ['geneY']}
    expected = pd.DataFrame({
        'cell_line': ['cellA', 'cellB'],
        'gene': ['geneX', 'geneY'],
        'cell_line_buckets': [['bucket1'], ['bucket2']],
        'gene_buckets': [['bucket3'], ['bucket4']]
    })
    
    # Act
    result = dme.assign_buckets(df, cell_line_buckets, gene_buckets)
    
    # Assert
    pdt.assert_frame_equal(result, expected)

def test_determine_gene_selectivity():
    # Arrange
    df = pd.DataFrame({
        'gene': ['geneX', 'geneX', 'geneX', 'geneX', 'geneX', 'geneX', 'geneY', 'geneY', 'geneY', 'geneY', 'geneY', 'geneY'],
        'gene_effect': [-0.65, -0.35, -0.62, -0.38, -0.68, -0.42, -0.52, -0.53, -0.54, -0.55, -0.56, -0.57]
    })
    expected = pd.DataFrame({
        'gene': ['geneX', 'geneX', 'geneX', 'geneX', 'geneX', 'geneX', 'geneY', 'geneY', 'geneY', 'geneY', 'geneY', 'geneY'],
        'gene_effect': [-0.65, -0.35, -0.62, -0.38, -0.68, -0.42, -0.52, -0.53, -0.54, -0.55, -0.56, -0.57],
        'selectivity': pd.Series(['selective', 'selective', 'selective', 'selective', 'selective', 'selective', 'common_essential', 'common_essential', 'common_essential', 'common_essential', 'common_essential', 'common_essential'])
    })
    
    # Act
    result = dme.determine_gene_selectivity(df)
    result['selectivity'] = result['selectivity'].astype(str)
    
    # Assert
    pdt.assert_frame_equal(result, expected)

def test_select_top_n_samples():
    # Arrange
    df = pd.DataFrame({
        'gene': ['geneX', 'geneX', 'geneX', 'geneY', 'geneY'],
        'gene_effect': [-0.6, -0.4, -0.7, -0.5, -0.3]
    })
    expected = pd.DataFrame({
        'gene': ['geneX', 'geneX', 'geneY', 'geneY'],
        'gene_effect': [-0.7, -0.4, -0.5, -0.3]
    })
    
    # Act
    result = dme.select_top_n_samples(df, 1)
    
    # Assert
    pdt.assert_frame_equal(result, expected)

def test_create_experiments():
    # Arrange
    df = pd.DataFrame({
        'cell_line': ['cellA', 'cellB'],
        'gene': ['geneX', 'geneY'],
        'gene_effect': [-0.6, -0.4],
        'cell_line_buckets': [['bucket1'], ['bucket2']]
    })
    expected = [
        {'cell_line': 'cellA', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellB', 'KO': 'geneY', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0}
    ]
    
    # Act
    result = dme.create_experiments(df)
    
    # Assert
    assert result == expected

def test_stratified_split():
    # Arrange
    experiments = [
        {'cell_line': 'cellA', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellB', 'KO': 'geneX', 'viability': 1, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellC', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellD', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellE', 'KO': 'geneX', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellF', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellG', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellH', 'KO': 'geneX', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellI', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellJ', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0}
    ]
    expected_train = [
        {'cell_line': 'cellA', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellC', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellD', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellF', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket2'], 'weight': 1.0}
    ]
    expected_test = [
        {'cell_line': 'cellB', 'KO': 'geneX', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellE', 'KO': 'geneX', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0}
    ]
    
    # Act
    train_set, test_set = dme.stratified_split(experiments)
    
    # Assert
    assert train_set == expected_train
    assert test_set == expected_test

def test_assign_weights():
    # Arrange
    experiments = [
        {'cell_line': 'cellA', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellB', 'KO': 'geneY', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellC', 'KO': 'geneZ', 'viability': 0, 'buckets': ['bucket3'], 'weight': 1.0},
        {'cell_line': 'cellD', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket4'], 'weight': 1.0},
        {'cell_line': 'cellE', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket5'], 'weight': 1.0},
        {'cell_line': 'cellF', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket1'], 'weight': 1.0},
        {'cell_line': 'cellG', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket6'], 'weight': 1.0},
        {'cell_line': 'cellH', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket7'], 'weight': 1.0},
    ]
    important_bucket = 'bucket1'
    expected = [
        {'cell_line': 'cellA', 'KO': 'geneX', 'viability': 0, 'buckets': ['bucket1'], 'weight': 2.0},
        {'cell_line': 'cellB', 'KO': 'geneY', 'viability': 1, 'buckets': ['bucket2'], 'weight': 1.0},
        {'cell_line': 'cellC', 'KO': 'geneZ', 'viability': 0, 'buckets': ['bucket3'], 'weight': 1.0},
        {'cell_line': 'cellD', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket4'], 'weight': 1.0},
        {'cell_line': 'cellE', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket5'], 'weight': 1.0},
        {'cell_line': 'cellF', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket1'], 'weight': 2.0},
        {'cell_line': 'cellG', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket6'], 'weight': 1.0},
        {'cell_line': 'cellH', 'KO': 'geneZ', 'viability': 1, 'buckets': ['bucket7'], 'weight': 1.0},
    ]
    
    # Act
    result = dme.assign_weights(experiments, important_bucket)
    
    print([w['weight'] for w in expected])
    print([w['weight'] for w in result])

    # Assert
    assert result == expected
