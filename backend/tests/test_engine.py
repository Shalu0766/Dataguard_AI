import pandas as pd
from app.main import analyze, hash_password, verify_password, cleaned_frame, CleanRequest

def test_password_hash_roundtrip():
    encoded = hash_password('correct-horse-42')
    assert encoded != 'correct-horse-42'
    assert verify_password('correct-horse-42', encoded)
    assert not verify_password('incorrect-password', encoded)

def test_detects_missing_duplicate_email_and_negative_age():
    frame = pd.DataFrame({
        'email': ['bad-address', None, 'good@example.org', 'good@example.org'],
        'age': [24, -2, 33, 33],
        'score': [1, 2, 4, 4],
    })
    result = analyze(frame)
    by_category = {issue['category'] for issue in result['issues']}
    assert {'Completeness', 'Uniqueness', 'Validity'} <= by_category
    assert 0 <= result['overall_score'] <= 100
    assert result['quality_category'] in {'Excellent', 'Good', 'Needs Improvement', 'Poor'}
    assert any('negative' in issue['description'] for issue in result['issues'])

def test_outliers_are_described_as_suspicious_not_confirmed_errors():
    frame = pd.DataFrame({'income': [10, 11, 10, 12, 11, 1000]})
    result = analyze(frame)
    flagged = [issue for issue in result['issues'] if issue['category'] == 'Plausibility']
    assert flagged
    assert 'not confirmed errors' in flagged[0]['description']

def test_custom_weights_are_normalized_and_scores_stay_bounded():
    result = analyze(pd.DataFrame({'x':[1,2,3]}), custom_weights={'Completeness': 1.0, 'Validity': 1.0})
    assert sum(result['weights'].values()) == 1.0
    assert result['weights']['Completeness'] == result['weights']['Validity']
    assert 0 <= result['overall_score'] <= 100

def test_single_row_and_empty_columns_return_bounded_score():
    result = analyze(pd.DataFrame({'empty': [None], 'constant': [1]}))
    assert 0 <= result['overall_score'] <= 100
    assert len(result['profiles']) == 2

def test_target_imbalance_and_cross_split_overlap_are_flagged():
    frame = pd.DataFrame({
        'feature': [1, 1, 2, 3, 4],
        'target': ['yes', 'yes', 'yes', 'yes', 'no'],
        'split': ['train', 'test', 'train', 'train', 'test'],
    })
    result = analyze(frame, target_column='target', split_column='split')
    assert result['readiness']['class_imbalance'] == .8
    assert result['readiness']['cross_split_duplicate_rows'] == 2
    assert any('imbalanced' in issue['description'] for issue in result['issues'])
    assert any('more than one split' in issue['description'] for issue in result['issues'])

def test_cleaning_preview_transformations_are_separate_and_composable():
    source = pd.DataFrame({'name': [' A ', 'A', 'B'], 'age': [10, 10, 200], 'empty': [None, None, None]})
    cleaned, operations = cleaned_frame(source, CleanRequest(remove_duplicates=True, trim_whitespace=True,
                                      remove_empty_columns=True, range_rules={'age': {'max': 100}}))
    assert len(source) == 3 and 'empty' in source.columns
    assert len(cleaned) == 2 and 'empty' not in cleaned.columns
    assert pd.isna(cleaned.loc[2, 'age'])
    assert 'remove_duplicates' in operations and any(x.startswith('range:age') for x in operations)

def test_cleaning_can_standardize_text_case():
    cleaned, operations = cleaned_frame(pd.DataFrame({'city':['New YORK','bOSTON']}), CleanRequest(text_case='lower'))
    assert cleaned['city'].tolist() == ['new york','boston']
    assert 'text_case:lower' in operations
