import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / 'experiments' / 'adaptation' / 'train_lora.py'
spec = importlib.util.spec_from_file_location('train_lora_module', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_normalize_answer() -> None:
    assert module.normalize_answer('  YES  ') == 'yes'
    assert module.normalize_answer('Rural Area') == 'rural area'
    assert module.normalize_answer('  7  ') == '7'


def test_split_train_val_is_deterministic_and_nonempty() -> None:
    records = [{'id': index} for index in range(5)]

    train, validation = module.split_train_val(records, val_ratio=0.2, seed=42)
    repeat_train, repeat_validation = module.split_train_val(records, val_ratio=0.2, seed=42)

    assert train == repeat_train
    assert validation == repeat_validation
    assert len(train) == 4
    assert len(validation) == 1
    assert {item['id'] for item in train}.isdisjoint(item['id'] for item in validation)


def test_split_train_val_rejects_too_small_dataset() -> None:
    try:
        module.split_train_val([{'id': 1}], val_ratio=0.2)
    except ValueError as exc:
        assert 'At least two records' in str(exc)
    else:
        raise AssertionError('Expected a ValueError for a one-record dataset')
