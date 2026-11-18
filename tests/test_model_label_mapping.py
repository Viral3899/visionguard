from app.detectors import merge_fallback_names


def test_preserves_real_model_labels_instead_of_guessing_class_ids():
    assert merge_fallback_names({0: 'person', 1: 'knife'}, {0: 'fight', 1: 'assault'}) == {
        0: 'person', 1: 'knife'
    }


def test_uses_fallback_only_for_missing_or_numeric_labels():
    assert merge_fallback_names({0: '0', 1: 'smoke'}, {0: 'fire', 1: 'flame', 2: 'weapon'}) == {
        0: 'fire', 1: 'smoke', 2: 'weapon'
    }
