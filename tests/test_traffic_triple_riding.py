from app.traffic import TripleRidingDetector

def test_triple_riding_single_and_double_rider_allowed():
    detector = TripleRidingDetector()
    moto = {'id': 1, 'bbox': [100, 100, 200, 250]}
    # 2 riders overlapping motorcycle
    persons = [
        {'id': 10, 'bbox': [110, 80, 150, 180]},
        {'id': 11, 'bbox': [150, 80, 190, 180]},
    ]
    violations = detector.detect_violations([moto], persons)
    assert len(violations) == 0

def test_triple_riding_three_riders_violation():
    detector = TripleRidingDetector()
    moto = {'id': 2, 'bbox': [100, 100, 250, 250]}
    # 3 riders overlapping motorcycle
    persons = [
        {'id': 20, 'bbox': [110, 70, 145, 170]},
        {'id': 21, 'bbox': [145, 70, 180, 170]},
        {'id': 22, 'bbox': [180, 70, 220, 170]},
    ]
    violations = detector.detect_violations([moto], persons)
    assert len(violations) == 1
    assert violations[0]['type'] == 'triple_riding'
    assert violations[0]['rider_count'] == 3
    assert violations[0]['motorcycle_id'] == 2
