import numpy as np
from app.detectors import ObjectDetector, DetectionResult, get_category_style, format_daily_object_label
from app.config import Settings

def test_empty_object_classes_allows_all():
    s = Settings(object_classes=())
    assert s.object_classes == ()

def test_category_styles():
    assert get_category_style('cell phone')['category'] == 'electronics'
    assert get_category_style('car')['category'] == 'vehicle'
    assert get_category_style('backpack')['category'] == 'daily'
    assert get_category_style('dog')['category'] == 'animal'
    assert get_category_style('chair')['category'] == 'furniture'

def test_draw_daily_objects_renders_multiple_categories():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = DetectionResult(
        boxes=[[10, 10, 80, 80], [100, 100, 200, 200]],
        classes=[67, 2],  # cell phone, car
        confidences=[0.92, 0.88],
        class_names={67: 'cell phone', 2: 'car'}
    )
    annotated = detector.draw_daily_objects(frame, result)
    assert annotated is not None
    assert annotated.shape == frame.shape
    assert np.any(annotated > 0)
