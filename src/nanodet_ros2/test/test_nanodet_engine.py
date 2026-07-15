import numpy as np

from nanodet_ros2.nanodet_engine import (
    COCO_CLASS_NAMES,
    _hard_nms,
    decode_predictions,
    generate_center_priors,
)


def test_center_priors_match_320_model_shape():
    priors = generate_center_priors((320, 320), (8, 16, 32, 64))

    assert priors.shape == (2125, 4)
    np.testing.assert_array_equal(priors[0], [0.0, 0.0, 8.0, 8.0])
    np.testing.assert_array_equal(priors[-1], [256.0, 256.0, 64.0, 64.0])


def test_hard_nms_removes_overlapping_lower_score_box():
    boxes = np.asarray(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20, 20, 30, 30]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    assert _hard_nms(boxes, scores, iou_threshold=0.5, max_detections=10) == [0, 2]


def test_decode_predictions_returns_scaled_detection():
    priors = generate_center_priors((320, 320), (8, 16, 32, 64))
    predictions = np.zeros((1, priors.shape[0], 112), dtype=np.float32)
    target_index = 41  # First cell of the second row: center (8, 8), stride 8.
    predictions[0, target_index, 0] = 0.9

    regression = predictions[0, target_index, 80:].reshape(4, 8)
    regression.fill(-10.0)
    regression[:, 1] = 10.0

    detections = decode_predictions(
        predictions=predictions,
        priors=priors,
        original_size=(640, 480),
        input_size=(320, 320),
        class_names=COCO_CLASS_NAMES,
        score_threshold=0.4,
        nms_threshold=0.6,
        max_detections=100,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "person"
    assert detection.score == np.float32(0.9)
    np.testing.assert_allclose(detection.box, (0.0, 0.0, 32.0, 24.0), atol=1e-3)


def test_decode_predictions_rejects_wrong_output_shape():
    priors = generate_center_priors((320, 320), (8, 16, 32, 64))
    predictions = np.zeros((1, priors.shape[0], 111), dtype=np.float32)

    try:
        decode_predictions(
            predictions,
            priors,
            (320, 320),
            (320, 320),
            COCO_CLASS_NAMES,
            0.4,
            0.6,
            100,
        )
    except ValueError as error:
        assert "output channels" in str(error)
    else:
        raise AssertionError("decode_predictions accepted an invalid model output")
