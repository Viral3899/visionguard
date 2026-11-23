import time
from pathlib import Path
import numpy as np
import pytest

from app.database import Database
from app.recorder import VideoRecorder


@pytest.fixture
def temp_dir(tmp_path):
    recordings_dir = tmp_path / 'recordings'
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / 'test_vg.db'
    db = Database(db_file)
    db.init()
    return db


def test_video_recorder_start_write_stop(temp_dir, temp_db):
    recorder = VideoRecorder(output_dir=temp_dir, db=temp_db)

    cam_id = 'cam_test_1'
    cam_name = 'Front Gate'

    # 1. Start recording
    rec_info = recorder.start_recording(cam_id, cam_name, width=320, height=240, fps=10.0)
    assert rec_info is not None
    assert rec_info['camera_id'] == cam_id
    assert recorder.is_recording(cam_id) is True

    status = recorder.get_status(cam_id)
    assert status['is_recording'] is True

    # 2. Write 15 mock frames
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(15):
        frame[50:100, 50:100] = (i * 10, 255 - i * 10, 128)
        recorder.write_frame(cam_id, frame)
        time.sleep(0.01)

    # 3. Stop recording
    completed = recorder.stop_recording(cam_id)
    assert completed is not None
    assert recorder.is_recording(cam_id) is False

    # Check file exists and has non-zero size
    video_path = Path(completed['file_path'])
    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert completed['frames_written'] >= 10
    assert completed['duration_seconds'] > 0

    # 4. Check database record
    records = temp_db.list_recordings(camera_id=cam_id)
    assert len(records) == 1
    assert records[0]['filename'] == completed['filename']
    assert records[0]['camera_name'] == cam_name


def test_recorder_duplicate_start_ignored(temp_dir, temp_db):
    recorder = VideoRecorder(output_dir=temp_dir, db=temp_db)
    rec1 = recorder.start_recording('cam_2', 'Backyard', 320, 240, 10.0)
    rec2 = recorder.start_recording('cam_2', 'Backyard', 320, 240, 10.0)
    assert rec1['id'] == rec2['id']
    recorder.stop_recording('cam_2')


def test_recorder_delete_recording(temp_dir, temp_db):
    recorder = VideoRecorder(output_dir=temp_dir, db=temp_db)
    recorder.start_recording('cam_3', 'Lobby', 320, 240, 10.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    recorder.write_frame('cam_3', frame)
    completed = recorder.stop_recording('cam_3')

    filename = completed['filename']
    assert (temp_dir / filename).is_file()

    # Delete via db/recorder helper
    deleted = temp_db.delete_recording(filename)
    assert deleted is True
    assert not (temp_dir / filename).exists()
    assert len(temp_db.list_recordings()) == 0
