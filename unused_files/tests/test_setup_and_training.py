"""Offline tests: model manifest, dataset conversion, class mapping, credentials."""
from pathlib import Path
import textwrap
import cv2
import numpy as np
import yaml
import pytest
from prepare_datasets import valid_yolo, parse_voc, materialize
from init_config import initialize
from model_registry import SPECS
from install_security_models import verify_labels


def test_manifest_exact_filenames_and_no_placeholder():
    assert {x.key for x in SPECS}=={'fire','fight','threat','3dcnn'}
    assert next(x.filename for x in SPECS if x.key=='fire')=='best.pt'
    assert next(x.filename for x in SPECS if x.key=='threat')=='weights/best.pt'
    assert all('<' not in x.filename for x in SPECS)


def test_yolo_source_class_mapping_and_invalid_coordinates():
    lines=valid_yolo('0 .5 .5 .5 .5\n1 0.2 0.3 0.1 0.1\n7 .5 .5 .1 .1\n1 100 0 0 0', {0:0,1:1})
    assert len(lines)==2 and lines[0].startswith('0 ') and lines[1].startswith('1 ')


def test_voc_to_yolo_conversion(tmp_path):
    sample=tmp_path/'sample.xml'
    sample.write_text('<annotation><size><width>200</width><height>100</height></size>'
                      '<object><name>knife</name><bndbox><xmin>50</xmin>'
                      '<ymin>10</ymin><xmax>100</xmax><ymax>40</ymax></bndbox></object>'
                      '<object><name>unknown</name></object></annotation>')
    assert parse_voc(sample)==['1 0.375000 0.250000 0.250000 0.300000']


def test_dataset_split_and_no_duplicate_image_leak(tmp_path):
    records=[]
    for idx in range(7):
        img=tmp_path/(str(idx)+'.png')
        assert cv2.imwrite(str(img),np.full((12,12,3), idx*30,dtype=np.uint8))
        records.append((img,['1 0.5 0.5 0.3 0.3'],'synthetic'))
    records.append(records[0])
    target=tmp_path/'prepared'
    path=materialize(records,target)
    assert path.is_file()
    assert len(list((target/'train'/'images').glob('*')))==6
    assert len(list((target/'val'/'images').glob('*')))==1
    assert yaml.safe_load(path.read_text())['names']=={0:'knife'}


def test_empty_or_wrong_labels_do_not_verify():
    assert verify_labels('weapon',['person','Knife'])
    with pytest.raises(ValueError):verify_labels('weapon',['person','car'])


def test_secure_init_does_not_change_existing_password(tmp_path):
    (tmp_path/'.env.example').write_text('ADMIN_PASSWORD=\nSESSION_SECRET=CHANGE_THIS\n')
    initialize(tmp_path)
    first=(tmp_path/'.env').read_text()
    assert 'CHANGE_THIS' not in first
    assert len(first.split('ADMIN_PASSWORD=')[1].splitlines()[0])>16
    initialize(tmp_path)
    assert (tmp_path/'.env').read_text()==first
