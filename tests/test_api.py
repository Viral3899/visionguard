from fastapi.testclient import TestClient
from app.main import create_app


def test_full_camera_crud_and_events(tmp_path):
    app=create_app(tmp_path/'test.db',start_cameras=False,auto_camera=False)
    with TestClient(app) as client:
        assert client.get('/api/stats').json()['cameras']==0
        r=client.post('/api/cameras',json={'name':'Gate A','source':'0','enabled':False})
        assert r.status_code==201, r.text
        cam=r.json(); cid=cam['id']
        assert cam['enabled'] is False
        r=client.patch(f'/api/cameras/{cid}',json={
            'zone':[[.1,.1],[.9,.1],[.9,.9]],'tripwire':[[.1,.5],[.9,.5]]})
        assert r.status_code==200
        assert len(r.json()['zone'])==3
        assert len(r.json()['tripwire'])==2
        event=app.state.db.store_event(cid,'Gate A','intrusion',7,'Person #7 entered restricted area')
        assert event['kind']=='intrusion'
        assert len(client.get('/api/events?kind=intrusion').json())==1
        assert client.get('/api/stats').json()['intrusions_24h']==1
        assert 'Person #7' in client.get('/api/events.csv').text
        assert client.get('/stream/'+cid).status_code==404
        assert client.delete(f'/api/cameras/{cid}').status_code==200
        assert client.get('/api/stats').json()['cameras']==0
        assert len(client.get('/api/events').json())==1  # audit history is retained
        assert client.delete(f'/api/cameras/{cid}').status_code==404


def test_schema_validations_and_redaction(tmp_path):
    app=create_app(tmp_path/'test.db',start_cameras=False)
    with TestClient(app) as client:
        assert client.post('/api/cameras',json={'name':'Bad','source':'javascript:alert(1)'}).status_code==422
        assert client.post('/api/cameras',json={'name':'Bad','source':'0','zone':[[.1,.1]]}).status_code==422
        r=client.post('/api/cameras',json={'name':'RTSP Gate','source':'rtsp://admin:secret@192.168.1.22:554/Streaming/Channels/101','enabled':False})
        assert r.status_code==201
        assert 'secret' not in r.text
        assert 'secret' not in client.get('/api/cameras').text
        assert client.get('/api/snapshots/../../../etc/passwd').status_code==404
        assert client.get('/api/events?limit=999').status_code==422
