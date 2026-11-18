from fastapi.testclient import TestClient
from app.main import create_app


def test_dashboard_and_realtime_event_bus(tmp_path):
    app=create_app(tmp_path/'ws.db',start_cameras=False)
    with TestClient(app) as client:
        page=client.get('/')
        assert page.status_code==200
        assert 'Security overview' in page.text
        assert client.get('/static/app.js').status_code==200
        with client.websocket_connect('/ws/alerts') as websocket:
            app.state.bus.publish({'type':'event','event':{'kind':'intrusion','camera_name':'Gate A'}})
            received=websocket.receive_json()
            assert received['type']=='event'
            assert received['event']['kind']=='intrusion'
