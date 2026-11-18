from dataclasses import replace
from fastapi.testclient import TestClient
import app.main as module
from app.config import settings


def test_password_protects_api_and_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'settings', replace(settings, dashboard_password='difficult-demo-password'))
    app=module.create_app(tmp_path/'auth.db',start_cameras=False)
    with TestClient(app) as client:
        assert client.get('/api/stats').status_code==401
        assert client.get('/api/session').json()=={'password_required':True,'authenticated':False}
        assert client.post('/api/login',json={'password':'wrong'}).status_code==401
        assert client.post('/api/login',json={'password':'difficult-demo-password'}).status_code==200
        assert client.get('/api/stats').status_code==200
        assert client.post('/api/logout').status_code==200
        assert client.get('/api/stats').status_code==401


def test_admin_client_camera_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'settings', replace(settings, admin_password='admin-test-password', dashboard_password=''))
    app=module.create_app(tmp_path/'tenant.db',start_cameras=False,auto_camera=False)
    with TestClient(app) as admin:
        assert admin.post('/api/login',json={'username':'admin','password':'admin-test-password'}).status_code==200
        r=admin.post('/api/clients',json={'name':'Client Home','username':'clientone','password':'client-test-password'})
        assert r.status_code==201, r.text
        client_id=r.json()['id']
        assigned=admin.post('/api/cameras',json={'name':'Client Camera','source':'0','enabled':False,'client_id':client_id})
        assert assigned.status_code==201, assigned.text
        unassigned=admin.post('/api/cameras',json={'name':'Admin Camera','source':'1','enabled':False})
        assert unassigned.status_code==201
        cid=assigned.json()['id']
    with TestClient(app) as client:
        assert client.post('/api/login',json={'username':'clientone','password':'client-test-password'}).status_code==200
        cameras=client.get('/api/cameras').json()
        assert [c['name'] for c in cameras]==['Client Camera']
        assert client.patch(f'/api/cameras/{cid}',json={'source':'1'}).status_code==403
        assert client.post('/api/cameras',json={'name':'Nope','source':'0'}).status_code==403
        assert client.get('/api/clients').status_code==403


def test_admin_source_edit_endpoint_is_not_exposed_to_client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'settings', replace(settings, admin_password='admin-test-password', dashboard_password=''))
    app=module.create_app(tmp_path/'source.db',start_cameras=False,auto_camera=False)
    with TestClient(app) as admin:
        admin.post('/api/login',json={'username':'admin','password':'admin-test-password'})
        created=admin.post('/api/cameras',json={'name':'Private RTSP','source':'rtsp://user:pass@10.0.0.2/live','enabled':False}).json()
        cid=created['id']
        raw=admin.get(f'/api/cameras/{cid}/source')
        assert raw.status_code==200 and raw.json()['source']=='rtsp://user:pass@10.0.0.2/live'
        client=admin.post('/api/clients',json={'name':'Tenant','username':'tenantuser','password':'tenant-pass-123'}).json()
        admin.patch(f'/api/cameras/{cid}',json={'client_id':client['id']})
    with TestClient(app) as tenant:
        tenant.post('/api/login',json={'username':'tenantuser','password':'tenant-pass-123'})
        assert tenant.get(f'/api/cameras/{cid}/source').status_code==403
