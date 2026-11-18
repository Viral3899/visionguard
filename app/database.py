"""SQLite persistence: WAL, short per-request connections, parameterized SQL."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .config import settings


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    @contextmanager
    def connection(self):
        with sqlite3.connect(str(self.path), timeout=10) as cx:
            cx.row_factory = sqlite3.Row
            cx.execute('PRAGMA journal_mode=WAL;')
            cx.execute('PRAGMA busy_timeout=10000;')
            yield cx

    def init(self):
        with self.lock, self.connection() as cx:
            cx.executescript('''
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    zone TEXT NOT NULL DEFAULT '[]',
                    tripwire TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    client_id TEXT
                );
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    track_id INTEGER,
                    message TEXT NOT NULL,
                    snapshot TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, created_at DESC);
                CREATE TABLE IF NOT EXISTS totals (
                    camera_id TEXT PRIMARY KEY,
                    entered INTEGER NOT NULL DEFAULT 0,
                    exited INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS alert_log (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alert_log_event ON alert_log(event_id);
                CREATE TABLE IF NOT EXISTS camera_config (
                    camera_id TEXT PRIMARY KEY,
                    fire_enabled INTEGER NOT NULL DEFAULT 1,
                    violence_enabled INTEGER NOT NULL DEFAULT 1,
                    anomaly_enabled INTEGER NOT NULL DEFAULT 1,
                    object_enabled INTEGER NOT NULL DEFAULT 1,
                    telegram_enabled INTEGER NOT NULL DEFAULT 1,
                    tts_enabled INTEGER NOT NULL DEFAULT 1,
                    sensitivity REAL NOT NULL DEFAULT 1.0,
                    FOREIGN KEY(camera_id) REFERENCES cameras(id)
                );
            ''')
            # One-time migration: enable Telegram per-camera for existing installs.
            # The admin can turn it off per camera afterwards; never overwrite that choice on startup.
            cx.execute('CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            migrated = cx.execute("SELECT value FROM app_meta WHERE key='telegram_camera_opt_in_v1'").fetchone()
            if not migrated:
                cx.execute('UPDATE camera_config SET telegram_enabled=1')
                cx.execute("INSERT INTO app_meta(key,value) VALUES('telegram_camera_opt_in_v1','done')")
            # Safe schema migrations for multi-client ownership.
            try: cx.execute('ALTER TABLE cameras ADD COLUMN client_id TEXT')
            except sqlite3.OperationalError: pass
            # Migration: add metadata column if missing
            try:
                cx.execute('ALTER TABLE events ADD COLUMN metadata TEXT')
            except sqlite3.OperationalError:
                pass
            # Migration: add camera_config table entries for existing cameras
            cams = cx.execute('SELECT id FROM cameras').fetchall()
            for cam in cams:
                cx.execute('''
                    INSERT OR IGNORE INTO camera_config (camera_id, fire_enabled, violence_enabled, anomaly_enabled, object_enabled, telegram_enabled, tts_enabled, sensitivity)
                    VALUES (?, 1, 1, 1, 1, 1, 1, 1.0)
                ''', (cam['id'],))

    @staticmethod
    def camera(row, safe=True):
        item = dict(row)
        item['zone'] = json.loads(item['zone'])
        item['tripwire'] = json.loads(item['tripwire'])
        item['enabled'] = bool(item['enabled'])
        if safe:
            from urllib.parse import urlsplit, urlunsplit
            source = item['source']
            try:
                parts = urlsplit(source)
                if parts.netloc:
                    host = parts.hostname or ''
                    if ':' in host: host = f'[{host}]'
                    port = f':{parts.port}' if parts.port else ''
                    user = f'{parts.username or "user"}:***@' if parts.username or parts.password else ''
                    item['source'] = urlunsplit((parts.scheme, user + host + port, parts.path, '[REDACTED]' if parts.query else '', parts.fragment))
            except ValueError:
                item['source'] = '[configured stream]'
        return item

    def list_cameras(self, safe=True):
        with self.connection() as cx:
            return [self.camera(row, safe) for row in cx.execute('SELECT * FROM cameras ORDER BY created_at')]

    def get_camera(self, cam_id, safe=True):
        with self.connection() as cx:
            row = cx.execute('SELECT * FROM cameras WHERE id=?', (cam_id,)).fetchone()
        return self.camera(row, safe) if row else None

    def get_camera_config(self, cam_id):
        with self.connection() as cx:
            row = cx.execute('SELECT * FROM camera_config WHERE camera_id=?', (cam_id,)).fetchone()
        return dict(row) if row else None

    def update_camera_config(self, cam_id, changes):
        allowed = {'fire_enabled', 'violence_enabled', 'anomaly_enabled', 'object_enabled', 'telegram_enabled', 'tts_enabled', 'sensitivity'}
        values = {}
        for k, v in changes.items():
            if k not in allowed: continue
            values[k] = int(v) if k.endswith('_enabled') else v
        if not values: return self.get_camera_config(cam_id)
        with self.lock, self.connection() as cx:
            cx.execute('UPDATE camera_config SET '+', '.join(f'{k}=?' for k in values)+' WHERE camera_id=?', (*values.values(), cam_id))
        return self.get_camera_config(cam_id)

    def list_clients(self):
        with self.connection() as cx:
            return [dict(r) for r in cx.execute('SELECT id,name,username,active,created_at FROM clients ORDER BY created_at DESC')]

    def create_client(self, name, username, password_hash):
        cid = uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self.connection() as cx:
            cx.execute('INSERT INTO clients(id,name,username,password_hash,active,created_at) VALUES(?,?,?,?,1,?)',
                       (cid, name, username, password_hash, now))
        return {'id': cid, 'name': name, 'username': username, 'active': True, 'created_at': now}

    def get_client(self, client_id):
        with self.connection() as cx:
            row = cx.execute('SELECT * FROM clients WHERE id=?', (client_id,)).fetchone()
        return dict(row) if row else None

    def get_client_login(self, username):
        with self.connection() as cx:
            row = cx.execute('SELECT * FROM clients WHERE username=? AND active=1', (username,)).fetchone()
        return dict(row) if row else None

    def set_client_active(self, client_id, active):
        with self.lock, self.connection() as cx:
            cur = cx.execute('UPDATE clients SET active=? WHERE id=?', (int(active), client_id))
        return cur.rowcount > 0

    def create_camera(self, payload):
        cam_id = uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self.connection() as cx:
            cx.execute('INSERT INTO cameras (id,name,source,zone,tripwire,enabled,created_at,client_id) VALUES (?,?,?,?,?,?,?,?)', (
                cam_id, payload.name, payload.source, json.dumps(payload.zone),
                json.dumps(payload.tripwire), int(payload.enabled), now, getattr(payload, 'client_id', None)))
            cx.execute('INSERT INTO totals(camera_id) VALUES (?)', (cam_id,))
            cx.execute('''
                INSERT INTO camera_config (camera_id, fire_enabled, violence_enabled, anomaly_enabled, object_enabled, telegram_enabled, tts_enabled, sensitivity)
                VALUES (?, 1, 1, 1, 1, 1, 1, 1.0)
            ''', (cam_id,))
        return self.get_camera(cam_id)

    def update_camera(self, cam_id, changes):
        allowed = {'name', 'source', 'zone', 'tripwire', 'enabled', 'client_id'}
        values = {}
        for k, v in changes.items():
            if k not in allowed: continue
            values[k] = json.dumps(v) if k in ('zone', 'tripwire') else (int(v) if k == 'enabled' else v)
        if not values: return self.get_camera(cam_id)
        with self.lock, self.connection() as cx:
            result = cx.execute('UPDATE cameras SET '+', '.join(f'{k}=?' for k in values)+' WHERE id=?', (*values.values(), cam_id))
        return self.get_camera(cam_id) if result.rowcount else None

    def delete_camera(self, cam_id):
        with self.lock, self.connection() as cx:
            result = cx.execute('DELETE FROM cameras WHERE id=?', (cam_id,))
            cx.execute('DELETE FROM totals WHERE camera_id=?', (cam_id,))
            cx.execute('DELETE FROM camera_config WHERE camera_id=?', (cam_id,))
        return result.rowcount > 0

    def store_event(self, camera_id, camera_name, kind, track_id, message, snapshot=None, metadata=None):
        import json as json_lib
        item = dict(id=uuid4().hex, camera_id=camera_id, camera_name=camera_name,
                    kind=kind, track_id=track_id, message=message,
                    snapshot=snapshot, metadata=json_lib.dumps(metadata) if metadata else None,
                    created_at=datetime.now(timezone.utc).isoformat())
        with self.lock, self.connection() as cx:
            cx.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)', tuple(item.values()))
            if kind in ('entry','exit'):
                col = 'entered' if kind == 'entry' else 'exited'
                cx.execute(f'UPDATE totals SET {col}={col}+1 WHERE camera_id=?', (camera_id,))
        return item

    def events(self, camera_id=None, kind=None, limit=100):
        terms, params = [], []
        if camera_id:
            terms.append('camera_id=?'); params.append(camera_id)
        if kind:
            terms.append('kind=?'); params.append(kind)
        query = 'SELECT * FROM events' + (' WHERE ' + ' AND '.join(terms) if terms else '') + ' ORDER BY created_at DESC LIMIT ?'
        with self.connection() as cx:
            rows = cx.execute(query, (*params, limit)).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if item['metadata']:
                try:
                    item['metadata'] = json.loads(item['metadata'])
                except json.JSONDecodeError:
                    pass
            result.append(item)
        return result

    def stats(self):
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with self.connection() as cx:
            counts = cx.execute('SELECT COALESCE(SUM(entered),0) AS entered,COALESCE(SUM(exited),0) AS exited FROM totals').fetchone()
            lastday = cx.execute('SELECT COUNT(*) AS n FROM events WHERE created_at>=? AND kind=?', (since, 'intrusion')).fetchone()
            # Stats by event kind
            kind_stats = cx.execute('SELECT kind, COUNT(*) as n FROM events WHERE created_at>=? GROUP BY kind', (since,)).fetchall()
            per = {r['camera_id']: {'entered':r['entered'], 'exited':r['exited']} for r in cx.execute('SELECT * FROM totals')}
        return {
            'entered': counts['entered'], 
            'exited': counts['exited'], 
            'intrusions_24h': lastday['n'],
            'events_by_kind_24h': {r['kind']: r['n'] for r in kind_stats},
            'per_camera': per
        }

    def log_alert(self, event_id, channel, status, message=None):
        item = dict(id=uuid4().hex, event_id=event_id, channel=channel,
                    status=status, message=message,
                    created_at=datetime.now(timezone.utc).isoformat())
        with self.lock, self.connection() as cx:
            cx.execute('INSERT INTO alert_log VALUES (?,?,?,?,?,?)', tuple(item.values()))

    def alert_log(self, event_id=None, limit=100):
        terms, params = [], []
        if event_id:
            terms.append('event_id=?'); params.append(event_id)
        query = 'SELECT * FROM alert_log' + (' WHERE ' + ' AND '.join(terms) if terms else '') + ' ORDER BY created_at DESC LIMIT ?'
        with self.connection() as cx:
            return [dict(r) for r in cx.execute(query, (*params, limit))]

    def purge_old(self, days=None):
        days = days or settings.snapshot_retention_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.lock, self.connection() as cx:
            snapshots = [r['snapshot'] for r in cx.execute('SELECT snapshot FROM events WHERE created_at<? AND snapshot IS NOT NULL', (cutoff,))]
            cx.execute('DELETE FROM events WHERE created_at<?', (cutoff,))
            cx.execute('DELETE FROM alert_log WHERE created_at<?', (cutoff,))
        for name in snapshots:
            if name and Path(name).name == name:
                (settings.data_dir / 'snapshots' / name).unlink(missing_ok=True)

    def clear_all_data(self):
        """Clear all events, alert logs, and snapshots for a fresh start."""
        with self.lock, self.connection() as cx:
            # Get all snapshot names to delete files
            snapshots = [r['snapshot'] for r in cx.execute('SELECT snapshot FROM events WHERE snapshot IS NOT NULL')]
            # Clear all events and alert logs
            cx.execute('DELETE FROM events')
            cx.execute('DELETE FROM alert_log')
            # Reset totals (only columns that exist in schema)
            cx.execute('UPDATE totals SET entered=0, exited=0')
        # Delete snapshot files
        for name in snapshots:
            if name and Path(name).name == name:
                (settings.data_dir / 'snapshots' / name).unlink(missing_ok=True)
        # Clear test video output
        test_dir = settings.data_dir / 'test_videos'
        if test_dir.exists():
            for f in test_dir.glob('*'):
                f.unlink(missing_ok=True)