"""Initialize local credentials without overwriting existing user settings."""
from pathlib import Path
import re
import secrets
import shutil

ROOT=Path(__file__).resolve().parent

def initialize(root=ROOT):
    env=root/'.env'
    if not env.is_file():
        shutil.copyfile(root/'.env.example',env)
        print('[config] Created .env from template')
    text=env.read_text(encoding='utf-8')
    pwd_match=re.search(r'(?m)^ADMIN_PASSWORD=(.*)$',text)
    new_password=None
    if pwd_match and not pwd_match.group(1).strip():
        new_password=secrets.token_urlsafe(18)
        text=re.sub(r'(?m)^ADMIN_PASSWORD=.*$',lambda _: 'ADMIN_PASSWORD='+new_password,text,count=1)
    if not pwd_match:
        new_password=secrets.token_urlsafe(18)
        text+='\nADMIN_PASSWORD='+new_password+'\n'
    session_match=re.search(r'(?m)^SESSION_SECRET=(.*)$',text)
    if not session_match or not session_match.group(1).strip() or session_match.group(1).startswith(('CHANGE_THIS','change-this')):
        secret=secrets.token_urlsafe(48)
        if session_match:
            text=re.sub(r'(?m)^SESSION_SECRET=.*$',lambda _: 'SESSION_SECRET='+secret,text,count=1)
        else:text+='\nSESSION_SECRET='+secret+'\n'
    # Migrate only the known obsolete default, never a user-provided custom path.
    text=text.replace('WEAPON_MODEL_PATH=models/cs2-yolo12-weapon-detection.pt',
                      'WEAPON_MODEL_PATH=models/threat-yolov8n.pt')
    env.write_text(text,encoding='utf-8')
    if new_password:
        print('\n[NEW ADMIN LOGIN]')
        print('Username: admin')
        print('Password:',new_password)
        print('Saved to .env (ADMIN_PASSWORD). Keep it private and change it if necessary.\n')
    else:
        print('Using existing .env admin credentials; no password overwritten.')

if __name__=='__main__':initialize()
