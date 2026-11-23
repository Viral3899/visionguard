from verify_models import verify
if __name__ == '__main__':
    rows=verify()
    if not any(name=='fire' and status=='READY' for name,status,_,_ in rows):raise SystemExit(1)
