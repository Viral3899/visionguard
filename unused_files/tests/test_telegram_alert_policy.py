from app.alerts import telegram_event_eligible


def test_telegram_accepts_suspicious_events_at_or_above_50_percent():
    assert telegram_event_eligible('fight', 0.50)
    assert telegram_event_eligible('knife', '0.91')
    assert telegram_event_eligible('fire_detected', 0.75)
    assert not telegram_event_eligible('fight', 0.499)
    assert not telegram_event_eligible('person', 0.99)
    assert not telegram_event_eligible('loitering', 0.99)
    assert not telegram_event_eligible('intrusion', None)
    assert not telegram_event_eligible('weapon', 'not-a-score')
