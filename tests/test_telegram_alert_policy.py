from app.alerts import telegram_event_eligible


def test_telegram_requires_suspicious_type_and_strictly_more_than_75_percent():
    assert telegram_event_eligible('fight', 0.751)
    assert telegram_event_eligible('knife', '0.91')
    assert not telegram_event_eligible('fight', 0.75)
    assert not telegram_event_eligible('person', 0.99)
    assert not telegram_event_eligible('intrusion', None)
    assert not telegram_event_eligible('weapon', 'not-a-score')
