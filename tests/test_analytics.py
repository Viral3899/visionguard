from app.analytics import Analytics, crossing_direction, point_in_polygon


def test_polygon_footprint():
    zone = [(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)]
    assert point_in_polygon((.5,.5),zone)
    assert not point_in_polygon((.1,.5),zone)


def test_crossing_finite_segment_and_direction():
    line=[(.2,.5),(.8,.5)]
    assert crossing_direction((.5,.45),(.5,.55),line)=='entry'
    assert crossing_direction((.5,.55),(.5,.45),line)=='exit'
    assert crossing_direction((.95,.45),(.95,.55),line) is None
    assert crossing_direction((.5,.3),(.5,.7),line) is None  # unusually large jump ignored


def test_debounced_intrusion_cooldown_and_reentry():
    analytics=Analytics(zone=[(.2,.2),(.8,.2),(.8,.8),(.2,.8)],cooldown=10)
    p=lambda x,y: [{'id':3,'foot':(x,y)}]
    assert analytics.process(p(.5,.5),now=0)==[]
    assert analytics.process(p(.5,.5),now=1)[0]['kind']=='intrusion'
    assert analytics.process(p(.5,.5),now=2)==[]
    analytics.process(p(.1,.5),now=3)
    analytics.process(p(.1,.5),now=4)
    analytics.process(p(.5,.5),now=5)
    assert analytics.process(p(.5,.5),now=6)==[]
    analytics.process(p(.1,.5),now=11)
    analytics.process(p(.1,.5),now=12)
    analytics.process(p(.5,.5),now=13)
    assert analytics.process(p(.5,.5),now=14)[0]['kind']=='intrusion'


def test_occupancy_and_track_cleanup():
    analytics=Analytics()
    analytics.process([{'id':9,'foot':(.3,.2)}])
    assert analytics.current_people==1
    for _ in range(100): analytics.process([])
    assert analytics.current_people==0
    assert 9 not in analytics.tracks
