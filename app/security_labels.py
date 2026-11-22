"""Conservative label interpretation for security detections.

Never infer classes from numeric IDs: read metadata in each actual checkpoint.
Game-only Counter-Strike class names are not evidence of a real-world firearm.
"""
import re

CS2_FIREARMS = frozenset({
    'AK47', 'M4A1-S', 'M4A1', 'GALIL', 'FAMAS', 'TEC-9', 'FIVE-SEVEN',
    'GLOCK-18', 'USP-S', 'EAGLE', 'BERETTAS', 'P2000', 'MAC10', 'MP5',
    'MP9', 'P90', 'P250', 'SSG08', 'AWP',
})


def _words(label: str) -> list[str]:
    return re.sub(r'[^a-z0-9]+', ' ', str(label).lower()).split()


def is_fight_label(label: str) -> bool:
    words = _words(label)
    compact = ''.join(words)
    if any(t in compact for t in ('nonviolence', 'nonviolent', 'nofight', 'nonfight', 'normal')):
        return False
    if any(t in words for t in ('no', 'not', 'negative', 'safe', 'peaceful')):
        return False
    return any(t in words for t in ('fight', 'fighting', 'violence', 'violent',
                                    'assault', 'aggression', 'aggressive'))


def weapon_class(label: str) -> str | None:
    """Only explicit, real-world generic labels are admissible as security events.

    Do not translate CS2 weapon codes such as AK47 into real-world gun evidence:
    those models frequently classify videogame screenshots instead of CCTV.
    """
    normal = ' '.join(_words(label))
    if normal in ('knife', 'blade', 'dagger', 'machete'):
        return 'knife'
    if normal in ('gun', 'pistol', 'rifle', 'firearm', 'revolver', 'handgun'):
        return 'gun'
    if normal in ('grenade', 'explosive', 'explosives', 'weapon'):
        return 'weapon'
    return None
