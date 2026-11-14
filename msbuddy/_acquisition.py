"""Optional acquisition metadata with explicit collision-energy units."""
import math
import re


def instrument_family(value):
    text = str(value or '').lower().replace('-', '').replace(' ', '')
    if 'timstof' in text:
        return 'timstof'
    if 'orbitrap' in text or 'qexactive' in text:
        return 'orbitrap'
    if 'qtof' in text or 'qtimeofflight' in text:
        return 'qtof'
    if 'fticr' in text:
        return 'fticr'
    return None


def collision_energy(value, unit=None):
    """Return (value, unit) or (None, None); never guess bare numeric units."""
    if value is None or isinstance(value, bool):
        return None, None
    aliases = {'ev': 'ev', 'electronvolt': 'ev', 'electronvolts': 'ev',
               'nce': 'nce', '%': 'nce', 'percent': 'nce'}
    supplied_unit = aliases.get(str(unit or '').strip().lower())
    if isinstance(value, str):
        text = value.lower().strip()
        if (re.search(r'\d\s*[-,/–—]\s*\d', text)
                or any(word in text for word in ('stepped', 'ramp'))):
            return None, None
        ev = re.findall(r'(\d+(?:\.\d+)?)\s*ev\b', text)
        if len(ev) == 1:
            value, supplied_unit = ev[0], 'ev'
        else:
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if len(numbers) != 1:
                return None, None
            if 'nce' in text or '%' in text:
                supplied_unit = 'nce'
            value = numbers[0]
        if text.startswith(('-', '−')):
            return None, None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None, None
    if supplied_unit is None or not math.isfinite(number) or number < 0:
        return None, None
    return number, supplied_unit
