import textgrid


def words_to_textgrid(words, duration: float, path: str, tier_name: str = "words") -> str:
    """words: [{"word","start","end"}, ...] in seconds. Writes a contiguous Praat TextGrid."""
    tg = textgrid.TextGrid(minTime=0.0, maxTime=duration)
    tier = textgrid.IntervalTier(name=tier_name, minTime=0.0, maxTime=duration)
    prev = 0.0
    for w in words:
        s = max(w["start"], prev)  # no overlaps
        e = min(max(w["end"], s + 0.001), duration)
        if e <= s:
            continue
        if s > prev:
            tier.add(prev, s, "")
        tier.add(s, e, w["word"])
        prev = e
    if prev < duration:
        tier.add(prev, duration, "")
    tg.append(tier)
    tg.write(path)
    return path