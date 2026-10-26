import textgrid


def qwen_to_textgrid(output, tier_name:str="words", output_name:str="output"):
    tg = textgrid.TextGrid(maxTime=audio_duration)
    tier = textgrid.IntervalTier(name=tier_name, minTime=0.0, maxTime=audio_duration)
    tier.add(start_time, end_time, tier_name)
    tg.append(tier)
    tg.write(f"{output_name}.TextGrid")
