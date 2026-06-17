PROMPTS = {
    "baseline": (
        "What is the skill level of the person in this video? "
        "Answer only one: Novice / Early Expert / Intermediate Expert / Late Expert"
    ),

    "binary": (
        "Is this person a beginner or an expert at this activity? "
        "Answer only: Beginner or Expert"
    ),

    "structured": (
        "Watch this video carefully.\n"
        "Step 1: Describe the person's body position and technique in detail.\n"
        "Step 2: Identify any errors or imprecisions in their movement.\n"
        "Step 3: Based only on what you observed, classify skill level: "
        "Novice / Early Expert / Intermediate Expert / Late Expert.\n"
        "Format your answer as:\n"
        "Observations: ...\n"
        "Errors: ...\n"
        "Skill Level: ..."
    ),

    "reasoning": (
        "You are an expert coach evaluating this person's technique.\n"
        "Focus on: body alignment, movement fluency, technical precision.\n"
        "What is their skill level? "
        "Novice / Early Expert / Intermediate Expert / Late Expert\n"
        "Explain your reasoning."
    )
}
