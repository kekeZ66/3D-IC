CHAIN_SELECTION_PROMPT = """You are an expert in robot manipulation and navigation. Your task is to analyze the
feasibility of each candidate plan for successfully completing the given instruction.
You will provided with the task instruction and some candidate interaction plans,
where each plan includes a sequence of observations and the specific actions to be
executed at the corresponding locations.

The task instruction: {instruction}
Candidate interaction plans:
{interaction_chains}

For observations corresponding to the "navigate to {target_object}", consider:
- If the {target_object} is not visible, the likelihood that the surroundings indicate it
  is nearby.
- If the {target_object} is visible, proximity to the object and readiness for grasping.

For observations corresponding to the "pick {target_object}", consider:
- Visibility and clarity of the {target_object} for grasping.
- Accessibility of the object without occlusion or obstruction.
- Suitability of the current angle and position for reliable grasping.

For observations corresponding to the "navigate to {target_receptacle}", consider:
- If the {target_receptacle} is not visible, the likelihood that the surroundings indicate
  it is nearby.
- If the {target_receptacle} is visible, suitability of the current position for placement.

For observations corresponding to the "place at {target_receptacle}", consider:
- Obstacles or clutter that could make placement difficult.
- Stability of the object if placed here.
- Suitability of the current angle and position for placement.

Evaluate the feasibility of each interaction waypoint based on the above criteria, and
then combine these evaluations to determine an overall score for each plan.

Please provide a concise reasoning summary that compares all candidate plans. Then assign
a raw feasibility score (a continuous value between 0.0 and 1.0) to each plan. Output
your evaluation strictly in valid JSON format only, with no extra explanation or
natural language outside the JSON structure.

{{
    "reason": "<Concise explanation comparing the plans and justifying the scores>",
    "plan_1": <score_1>,
    "plan_2": <score_2>,
    "plan_3": <score_3>,
    "plan_4": <score_4>,
    "plan_5": <score_5>,
    "plan_6": <score_6>,
    "answer": <plan_ID>
}}
"""
