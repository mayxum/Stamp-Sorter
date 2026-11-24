CLASSES = ["nature", "people", "places", "graphic"]

MODEL = "openai/clip-vit-large-patch14"

PROMPTS = {
    "nature": "a postage stamp with a picture of wildlife or plants",
    "people": "a postage stamp with a portrait of a person's face",
    "places": "a postage stamp depicting a building or a place",
    "graphic": "a postage stamp showing a flag, a map, a coat of arms, a vehicle or an object",
}

# Below this confidence a stamp goes to the reject bin instead of a real one.
# Measured on 422 stamps: 0.60 gives 96.6% on what it accepts and rejects 16%,
# 0.70 gives 99.0% and rejects 29%. Raise it to bin fewer stamps more carefully,
# lower it to hand-sort less.
REJECT_THRESHOLD = 0.70
