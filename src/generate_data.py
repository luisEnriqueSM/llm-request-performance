import random
import csv

RANDOM_SEED = 42
NUM_REQUEST = 500

random.seed(RANDOM_SEED)

MODELS = {
    "small": 400,
    "medium": 800,
    "large":  1400
}

REGIONS = {
    "us-east": 0,
    "us-west": 120,
    "eu-west": 200
}

rows = []

for _ in range(NUM_REQUEST):
    model = random.choice(list(MODELS.keys()))
    region = random.choice(list(REGIONS.keys()))

    input_tokens = random.randint(100, 4000)
    output_tokens = random.randint(50, 1000)
    concurrent_requests = random.randint(1, 10)

    latency_ms = (
        MODELS[model]
        + input_tokens * 0.20
        + output_tokens * 1.20
        + concurrent_requests * 100
        + REGIONS[region]
        + random.randint(-200, 200)
    )

    latency_ms = max(100, int(latency_ms))

    rows.append(
        [
            model,
            input_tokens,
            output_tokens,
            concurrent_requests,
            region,
            latency_ms
        ]
    )

with open("data/llm_requests.csv", "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(
        [
            "model",
            "input_tokens",
            "output_tokens",
            "concurrent_requests",
            "region",
            "latency_ms"
        ]
    )
    writer.writerows(rows)