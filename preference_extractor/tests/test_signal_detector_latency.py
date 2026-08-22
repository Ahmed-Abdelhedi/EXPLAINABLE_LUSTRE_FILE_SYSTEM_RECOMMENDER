import time

from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector
)


def test_inference_latency():

    detector = PreferenceSignalDetector()

    samples = [
        "Reliability is our main priority.",
        "Performance matters more than cost.",
        "Maximum power is 15 kW.",
        "The system requires 500 TiB storage.",
    ]

    warmup = 5

    for _ in range(warmup):
        detector.predict(samples[0])


    start = time.perf_counter()

    n = 100

    for i in range(n):
        detector.predict(
            samples[i % len(samples)]
        )

    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n) * 1000

    print(
        f"\nAverage inference latency: {avg_ms:.2f} ms"
    )

    # garde raisonnable CPU
    assert avg_ms < 5000