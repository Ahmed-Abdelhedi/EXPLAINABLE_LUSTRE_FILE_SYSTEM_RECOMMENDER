# Preference Signal Detector — Layer 1 Dataset V1

Binary task:
- 0 = NO_PREFERENCE_SIGNAL
- 1 = PREFERENCE_SIGNAL

Scope: detect whether the message contains at least one user preference about cost, power, performance or reliability.

Important:
- `Maximum power is 15 kW.` -> NO
- `Keeping power consumption low is very important.` -> YES
- `Cost is not important.` -> YES
- `We need 500 TiB and 200 clients.` -> NO
- `We need 500 TiB, but reliability is critical.` -> YES

Splits:
- TRAIN: 20,000
- VALIDATION: 4,000
- TEST: 4,000
- TOTAL: 28,000

Every split is:
- 50% YES / 50% NO
- 40% English / 40% French / 20% mixed
- balanced across 20 stress families

Use for training:
- input column: `text`
- target column: `label_id`

Recommended model:
`distilbert-base-multilingual-cased`

Protocol:
TRAIN -> VALIDATION -> freeze model/checkpoint -> TEST once.
