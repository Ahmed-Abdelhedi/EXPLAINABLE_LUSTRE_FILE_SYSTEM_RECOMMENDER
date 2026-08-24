import pytest

from preference_extractor.signal_detector.context_guard import (
    PreferenceContextGuard,
    PreferenceGuardDecision,
)


@pytest.fixture()
def guard():
    return PreferenceContextGuard()


@pytest.mark.parametrize(
    "text",
    [
        "If both options meet capacity, select I/O responsiveness over purchase cost.",
        "If both options respectent capacity, choose le coût d'exploitation over peak throughput.",
    ],
)
def test_force_positive_conditional_choice(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_SIGNAL
    assert result.reason == "conditional_choice_preference"


@pytest.mark.parametrize(
    "text",
    [
        "Can we optimize for energy efficiency, even at the expense of hardware density?",
        "Pouvons-nous optimiser la tolérance aux pannes, même au détriment de la vitesse brute ?",
        "Can we optimiser la réactivité des E/S, even au détriment de purchase cost?",
    ],
)
def test_force_positive_question_tradeoff(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_SIGNAL
    assert result.reason == "question_tradeoff_preference"


def test_force_positive_final_choice_mixed(guard):
    result = guard.resolve(
        "The sizing sheet lists 80 GB/s read throughput and 200 clients. "
        "Several designs satisfy those values. For the final choice, "
        "nous privilégions la réactivité des E/S over purchase cost."
    )
    assert result.decision == PreferenceGuardDecision.FORCE_SIGNAL
    assert result.reason == "final_choice_preference"


def test_force_positive_telegraphic_choice(guard):
    result = guard.resolve(
        "Choix production : le coût d'exploitation d'abord ; "
        "le débit de pointe ensuite."
    )
    assert result.decision == PreferenceGuardDecision.FORCE_SIGNAL
    assert result.reason == "telegraphic_ranked_choice"


@pytest.mark.parametrize(
    "text",
    [
        "The vendor brochure says 'fault tolerance is our priority'; "
        "that quotation is not our requirement.",

        "La brochure du fournisseur dit « l'efficacité énergétique est notre priorité » ; "
        "cette citation n'est pas notre exigence.",

        "The vendor brochure dit « la simplicité d'exploitation is our priority »; "
        "cette citation is not our requirement.",
    ],
)
def test_force_negative_third_party_quote(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "quoted_third_party_explicitly_rejected"


@pytest.mark.parametrize(
    "text",
    [
        "The minutes do not state that operational cost is preferred.",
        "Le compte rendu ne dit pas que la tolérance aux pannes est privilégiée.",
        "The minutes ne disent pas que l'efficacité énergétique is preferred.",
    ],
)
def test_force_negative_negated_report(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "negated_preference_report"


@pytest.mark.parametrize(
    "text",
    [
        "Le contrôleur peut régler automatiquement l'efficacité énergétique.",
        "The controller peut régler la réactivité des E/S automatically.",
    ],
)
def test_force_negative_capability(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "system_capability_not_user_preference"


@pytest.mark.parametrize(
    "text",
    [
        "A rejected draft would have prioritized fault tolerance; "
        "it does not describe the current request.",

        "A rejected draft would have prioritized operational simplicity; "
        "it does notdescribe the current request.",
    ],
)
def test_force_negative_rejected_draft(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "rejected_hypothesis_or_draft"


def test_force_negative_priority_queue(guard):
    result = guard.resolve(
        "La file de priorité planifie la tâche qui mesure l'efficacité énergétique."
    )
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "lexical_priority_queue_trap"


def test_force_negative_bare_deployment_fact(guard):
    result = guard.resolve(
        "Lustre will be deployed on the production cluster."
    )
    assert result.decision == PreferenceGuardDecision.FORCE_NO_SIGNAL
    assert result.reason == "bare_deployment_fact"


@pytest.mark.parametrize(
    "text",
    [
        "Reliability is our main priority.",
        "Cost is not important for us.",
        "We do not care about power consumption.",
        "Maximum power is 15 kW.",
        "The system requires 500 TiB usable storage.",
    ],
)
def test_non_guarded_cases_are_left_to_transformer(guard, text):
    result = guard.resolve(text)
    assert result.decision == PreferenceGuardDecision.PASS_TO_MODEL
