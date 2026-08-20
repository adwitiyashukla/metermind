from __future__ import annotations

import numpy as np
import pytest

from metermind.models.shapes import (
    LATENT_DIM,
    assign_personas,
    build_autoencoder,
    describe_persona,
    encode,
    normalise_profiles,
    reconstruction_error,
    train_autoencoder,
)

torch = pytest.importorskip("torch")

SLOTS = [f"h{i}" for i in range(48)]


def test_autoencoder_round_trips_shapes():
    model = build_autoencoder()
    batch = torch.randn(4, 1, 48)
    reconstruction, code = model(batch)
    assert reconstruction.shape == batch.shape
    assert code.shape == (4, LATENT_DIM)


def test_padding_is_circular_not_zero():
    model = build_autoencoder()
    conv = model.encoder[0]

    late_spike = torch.zeros(1, 1, 48)
    late_spike[0, 0, 47] = 1.0
    empty = torch.zeros(1, 1, 48)

    with torch.no_grad():
        response = conv(late_spike)[0, :, 0]
        baseline = conv(empty)[0, :, 0]

    assert not torch.allclose(response, baseline, atol=1e-6), (
        "the first output position ignored a spike at the end of the day, "
        "which means padding is not circular"
    )


def test_zero_padding_would_fail_the_boundary_test():
    conv = torch.nn.Conv1d(1, 8, 5, padding=2)

    late_spike = torch.zeros(1, 1, 48)
    late_spike[0, 0, 47] = 1.0
    empty = torch.zeros(1, 1, 48)

    with torch.no_grad():
        response = conv(late_spike)[0, :, 0]
        baseline = conv(empty)[0, :, 0]

    assert torch.allclose(response, baseline, atol=1e-6)


def test_normalisation_removes_level_and_keeps_shape(synthetic_daily):
    shapes, index = normalise_profiles(synthetic_daily)
    assert shapes.shape[1] == 48
    np.testing.assert_allclose(shapes.sum(axis=1), 1.0, rtol=1e-4)
    assert len(index) == len(shapes)


def test_normalisation_drops_dst_and_vacant_days(synthetic_daily):
    shapes, index = normalise_profiles(synthetic_daily)
    assert not index["flag_dst_day"].any(), "daylight saving days distort the pivoted profile"
    assert (index["kwh_total"] > 0).all(), "vacant days carry no shape"
    assert len(index) < len(synthetic_daily)


def test_two_households_of_the_same_scale_but_different_timing_separate(synthetic_daily):
    shapes, index = normalise_profiles(synthetic_daily)
    doubled = synthetic_daily.copy()
    doubled[SLOTS] = doubled[SLOTS] * 2.0
    doubled["kwh_total"] = doubled[SLOTS].sum(axis=1)
    doubled_shapes, _ = normalise_profiles(doubled)
    np.testing.assert_allclose(shapes, doubled_shapes, rtol=1e-4)


def test_training_reduces_reconstruction_error(synthetic_daily):
    shapes, _ = normalise_profiles(synthetic_daily)
    subset = shapes[:6000]

    untrained = build_autoencoder()
    untrained.eval()
    with torch.no_grad():
        batch = torch.from_numpy(subset[:512]).unsqueeze(1)
        before = float(((untrained(batch)[0] - batch) ** 2).mean())

    model = train_autoencoder(subset, epochs=6, max_train=6000)
    after = float(reconstruction_error(model, subset[:512]).mean())

    assert after < before, f"training did not improve reconstruction ({after} vs {before})"


def test_encoding_is_deterministic(synthetic_daily):
    shapes, _ = normalise_profiles(synthetic_daily)
    model = train_autoencoder(shapes[:4000], epochs=3, max_train=4000)
    first = encode(model, shapes[:256])
    second = encode(model, shapes[:256])
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_personas_partition_every_profile(synthetic_daily):
    shapes, _ = normalise_profiles(synthetic_daily)
    subset = shapes[:5000]
    model = train_autoencoder(subset, epochs=3, max_train=5000)
    codes = encode(model, subset)

    labels, _, summary = assign_personas(codes, n_personas=4, shapes=subset)
    assert len(labels) == len(subset)
    assert set(labels) <= {0, 1, 2, 3}
    assert summary["n_days"].sum() == len(subset)
    assert abs(summary["share_pct"].sum() - 100) < 0.05


def test_persona_names_are_relative_to_the_population():
    hours = np.arange(48) / 2
    population = np.exp(-((hours - 19) ** 2) / 8) + 0.4
    population = population / population.sum()

    overnight = np.exp(-((hours - 2) ** 2) / 8) + 0.4
    overnight = overnight / overnight.sum()

    assert describe_persona(population, population) != describe_persona(overnight, population)
    assert "Overnight" in describe_persona(overnight, population)


def test_reconstruction_error_flags_a_nonsense_day(synthetic_daily):
    shapes, _ = normalise_profiles(synthetic_daily)
    model = train_autoencoder(shapes[:6000], epochs=6, max_train=6000)

    normal = reconstruction_error(model, shapes[:1000])
    spike = np.zeros((1, 48), dtype=np.float32)
    spike[0, 13] = 1.0
    weird = reconstruction_error(model, spike)

    assert weird[0] > np.percentile(normal, 99), "a single-spike day should be flagged"
