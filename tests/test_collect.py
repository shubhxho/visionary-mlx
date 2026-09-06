from visionary_mlx.data import ClipLoader, collect_rollouts


def test_collect_and_sample():
    ds = collect_rollouts("balls", num_episodes=4, episode_len=8, size=32, seed=0, parallel=4)
    assert ds["video"].shape == (4, 8, 32, 32, 3)
    assert ds["actions"].shape == (4, 8)
    loader = ClipLoader(ds, seq_len=4, batch_size=2, seed=0)
    b = loader.sample()
    assert b["video"].shape == (2, 4, 32, 32, 3)
    assert b["video"].max() <= 1.0 + 1e-5
