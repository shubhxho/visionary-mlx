from visionary_mlx.benchmark import BenchmarkResult, leaderboard_markdown, write_leaderboard


def test_leaderboard_ranks_by_psnr_then_mae(tmp_path):
    results = [BenchmarkResult("smaller", 22.0, 0.11), BenchmarkResult("champion", 25.0, 0.09, 20.0, 0.14)]
    text = leaderboard_markdown(results)
    assert text.index("champion") < text.index("smaller")
    json_path, markdown_path = write_leaderboard(results, tmp_path)
    assert json_path.exists()
    assert markdown_path.read_text() == text
