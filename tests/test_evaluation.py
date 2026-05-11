from evaluation.evaluate import compute_rouge_l_scores, perplexity_from_loss


def test_compute_rouge_l_scores() -> None:
    score = compute_rouge_l_scores(["The cat sat on the mat."], ["The cat sat on the mat."])
    assert abs(score - 1.0) < 1e-6

    score_partial = compute_rouge_l_scores(["The cat sat."], ["The cat sat on the mat."])
    assert 0.0 < score_partial < 1.0


def test_perplexity_from_loss() -> None:
    assert perplexity_from_loss(0.0) == 1.0
    assert abs(perplexity_from_loss(0.6931471805599453) - 2.0) < 1e-6
