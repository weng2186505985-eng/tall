import numpy as np
from deepfake_detection.utils.metrics import calculate_eer, calculate_auc

def test_eer_logic():
    print("Testing EER and AUC calculation logic...")
    
    # Perfect predictions
    y_true_perfect = [0, 0, 1, 1]
    y_scores_perfect = [0.1, 0.2, 0.8, 0.9]
    eer_perfect = calculate_eer(y_true_perfect, y_scores_perfect)
    auc_perfect = calculate_auc(y_true_perfect, y_scores_perfect)
    print(f"Perfect - EER: {eer_perfect:.4f}, AUC: {auc_perfect:.4f}")
    
    # Overlapping predictions
    y_true_mixed = [0, 0, 1, 1]
    y_scores_mixed = [0.4, 0.6, 0.5, 0.7]
    eer_mixed = calculate_eer(y_true_mixed, y_scores_mixed)
    auc_mixed = calculate_auc(y_true_mixed, y_scores_mixed)
    print(f"Mixed - EER: {eer_mixed:.4f}, AUC: {auc_mixed:.4f}")

    assert eer_perfect < 0.01, f"Expected near zero EER, got {eer_perfect}"
    assert auc_perfect > 0.99, f"Expected near 1.0 AUC, got {auc_perfect}"
    assert eer_mixed > 0.1, f"Expected higher EER for mixed data, got {eer_mixed}"
    
    print("\nEER Logic Verification PASSED!")

if __name__ == "__main__":
    try:
        test_eer_logic()
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
