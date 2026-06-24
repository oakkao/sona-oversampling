from .config import DATASET_NAMES, BEST_PARAM_CSV, RESULTS_CSV
from .data import load_datasets
from .tuning import run_tuning
from .experiment import load_best_params, run_experiments


def main():
    print("=" * 60)
    print("SONA Benchmark Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading datasets...")
    train_test_list = load_datasets(DATASET_NAMES)

    print("\n[2/4] Parameter tuning...")
    run_tuning(train_test_list, BEST_PARAM_CSV)

    print("\n[3/4] Loading tuned parameters...")
    param_lookup = load_best_params(BEST_PARAM_CSV)

    print("\n[4/4] Running experiments...")
    run_experiments(train_test_list, param_lookup, RESULTS_CSV)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Parameters: {BEST_PARAM_CSV}")
    print(f"  Results:    {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
