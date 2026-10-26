import argparse
import importlib
import random
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import t as student_t
from NET import MGCD
ROOT = Path(__file__).resolve().parent
BASE_SEED = 2027
REPEAT_COUNT = 5
NUM_SUBJECTS = 3
NUM_TRAIN_SAMPLES = 8
NUM_TEST_SAMPLES = 2
TIME_SERIES_LENGTH = 200

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dgp(generator_root, n_nodes):
    """Load the official NonLinearVAR simulator and its DGP parameters."""
    generator_root = generator_root.resolve()
    config_path = generator_root / "configs" / "NonLinearVAR.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"NonLinearVAR config not found: {config_path}")

    root_string = str(generator_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    params = OmegaConf.to_container(
        OmegaConf.load(config_path).DGP, resolve=True
    )
    params["num_subjects"] = NUM_SUBJECTS
    params["num_nodes"] = n_nodes
    simulator_module = importlib.import_module("generator.simulators")
    simulator_class = getattr(simulator_module, f"sim{params['dgp_str']}")
    return simulator_class, params


def _simulate_one_trajectory(simulator, gc, sample_length, burn_in=200):
    """Simulate OneBanded2VAR with an explicitly initialized starting state.

    This follows ``simOneBanded2VAR._sim_one_trajectory``.  The upstream code
    allocates ``x`` with ``np.empty`` and never sets ``x[0]``; using zeros here
    prevents undefined memory from occasionally producing infinities/NaNs.
    """
    total_length = sample_length + burn_in
    noise = simulator.sigma_obs * np.random.normal(
        size=(total_length, simulator.n_nodes)
    )
    x = np.zeros((total_length, simulator.n_nodes), dtype=np.float64)
    for time_index in range(1, total_length):
        previous = x[time_index - 1]
        for node in range(simulator.n_nodes):
            if node == 0:
                signal = 0.4 * previous[node] - 0.5 * previous[node + 1]
            elif node == simulator.n_nodes - 1:
                signal = 0.4 * previous[node] - 0.5 * previous[node - 1]
            elif node == 1 or node == simulator.n_nodes - 2:
                signal = (
                    np.sin(previous[node] * previous[node + 1])
                    + 0.5 * np.cos(previous[node] + previous[node - 1])
                )
            else:
                source_left, self_node, source_right = np.flatnonzero(gc[node])
                if self_node != node:
                    raise ValueError(f"Invalid self edge in GC row {node}")
                signal = (
                    0.25 * previous[node]
                    + np.sin(previous[source_left] * previous[source_right])
                    + np.cos(previous[source_left] + previous[source_right])
                )
            x[time_index, node] = signal + noise[time_index, node]
    return x[burn_in:]


def _simulate_independent_samples(simulator, sample_count, sample_length):
    """Restart the 200-step burn-in for every independent sample."""
    samples = [[] for _ in range(NUM_SUBJECTS)]
    for _ in range(sample_count):
        for subject in range(NUM_SUBJECTS):
            trajectory = _simulate_one_trajectory(
                simulator,
                simulator.VAR_params["GCs"][subject],
                sample_length,
            )
            samples[subject].append(trajectory)
    result = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonlinear VAR generator produced NaN or infinity")
    return result


def _standardize_from_training(train_samples, test_samples):
    # Statistics are fitted on training data only, independently per subject/node.
    mean = train_samples.mean(axis=(1, 2), keepdims=True)
    std = train_samples.std(axis=(1, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_samples - mean) / std, (test_samples - mean) / std


def _to_mscd_batches(samples, batch_size, input_step, pred_step, drop_last):
    """Window each 200-point trajectory without crossing sample boundaries."""
    subject_count, sample_count, time_count, node_count = samples.shape
    windows_per_sample = time_count - input_step - pred_step + 1
    if windows_per_sample < 1:
        raise ValueError(
            f"A {time_count}-point sample is too short for input_step="
            f"{input_step} and pred_step={pred_step}"
        )

    all_x = [[] for _ in range(subject_count)]
    all_y = [[] for _ in range(subject_count)]
    all_t = []
    for sample_id in range(sample_count):
        for offset in range(windows_per_sample):
            target_start = offset + input_step
            target_end = target_start + pred_step
            for subject in range(subject_count):
                trajectory = samples[subject, sample_id]
                all_x[subject].append(
                    trajectory[offset:target_start].T[..., None]
                )
                all_y[subject].append(
                    trajectory[target_start:target_end].T[..., None]
                )
            all_t.append(sample_id * time_count + target_start)

    window_count = len(all_t)
    shared_order = np.random.permutation(window_count)
    all_t = np.asarray(all_t, dtype=np.int64)[shared_order]
    subject_batches = []
    for subject in range(subject_count):
        x = torch.from_numpy(
            np.asarray(all_x[subject], dtype=np.float32)[shared_order])
        y = torch.from_numpy(
            np.asarray(all_y[subject], dtype=np.float32)[shared_order])
        t = torch.from_numpy(all_t.copy())
        batches = []
        for start in range(0, window_count, batch_size):
            end = min(start + batch_size, window_count)
            if drop_last and end - start < batch_size:
                break
            batches.append((x[start:end], y[start:end], t[start:end]))
        subject_batches.append(batches)
    return subject_batches


def generate_nonlinvar_dataset(simulator,sample_length,input_step,pred_step,train_batch_size,test_batch_size,):
    """Generate fresh sequences from one fixed three-subject causal system."""
    train_samples = _simulate_independent_samples(
        simulator, NUM_TRAIN_SAMPLES, sample_length
    )
    test_samples = _simulate_independent_samples(
        simulator, NUM_TEST_SAMPLES, sample_length
    )
    train_samples, test_samples = _standardize_from_training(
        train_samples, test_samples)

    # The generator stores GC[target, source]; MSCD evaluates [source, target].
    true_causal = np.stack(simulator.VAR_params["GCs"], axis=0).transpose(0, 2, 1)
    true_common = np.asarray(simulator.VAR_params["GCbar"]).T
    return (_to_mscd_batches(train_samples, train_batch_size, input_step, pred_step, drop_last=True),
        _to_mscd_batches(test_samples, test_batch_size, input_step, pred_step, drop_last=False),
        true_causal,true_common,)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--generator-root", type=Path, default=ROOT / "data" / "nonlinearvar",help="Generator root (defaults to the bundled NonLinearVAR simulator)",)
    parser.add_argument("--data-only", action="store_true",help="Generate and validate datasets without training MGCD")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    opt = OmegaConf.load(ROOT / "opt" / "NonLinear_best.yaml")
    data_opt = opt.data
    model_opt = opt.sota.ST_MSCD
    data_opt.n_sub = NUM_SUBJECTS
    model_opt.n_sub = NUM_SUBJECTS
    model_opt.batch_size = data_opt.train_batch_size
    model_opt.input_step = data_opt.input_step
    if args.epochs is not None:
        model_opt.total_epoch = args.epochs

    simulator_class, dgp_params = load_dgp(args.generator_root, data_opt.n_nodes)
    set_seed(args.seed)
    simulator = simulator_class(dgp_params)
    simulator.gen_VAR_params()
    print(
        f"Dataset per repeat: {NUM_SUBJECTS} subjects/classes, "
        f"{NUM_TRAIN_SAMPLES} independent train samples and "
        f"{NUM_TEST_SAMPLES} independent test samples per subject, "
        f"{TIME_SERIES_LENGTH} time points per sample; "
        f"MGCD window={data_opt.input_step}+{data_opt.pred_step}")
    print(f"Causal graphs generated once with seed={args.seed} and fixed for all repeats")

    metric_names = MGCD.METRIC_OUTPUT_NAMES
    metric_values = {name: [] for name in metric_names}
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        set_seed(seed)
        print(f"repeat={repeat + 1}/{args.repeats}, seed={seed}")
        train_data, test_data, true_causal, true_common = (
            generate_nonlinvar_dataset(simulator,TIME_SERIES_LENGTH,data_opt.input_step,data_opt.pred_step,data_opt.train_batch_size,data_opt.test_batch_size,
            )
        )
        print(
            f"train_batches={len(train_data[0])}, "
            f"train_batch={tuple(train_data[0][0][0].shape)}, "
            f"test_batches={len(test_data[0])}, "
            f"test_batch={tuple(test_data[0][0][0].shape)}, "
            f"graphs={true_causal.shape}"
        )
        if args.data_only:
            continue

        true_specific = MGCD.build_specific_ground_truth(true_causal, true_common)
        metrics = MGCD.main(train_data,test_data,true_causal,true_common,model_opt, true_specific=true_specific,)
        for name, value in zip(metric_names, metrics):
            metric_values[name].append(value)

    if args.data_only:
        print("Data generation validation completed; MGCD training was skipped.")
        return

    summary_lines = [
        f"All {args.repeats} deterministic runs",
        f"mean +/- 95% CI (Student's t, n={args.repeats})",
        (
            f"Fixed graphs: {NUM_SUBJECTS} subjects; each run: "
            f"{NUM_TRAIN_SAMPLES} train + {NUM_TEST_SAMPLES} test samples/subject"
        ),
        "",
    ]
    t_critical = (
        student_t.ppf(0.975, df=args.repeats - 1)
        if args.repeats > 1 else 0.0
    )
    for name in metric_names:
        values = np.asarray(metric_values[name], dtype=float)
        mean = float(values.mean())
        margin = (
            float(t_critical * values.std(ddof=1) / np.sqrt(args.repeats))
            if args.repeats > 1 else 0.0
        )
        print(f"{name}: {values.tolist()}")
        print(f"{name}: {mean:.6f} +/- {margin:.6f}")
        summary_lines.append(f"{name}: {mean:.6f} +/- {margin:.6f}")

    output_path = ROOT / "MGCD_nonlinvar_independent_summary.txt"
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
