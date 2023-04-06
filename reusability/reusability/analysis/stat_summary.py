import math
import numpy as np
from collections import defaultdict

from reusability.common import get_nth_from_tuple, standardize_name, is_weight_node, MIN_FLOAT


def shannon_entropy(dist):
    """Calculate shannon entropy from a given discrete probability distribution"""
    eps = 1e-8
    assert abs(sum(dist) - 1.) < eps

    entropy = 0.
    for p in dist:
        if p == 0:
            continue
        entropy -= p * math.log2(p)

    return entropy


def estimate_series(w, d):
    series = 0
    # avoid large numbers by dividing things into w**exp
    exp = d
    div_coeff = w ** exp
    total = (w ** (d - exp - 1) / (w - 1) - 1 / ((w - 1) * div_coeff))
    log2w = math.log2(w)
    for i in range(1, d):
        p = w ** (i - exp - 1) / total
        series += p * log2w * i

    return math.log2(w ** (d - 1) - 1) - series


def estimate_shannon_entropy(w, d, k):
    shannon_estimation = math.log2(k) + math.log2(w) - (math.log2(w - 1) if w != 1 else 0)

    # as d goes to inf the below series should converge to zero
    if d > 1000:
        print("Assuming convergence in entropy")
        return shannon_estimation, shannon_estimation

    series = estimate_series(w, d)

    return shannon_estimation, shannon_estimation + series


def estimate_expected_spread(w, d):
    return math.log2(w) * (((d - 1) * w ** d - d * w ** (d - 1) + 1) / ((w - 1) * (w ** (d - 1) - 1)) - 1)


def estimate_total_spread(w, d, k):
    return math.log2(k) + math.log2(w ** (d - 1) - 1) - math.log2(w - 1)


def estimate_mlp_shannon(w, d):
    k = w * (w + 1)
    inf_depth_shannon, current_depth_shannon = estimate_shannon_entropy(w, d, k)
    num_params = k * d
    expected_spread = estimate_expected_spread(w, d)
    total_spread = estimate_total_spread(w, d, k)
    return inf_depth_shannon, current_depth_shannon, num_params, expected_spread, total_spread


def estimate_cnn_shannon(num_channels, d, kernel_size):
    w = num_channels * kernel_size ** 2
    k = num_channels * (w + 1)
    inf_depth_shannon, current_depth_shannon = estimate_shannon_entropy(w, d, k)
    num_params = k * d
    expected_spread = estimate_expected_spread(w, d)
    total_spread = estimate_total_spread(w, d, k)
    return inf_depth_shannon, current_depth_shannon, num_params, expected_spread, total_spread


def divide_by_max(counts, max_value=None):
    if max_value is None:
        max_value = max(counts)

    counts = [c / max_value for c in counts]
    return counts


def convert_to_relative_freq(counts, max_value=None):
    counts = divide_by_max(counts, max_value=max_value)
    total = sum(counts)
    relative_freqs = [c / total for c in counts]

    return [p if p != 0 else MIN_FLOAT for p in relative_freqs]


def get_weight_items(frequency_dict):
    weight_items = []

    for k, v in frequency_dict.items():
        if is_weight_node(k):
            if v < 1:
                print(f"WARNING! Skipping weight item {k} since its count is {v}")
                continue
            weight_items.append((k, v))

    return weight_items


def get_weight_and_input_count_analysis(frequency_dict, ignore_suffix):
    weight_items = get_weight_items(frequency_dict)

    print("WARNING! Make sure all input node names end with the suffix: 'bid0'!")
    feature_items = [(k, v) for k, v in frequency_dict.items() if 'bid0' in k and v >= 1]

    if len(feature_items) == 0:
        raise ValueError("No input features are found in stats")

    return analyze_counts(weight_items, feature_items, ignore_suffix=ignore_suffix)


def get_weight_count_analysis(frequency_dict, ignore_suffix):
    weight_items = get_weight_items(frequency_dict)
    return analyze_counts(weight_items, None, ignore_suffix=ignore_suffix)


def analyze_counts(weight_items, feature_items, ignore_suffix):
    """Get model level quantities using the number of contexts for learnable parameters"""
    counts = defaultdict(int)

    for k, v in weight_items:
        name = standardize_name(k, ignore_suffix)
        counts[name] += v

    frequencies = list(counts.values()) + (get_nth_from_tuple(feature_items, 1) if feature_items is not None else [])

    max_freq = max(frequencies)
    probabilities = convert_to_relative_freq(frequencies, max_value=max_freq)
    spreads = [math.log2(f) for f in frequencies]

    total_bits = sum(spreads)

    # total surprisal
    # for very small probabilities this can give incorrect results
    # due to floating point limitations. There is an alternative way to calculate it.
    total_surpr = sum([-math.log2(p) for p in probabilities])

    spreads = [p * spread for p, spread in zip(probabilities, spreads)]
    shannon = shannon_entropy(probabilities)

    # expected spread
    exp_spread = sum(spreads)
    frequencies = divide_by_max(frequencies, max_value=max_freq)

    # log(N_c)
    log_n = math.log2(max_freq) + math.log2(sum(frequencies))

    # Here h = log(N_c) - D_kl(p_x(x)||p_u(x))
    num_weights, num_params = len(weight_items), len(counts)
    sh_exp_spread = shannon + exp_spread
    num_f = len(feature_items) if feature_items is not None else 0
    return shannon, exp_spread, log_n, total_bits, total_surpr, sh_exp_spread, num_weights, num_f, num_params, max_freq


def get_performance_estimations_from_summary(summary, input_img_size=None, num_input_channels=None,
                                             num_output_nodes=None):
    """Get surprisal and expected spread based estimation of performance

    To estimate model performance with larger image inputs,
    it is possible to use an analysis from the same model with a smaller image input.
    This gives a lower bound for performance estimation.
    Set input_img_size for a larger image size for such a scenario. """
    import math

    if not input_img_size:
        input_img_size = summary['image_width']
        assert summary['image_width'] == summary['image_height']

    if not input_img_size:
        if summary['name']:
            summary['default_size'] = int(summary['name'].split('_')[-1])
            assert summary['default_size'] == summary['image_width']
            input_img_size = summary['default_size']
        else:
            raise ValueError("Input size is not specified")

    if not num_input_channels:
        num_input_channels = summary['num_input_channels']

    if not num_output_nodes:
        num_output_nodes = summary['num_output_nodes']

    n_i = num_input_channels * input_img_size ** 2

    n_o = num_output_nodes
    n_i_over_g = n_i / (n_i + n_o + summary['num_weight_nodes'])

    est = n_i_over_g * summary['total_surprisal']
    surprisal_based_estimation = math.log2(est)

    est = n_i_over_g * (summary['expected_spread'] + 1) * summary['num_params']
    exp_spread_based_estimation = math.log2(est)

    return surprisal_based_estimation, exp_spread_based_estimation


def summarize_stats(count_stats, dag=None, ignore_suffix=False, dag_type=None,
                    kernel_size=None, image_width=None, image_height=None,
                    width=None, has_bias=None, depth=None, include_input_nodes=False,
                    name=None, num_input_channels=3, num_output_nodes=1000):
    """Summarize model level attributes and quantities derived from relative frequencies"""
    if not include_input_nodes:
        analysis = get_weight_count_analysis(count_stats, ignore_suffix)
    else:
        analysis = get_weight_and_input_count_analysis(count_stats, ignore_suffix)
    shannon, exp_spread, log_n, total_bits, total_surpr, sh_exp_spread, num_ws, num_f, num_params, max_freq = analysis
    max_entropy = math.log2(num_ws)

    # Alternative way to calculate total_surprisal to avoid the min floating point issue.
    # This approach is better than the total_surpr where the smallest probability is clipped at 1e-323,
    # which leads to the -log probabilities to be smaller. For larger models this would diverge from the actual value.
    # Small probs pose no problem for shannon as plogp goes to zero as p goes to zero.
    # plogp is never too different than p, and can be observed that:
    # shannon + expected_spread always exactly matches total_possible_spread (i.e. log N_c).
    total_surprisal = log_n * num_params - total_bits

    width = int(np.mean(np.array(dag.attributes['channels']))) if dag else width
    dag_type = dag.attributes['dag_type'] if dag else dag_type
    kernel_size = dag.attributes['kernel_size'] if dag else kernel_size
    image_width = dag.attributes['image_width'] if dag else image_width
    image_height = dag.attributes['image_height'] if dag else image_height
    has_bias = dag.attributes['has_bias'] if dag else has_bias
    depth = dag.attributes['depth'] if dag else depth

    if dag_type == 'cnn':
        estimation = estimate_cnn_shannon(width, depth, kernel_size)
        inf_depth_estimation, current_estimation, num_param_estimation, spread_estimation, total_spread_estimation = estimation
    elif dag_type == 'mlp':
        estimation = estimate_mlp_shannon(width, depth)
        inf_depth_estimation, current_estimation, num_param_estimation, spread_estimation, total_spread_estimation = estimation
    else:
        inf_depth_estimation = current_estimation = num_param_estimation = spread_estimation = total_spread_estimation = None

    summary = {
        'shannon': shannon,
        'shannon_estimation': current_estimation,
        'inf_depth_shannon_approximation': inf_depth_estimation,
        'expected_spread': exp_spread,
        'total_possible_spread': log_n,
        'total_bits': total_bits,
        'total_surprisal_from_probs': total_surpr,
        'total_surprisal': total_surprisal,
        'num_inp_features': num_f,
        'expected_spread_estimation': spread_estimation,
        'total_spread_estimation': total_spread_estimation,
        'shannon+expected_spread': sh_exp_spread,
        'expected_spread/total_possible_spread': exp_spread / log_n,
        'per_param_expected_spread': exp_spread / num_params,
        'max_nonshared_entropy': max_entropy,
        'max-max_shared': max_entropy - math.log2(num_params),
        'max-shannon': max_entropy - shannon,
        'dag_depth': depth,
        'dag_width': width,
        'dag_type': dag_type,
        'image_width': image_width,
        'image_height': image_height,
        'num_input_channels': num_input_channels,
        'num_output_nodes': num_output_nodes,
        'has_bias': has_bias,
        'num_weight_nodes': num_ws,
        'num_params': num_params,
        'num_params_est': num_param_estimation,
        'max_freq': max_freq,
        'ignore_suffix': ignore_suffix,
        'name': name,
        'default_size': depth if depth == width else None,
    }

    total_surprisal_based_perf_est, exp_spread_based_perf_est = get_performance_estimations_from_summary(summary)

    summary['total_surprisal_based_perf_est'] = total_surprisal_based_perf_est
    summary['exp_spread_based_perf_est'] = exp_spread_based_perf_est

    return summary



