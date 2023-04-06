# Graph construction follows the official tf.keras implementation
# Please refer to https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet_v2.py
# (tf.keras.applications.efficientnet_v2)


import copy
from reusability.examples.efficientnetv2_utils import (
    DEFAULT_BLOCKS_ARGS, round_filters,
    round_repeats
)
from reusability import common
from reusability.analysis.stat_summary import summarize_stats
from reusability.analysis.estimation import (
    add_layer_stats,
    add_mb_conv_block_steps,
    add_fused_mb_conv_block_steps,
    add_global_pooling_stats
)
from reusability.analysis.planner import push_step, make_feature_names, collect_stats_from_plan
from reusability.common import get_uuid


def add_efficientnet_v2_steps(
        plan,
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        blocks_args="default",
        model_name="efficientnetv2",
        include_top=True,
        classes=1000,
        include_preprocessing=True,
        use_bias_instead_of_bn=False,
        source=None,
        **_
):
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    if plan is None:
        plan = []

    x = source
    if source is None:
        # STEP n (last)
        # does not change stats count, no need to add to plan
        input_shape = (default_size, default_size, 3)
        img_input = make_feature_names(name='input',
                                       shape=input_shape,
                                       prefix='z0',
                                       suffix='bid0')
        x = img_input

    # skip the preprocessing layers
    if include_preprocessing:
        # x = rescaling (and normalization only if b model)
        pass

    # Build stem
    # noinspection PyTypeChecker
    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )

    # STEP n - 1, needs to be able to create img_input names
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=stem_filters,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      has_bias=use_bias_instead_of_bn,
                      has_bn=not use_bias_instead_of_bn,
                      name=f"k3s2sc3tc{stem_filters}",
                      suffix="stem_conv",
                      layer_type="conv")),
                  x)

    # skip activation layer

    blocks_args = copy.deepcopy(blocks_args)
    b = 0

    # total depth from repeated blocks
    blocks = sum(args["num_repeat"] for args in blocks_args)

    in_channels = stem_filters
    for i, args in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        in_channels = args["output_filters"]

        # Determine which conv type to use:
        block = {0: add_mb_conv_block_steps,
                 1: add_fused_mb_conv_block_steps}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
        )

        for j in range(repeats):
            # The first block needs to take care of stride and filter size
            # increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            # STEP k
            name = f"k{args['kernel_size']}s{args['strides']}#sc{args['input_filters']}#tc{args['output_filters']}"
            # push multiple steps
            x = block(
                plan=plan,
                source=x,
                target=None,

                # ignore activation & bn_momentum, ignore survival prob args
                # survival_probability=drop_connect_rate * b / float(blocks),
                name=name,
                suffix=f"bid{i + 1}_r{chr(j + 97)}",
                layer_type="conv",
                has_bias=use_bias_instead_of_bn,
                **args, )
            b += 1

    # Build top
    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )

    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=top_filters,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      has_bias=use_bias_instead_of_bn,
                      has_bn=not use_bias_instead_of_bn,
                      name=f"k1s1sc{in_channels}tc{top_filters}",
                      suffix=f"bid{blocks}_end",
                      layer_type='conv')),
                  x)
    # skip activation layer

    if include_top:
        x = push_step(plan,
                      (add_global_pooling_stats,
                       {"layer_type": "global_pool",
                        "name": f"final_global_pool_{get_uuid()}"}),
                      x)

        # skip dropout
        if dropout_rate > 0:
            pass

        # skip validate_activation

        x = push_step(plan,
                      (add_layer_stats, dict(
                          output_filters=classes,
                          has_bias=True,
                          has_bn=False,
                          name="predictions",
                          layer_type="dense",
                          is_output_layer=True,
                      )),
                      x)
    else:
        x = push_step(plan,
                      (add_global_pooling_stats,
                       {"layer_type": "global_pool",
                        "name": f"final_global_pool_{get_uuid()}",
                        "is_output_layer": True}),
                      x)
    return x


# noinspection DuplicatedCode,PyUnusedLocal
def estimate_efficientnet_v2_stats(
        stats=None,
        with_backprop=False,
        **kwargs
):
    """Approximation of efficientnet_v2 graph stats

    Avoids building the graph and instead directly computes the stats
    """
    plan = []
    common.TOTAL_CONV_LAYERS = 0
    add_efficientnet_v2_steps(plan, **kwargs)

    num_blocks = common.TOTAL_CONV_LAYERS
    print(f"Running estimate_efficientnet_v2 with # conv blocks: {common.TOTAL_CONV_LAYERS}")

    stats = collect_stats_from_plan(plan, stats=stats)
    print(f"Finished estimate_efficientnet_v2")
    common.TOTAL_CONV_LAYERS = 0
    return stats


######################################################################
# ###################   HELPER FUNCTIONS   ######################### #
######################################################################
def estimate_depth_and_width_from_model_name(model_name):
    blocks_args = DEFAULT_BLOCKS_ARGS[model_name]
    depth = 4
    total_width = blocks_args[0]['input_filters']

    for item in blocks_args:
        total_width += item['output_filters']
        is_expanding = item['expand_ratio'] != 1
        has_se = 0 < item['se_ratio'] <= 1

        if item['conv_type'] == 0:
            depth += (int(is_expanding) + 1 + int(has_se) * 4) * item['num_repeat']
        else:
            depth += (int(is_expanding) + 1 + int(has_se) * 3 + 1) * item['num_repeat']

    return depth, int(total_width / len(blocks_args))


def get_efficientnetv2_stat_summary(ignore_suffix=False, **kwargs):
    """Analyze the computational graph of EfficientNetv2 and return model level quantities"""
    depth, width = estimate_depth_and_width_from_model_name(kwargs.get('model_name'))
    return summarize_stats(estimate_efficientnet_v2_stats(**kwargs), dag=None,
                           ignore_suffix=ignore_suffix, dag_type='cnn',
                           kernel_size=3, width=width,
                           image_width=kwargs.get('default_size', None),
                           image_height=kwargs.get('default_size', None),
                           has_bias=False, depth=depth)


def estimate_efficientnet_v2b0_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        model_name="efficientnetv2-b0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2b1_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        model_name="efficientnetv2-b1",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2b2_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        model_name="efficientnetv2-b2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2b3_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=300,
        model_name="efficientnetv2-b3",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2s_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=384,
        model_name="efficientnetv2-s",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2m_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-m",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )


def estimate_efficientnet_v2l_stats(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        include_preprocessing=True,
):
    return estimate_efficientnet_v2_stats(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-l",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
    )
