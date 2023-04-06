from reusability import common
from reusability.analysis.stat_summary import summarize_stats
from reusability.analysis.estimation import (
    add_layer_stats,
    add_conv_block_steps,
    add_identity_block_steps,
    add_global_pooling_stats,
    add_local_pooling_stats
)
from reusability.analysis.planner import push_step, make_feature_names, collect_stats_from_plan
from reusability.common import get_uuid


def add_resnet50_steps(
        plan,
        default_size,
        include_top=True,
        classes=1000,
        base_width=64,
        use_bias_instead_of_bn=False,
        source=None):
    """Add steps to calculate the aggregated number of contexts for ResNet-50"""
    if plan is None:
        plan = []

    if source is None:
        # does not change stats count, no need to add to plan
        input_shape = (default_size, default_size, 3)
        img_input = make_feature_names(name='input',
                                       shape=input_shape,
                                       prefix='z0',
                                       suffix='bid0')
        source = img_input

    x = source
    # skip any preprocessing layers
    # already assuming zero padding
    stem_filters = base_width
    x = push_step(plan,
                  (add_layer_stats, dict(
                      input_filters=x.shape[-1],
                      output_filters=stem_filters,
                      kernel_size=7,
                      strides=2,
                      padding="same",
                      has_bias=use_bias_instead_of_bn,
                      has_bn=not use_bias_instead_of_bn,
                      name=f"k7s2sc3tc{stem_filters}",
                      suffix="conv1",
                      layer_type="conv")),
                  x)
    # skip activation and zero padding
    x = push_step(plan,
                  (add_local_pooling_stats,
                   {"layer_type": "local_pool",
                    "name": f"conv1_max_pool_{get_uuid()}",
                    "kernel_size": 3,
                    "strides": 2}),
                  x)

    # push multiple steps inside
    x = add_conv_block_steps(
        plan=plan,
        source=x,
        target=None,
        input_filters=x.shape[-1],
        output_filters=base_width * 4,
        filters=[base_width, base_width, base_width * 4],
        kernel_size=3,
        strides=1,
        stage=2,
        block='a',
        name=f"k3s1sc{x.shape[-1]}tc{base_width * 4}",
        suffix='s2a',
        has_bias=use_bias_instead_of_bn, )

    for block in ['b', 'c']:
        x = add_identity_block_steps(
            plan=plan,
            source=x,
            target=None,
            input_filters=x.shape[-1],
            output_filters=base_width * 4,
            filters=[base_width, base_width, base_width * 4],
            kernel_size=3,
            strides=1,
            block=block,
            stage=2,
            name=f"k3s1sc{x.shape[-1]}tc{base_width * 4}",
            suffix=f's2{block}',
            has_bias=use_bias_instead_of_bn, )

    x = add_conv_block_steps(
        plan=plan,
        source=x,
        target=None,
        input_filters=x.shape[-1],
        output_filters=base_width * 8,
        filters=[base_width * 2, base_width * 2, base_width * 8],
        kernel_size=3,
        strides=2,
        block='a',
        stage=3,
        name=f"k3s2sc{x.shape[-1]}tc{base_width * 8}",
        suffix='s3a',
        has_bias=use_bias_instead_of_bn, )

    for block in ['b', 'c', 'd']:
        x = add_identity_block_steps(
            plan=plan,
            source=x,
            target=None,
            input_filters=x.shape[-1],
            output_filters=base_width * 8,
            filters=[base_width * 2, base_width * 2, base_width * 8],
            kernel_size=3,
            strides=1,
            block=block,
            stage=3,
            name=f"k3s1sc{x.shape[-1]}tc{base_width * 8}",
            suffix=f's3{block}',
            has_bias=use_bias_instead_of_bn, )

    x = add_conv_block_steps(
        plan=plan,
        source=x,
        target=None,
        input_filters=x.shape[-1],
        output_filters=base_width * 16,
        filters=[base_width * 4, base_width * 4, base_width * 16],
        kernel_size=3,
        strides=2,
        block='a',
        stage=4,
        name=f"k3s2sc{x.shape[-1]}tc{base_width * 16}",
        suffix='s4a',
        layer_type="conv",
        has_bias=use_bias_instead_of_bn, )

    for block in ['b', 'c', 'd', 'e', 'f']:
        x = add_identity_block_steps(
            plan=plan,
            source=x,
            target=None,
            input_filters=x.shape[-1],
            output_filters=base_width * 16,
            filters=[base_width * 4, base_width * 4, base_width * 16],
            kernel_size=3,
            strides=1,
            block=block,
            stage=4,
            name=f"k3s1sc{x.shape[-1]}tc{base_width * 16}",
            suffix=f's4{block}',
            layer_type="conv",
            has_bias=use_bias_instead_of_bn, )

    x = add_conv_block_steps(
        plan=plan,
        source=x,
        target=None,
        input_filters=x.shape[-1],
        output_filters=base_width * 32,
        filters=[base_width * 8, base_width * 8, base_width * 32],
        kernel_size=3,
        strides=2,
        block='a',
        stage=5,
        name=f"k3s2sc{x.shape[-1]}tc{base_width * 32}",
        suffix='s5a',
        layer_type="conv",
        has_bias=use_bias_instead_of_bn, )

    for block in ['b', 'c']:
        x = add_identity_block_steps(
            plan=plan,
            source=x,
            target=None,
            input_filters=x.shape[-1],
            output_filters=base_width * 32,
            filters=[base_width * 8, base_width * 8, base_width * 32],
            kernel_size=3,
            strides=1,
            block=block,
            stage=5,
            name=f"k3s1sc{x.shape[-1]}tc{base_width * 32}",
            suffix=f"s5{block}",
            layer_type="conv",
            has_bias=use_bias_instead_of_bn, )

    if include_top:
        x = push_step(plan,
                      (add_global_pooling_stats,
                       {"layer_type": "global_pool",
                        "name": f"final_global_pool_{get_uuid()}"}),
                      x)
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


def estimate_resnet50_stats(
        stats=None,
        **kwargs):
    """Estimate the aggregated number of contexts for all learnable parameters of ResNet-50"""
    plan = []
    common.TOTAL_CONV_LAYERS = 0
    add_resnet50_steps(plan, **kwargs)

    num_blocks = common.TOTAL_CONV_LAYERS
    print(f"Running estimate_resnet_stats with # conv blocks: {common.TOTAL_CONV_LAYERS}")
    stats = collect_stats_from_plan(plan, stats=stats)
    print(f"Finished estimate_resnet_stats with # conv layers: {common.TOTAL_CONV_LAYERS - num_blocks + 1}")
    common.TOTAL_CONV_LAYERS = 0
    return stats


def get_resnet50_stat_summary(ignore_suffix=False, **kwargs):
    """Analyze the computational graph of ResNet-50 and return model level quantities"""
    depth = 50
    base_width = kwargs.get('base_width', 64)
    width = (base_width * 32 - base_width) // 2
    return summarize_stats(estimate_resnet50_stats(**kwargs), dag=None,
                           ignore_suffix=ignore_suffix, dag_type='cnn',
                           kernel_size=3, width=width,
                           image_width=kwargs.get('default_size', None),
                           image_height=kwargs.get('default_size', None),
                           has_bias=False, depth=depth)
