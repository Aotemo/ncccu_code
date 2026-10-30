"""Apply the configured Small final-downsample geometry after public loading."""


def apply_final_downsample_stride(model, job) -> None:
    """Change only the existing convolution's stride; rebuild it from job on load."""
    if 'final_downsample_stride' not in job:
        return
    value = job['final_downsample_stride']
    if job.get('model') != 'convnext_small' or type(value) is not int or value != 1:
        raise ValueError('final downsample stride supports only explicit convnext_small stride 1')

    from torch import nn

    target = model.network.features[6][1]
    if (not isinstance(target, nn.Conv2d)
            or target.kernel_size != (2, 2)
            or target.stride != (2, 2)
            or target.padding != (0, 0)
            or target.dilation != (1, 1)):
        raise ValueError('Small final downsample must be the original 2x2 stride-2 convolution')
    target.stride = (value, value)
