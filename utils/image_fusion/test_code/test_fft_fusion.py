
def _test_fft_fusion():
    import MyTools.polarization_utils.polarization_analyser as pu
    from MyTools.common import list2bhwc

    dataset_path = r'..\_test_images'
    image_src_path = r'dot\full_polarization\2024-12-25_16-06-10'

    dataset_manager = pu.PolarDatasetManager(dataset_type=pu.PolarDatasetManager.DTYPE_DOT_FP)
    dataset_manager.import_source(dataset_path, image_src_path, set_save=False)

    polar_analyser = pu.PolarizationAnalyser(dataset_manager)
    polar_analyser.calc_stokes(visualize=False)
    polar_analyser.calc_polar_features(visualize=True)

    iun = polar_analyser.iun
    rho = polar_analyser.rho
    phi = polar_analyser.phi

    images = list2bhwc([iun, rho, phi])
    fuser = FFTFuser()
    fused_image = fuser.fusion(images, norm_mode=None, visualize=True)


def main():
    _test_fft_fusion()


if __name__ == '__main__':
    main()
