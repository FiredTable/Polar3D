
def _test_dwt_fusion():
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '../_test_images')
    img1_path = os.path.join(img_dir, 'rho', '00026.png')
    img2_path = os.path.join(img_dir, 'vis', '00022.png')

    img1 = skimage.io.imread(img1_path)
    img2 = skimage.io.imread(img2_path)
    img1 = img1.astype(np.float64) / 255
    img2 = img2.astype(np.float64) / 255

    images_for_fusion = np.stack((img1, img2), axis=-1)
    dwt_fuser = DWTFuser(images_for_fusion, [0, 1], [1, 0], level=3, wavelet='db1',
                         caf_type='mean', cxf_type='mean', norm=True, visualize=True)
    plt.imshow(dwt_fuser.fused_image)