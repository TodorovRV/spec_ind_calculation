import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import sys
import os
from tempfile import TemporaryDirectory
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
from astropy.modeling import models, fitting
from spydiff import (clean_difmap, find_nw_beam, create_clean_image_from_fits_file,
            create_difmap_file_from_single_component, filter_difmap_CC_model_by_r,
            join_difmap_models, modelfit_difmap, get_uvrange, convert_difmap_model_file_to_CCFITS)
from uv_data import UVData
from image import plot as iplot
from utils import find_bbox, find_image_std
from alpha_utils import rebase_CLEAN_model


def load_data(sourse, date, freq):
        freqs_dict = {8.1:'x', 8.4:'y', 12.1:'j'}
        data_dir_multi = '/mnt/jet1/yyk/VLBI/2cmVLBA/data/multifreq'
        if freq == 15.4:
            return '/mnt/jet1/yyk/VLBI/2cmVLBA/data/{}/{}/{}.u.{}.uvf'.format(sourse, date, sourse, date)
        else:
            return '{}/{}.{}.{}.uvf'.format(data_dir_multi, sourse, freqs_dict[freq], date)


def create_individual_script(sourse, date, freqs, base_dir, deep_clean=False, outdir='./'):
    uv_min = 0
    uv_max = np.inf
    for freq in freqs:
        uv_min_, uv_max_ = get_uvrange(load_data(sourse, date, freq))
        if uv_min_ > uv_min:
            uv_min = uv_min_
        if uv_max_ < uv_max:
            uv_max = uv_max_                                

    with open(os.path.join(base_dir, "script_clean_rms.txt")) as f:
        lines = f.readlines()

    lines.insert(86, 'uvrange {}, {}'.format(uv_min, uv_max))
    if deep_clean:
        lines[90] = 'float overclean_coef; overclean_coef = 3.0\n'
    else:
        lines[90] = 'float overclean_coef; overclean_coef = 1.0\n'

    with open(os.path.join(outdir, 'script_clean_rms_{}.txt'.format(sourse)), 'w') as f:
        f.writelines(lines)
    
    return uv_min, uv_max


def get_core_img_mask(uvfits, mapsize_clean, beam_fractions, path_to_default_script, use_elliptical=False,
                               use_brightest_pixel_as_initial_guess=True, save_dir=None,
                               base_dir=None, beam=None, uv_range=None,
                               path_to_individual_script=None):

    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        if path_to_individual_script is None:
            path_to_individual_script = path_to_default_script
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean, path_to_script=path_to_individual_script,
                     mapsize_restore=None, beam_restore=beam, shift=None,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)

        if base_dir is not None:
            os.remove(os.path.join(base_dir, "difmap.log"))
        uvdata = UVData(uvfits)
        freq_hz = uvdata.frequency
        # Find beam
        if beam is None:
            bmin, bmaj, bpa = find_nw_beam(uvfits, stokes="i", mapsize=mapsize_clean, uv_range=uv_range, working_dir=working_dir)
        else:
            bmin, bmaj, bpa = beam
        print("NW beam : {:.2f} mas, {:.2f} mas, {:.2f} deg".format(bmaj, bmin, bpa))
        # find resulting image
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
        # detect core and find image
        beam = (bmin, bmaj, bpa)
        core = detect_core(uvfits, beam_fractions, beam, mapsize_clean, freq_hz, path_to_default_script,
                use_brightest_pixel_as_initial_guess, use_elliptical, working_dir)
        npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize_clean[1] ** 2)
        std = find_image_std(ccimage.image, npixels_beam)
        mask = create_beamlike_mask_from_core(core, mapsize_clean, beam, scale=1.8)
        #if base_dir is not None:
        #    os.remove(os.path.join(base_dir, "difmap.log"))

    return ccimage.image, core[1], mask, beam


def detect_core(uvfits, beam_fractions, beam, mapsize_clean, freq_hz, default_script,
                use_brightest_pixel_as_initial_guess, use_elliptical, working_dir):
    
    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc_for_core.fits"), "i",
                     mapsize_clean, path_to_script=default_script,
                     mapsize_restore=None, beam_restore=beam, shift=None,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc_for_core.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)
        
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc_for_core.fits"))

        core = dict()
        # Find the brightest pixel
        if use_brightest_pixel_as_initial_guess:
            im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
            print("indexes of max intensity ", im)
            # - to RA cause dx_RA < 0
            r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
                (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
        else:
            r_c = (0, 0)
        print("Brightest pixel coordinates (RA, DEC) : {:.2f}, {:.2f}".format(r_c[0], r_c[1]))

        # Create model with a single component
        if not use_elliptical:
            comp = (1., r_c[0], r_c[1], 0.25)
        else:
            comp = (1., r_c[0], r_c[1], 0.25, 1.0, 0.0)
        create_difmap_file_from_single_component(comp, os.path.join(working_dir, "1.mdl"), freq_hz)
        
        bmaj = beam[1]

        for beam_fraction in beam_fractions:
            # Filter CCs
            filter_difmap_CC_model_by_r(os.path.join(working_dir, "cc_for_core.mdl"),
                                        os.path.join(working_dir, "filtered_cc_for_core.mdl"),
                                        r_c, bmaj*beam_fraction)
            # Add single gaussian component model to CC model
            join_difmap_models(os.path.join(working_dir, "filtered_cc_for_core.mdl"),
                            os.path.join(working_dir, "1.mdl"),
                            os.path.join(working_dir, "hybrid.mdl"))
            modelfit_difmap(uvfits, mdl_fname=os.path.join(working_dir, "hybrid.mdl"),
                            out_fname=os.path.join(working_dir, "hybrid_fitted.mdl"),
                            niter=100, stokes='i', show_difmap_output=True)

            # Extract core parameters
            with open(os.path.join(working_dir, "hybrid_fitted.mdl"), "r") as fo:
                lines = fo.readlines()
                components = list()
                for line in lines:
                    if line.startswith("!"):
                        continue
                    splitted = line.split()
                    if len(splitted) == 3:
                        continue
                    if len(splitted) == 9:
                        flux, r, theta, major, axial, phi, type_, freq, spec  = splitted
                        flux = float(flux.strip("v"))
                        r = float(r.strip("v"))
                        theta = float(theta.strip("v"))
                        major = float(major.strip("v"))
                        axial = float(axial.strip("v"))
                        phi = float(phi.strip("v"))

                        theta = np.deg2rad(theta)
                        ra = r*np.sin(theta)
                        dec = r*np.cos(theta)

                        # CG
                        if type_ == "1":
                            component = (flux, ra, dec, major)
                        elif type_ == "2":
                            component = (flux, ra, dec, major, axial, phi)
                        else:
                            raise Exception("Component must be Circualr or Elliptical Gaussian!")
                        components.append(component)
                if len(components) > 1:
                    raise Exception("There should be only one core component!")
                if not components:
                    raise Exception("No core component found!")
                # return components[0]

                if not use_elliptical:
                    flux, ra, dec, size = components[0]
                    core.update({beam_fraction: {"flux": flux, "ra": ra,
                                                "dec": dec, "size": size,
                                                "rms": np.nan}})
                else:
                    flux, ra, dec, size, e, bpa = components[0]
                    core.update({beam_fraction: {"flux": flux, "ra": ra,
                                                "dec": dec, "size": size,
                                                "e": e, "bpa": bpa,
                                                "rms": np.nan}})
        try:
            if base_dir is not None:
                os.remove(os.path.join(base_dir, "difmap.log"))
        except:
            pass
        return core


def create_mask_from_core(core, mapsize, working_dir, beam, uvfits, std):
    convert_difmap_model_file_to_CCFITS(os.path.join(working_dir, "1.mdl"), stokes='i', mapsize=mapsize,
                                        restore_beam=beam, uvfits_template=uvfits,
                                        out_ccfits=os.path.join(working_dir, "1.fits"), shift=None,
                                        show_difmap_output=True)
    core_image = create_clean_image_from_fits_file(os.path.join(working_dir, "1.fits"))
    mask = np.ones((mapsize[0], mapsize[0]))
    mask[core_image.image > 5*std] = 0
    return mask


def create_beamlike_mask_from_core(core, mapsize, beam, scale=1):
    print(core)
    mask = np.ones((mapsize[0], mapsize[0]))
    shift_dec_pix = int(core[1]["dec"]/mapsize[1])
    shift_ra_pix = int(core[1]["ra"]/mapsize[1])
    for dec_pix in np.arange(mapsize[0]):
        for ra_pix in np.arange(mapsize[0]):
            x_rot = (dec_pix-mapsize[0]/2-shift_dec_pix)*np.cos(beam[2]*np.pi/180)-np.sin(beam[2]*np.pi/180)*(ra_pix-mapsize[0]/2-shift_ra_pix)
            y_rot = (dec_pix-mapsize[0]/2-shift_dec_pix)*np.sin(beam[2]*np.pi/180)+np.cos(beam[2]*np.pi/180)*(ra_pix-mapsize[0]/2-shift_ra_pix)
            if np.hypot(x_rot*mapsize[1]/beam[1], y_rot*mapsize[1]/beam[0]) <= scale:
                mask[dec_pix, ra_pix] = 0

    return mask


def create_mask_from_round_core(core, mapsize):
    # function creates mask for map deleting sourse core
    mask = np.ones((mapsize[0], mapsize[0]))
    shift_dec_pix = int(core[1]["dec"]/mapsize[1])
    shift_ra_pix = int(core[1]["ra"]/mapsize[1])
    r_pix = int(core["size"]/mapsize[1])
    for dec_pix in np.arange(mapsize[0]):
        for ra_pix in np.arange(mapsize[0]):
            if np.hypot(dec_pix-mapsize[0]/2-shift_dec_pix, ra_pix-mapsize[0]/2+shift_ra_pix) <= 10*r_pix:
                mask[dec_pix, ra_pix] = 0
    return mask


def get_spec_ind(imgs, freq, npixels_beam):
    imgs_log = []
    freqs_log = []
    for img in imgs:
        std = find_image_std(img, npixels_beam)
        img_copy = img.copy()
        img_copy[img < std] = std
        imgs_log.append(np.log(img_copy))    
    for freq in freq:
        freqs_log.append(np.log(freq))    

    shape = imgs_log[0].shape
    imgs_log = np.array(imgs_log)
    spec_ind_map = np.polyfit(freqs_log, imgs_log.reshape((len(freqs_log), shape[0]*shape[1])), 1)[0]
    spec_ind_map = spec_ind_map.reshape(shape)
    return spec_ind_map


def registrate_images(image1, image2, mask1, mask2, fit_gaussian=False, n=9, max_shift=500):
    """
    :param image1:
        2d numpy array with image.
    :param image2:
        2d numpy array with image.
    :param mask1:
        Must be ``True`` at valid pixels.
    :param mask2:
        Must be ``True`` at valid pixels.
    :param fit_gaussian: (optional)
        Fit 2D Gaussian to the peak of the correlation matrix?
        (default: ``True``)
    :param n: (optional)
        Half-width [pix] of the square, centered on the position of the maximum
        correlation, where to fit 2D Gaussian. (default: ``9``)
    :param max_shift: (optional)
        Maximal shift [pix] possible in the calculation
    """
    corr_matrix = cross_correlate_masked(image1, image2, mask1, mask2)
    plt.imshow(corr_matrix)
    plt.savefig('img.png')
    plt.close()
    max_pos = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    while np.hypot(max_pos[0]-image1.shape[0], max_pos[1]-image1.shape[1]) > max_shift:
        corr_matrix[max_pos[0], max_pos[1]] = 0
        max_pos = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    # plt.scatter(max_pos[1], max_pos[0])
    
    if fit_gaussian:
        # Grab a part pf image around maximal correlation coefficient
        sub = corr_matrix[max_pos[0]-n : max_pos[0]+n, max_pos[1]-n : max_pos[1]+n]
        # plt.imshow(sub)

        x, y = np.mgrid[:2*n, :2*n]
        p_init = models.Gaussian2D(1, n, n, n/2, n/2, 0)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, x, y, sub)
        result = p.x_mean.value-n+1+max_pos[0]-image1.shape[0], p.y_mean.value-n+1+max_pos[1]-image2.shape[1]
        if abs(result[0]) > max_shift or abs(result[1]) > max_shift:
            print('Fit unsucessful, returning max position!')
            result = max_pos[0]-image1.shape[0]+1, max_pos[1]-image2.shape[1]+1
    else:
        result = max_pos[0]-image1.shape[0]+1, max_pos[1]-image2.shape[1]+1
    return result


def create_shifted_img(uvfits, mapsize_clean, path_to_script, shift, beam=None, base_dir=None):
    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean, path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=beam, shift=shift,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
    if base_dir is not None:
        os.remove(os.path.join(base_dir, "difmap.log"))
    return ccimage.image


if __name__ == "__main__":
    # setting arguments
    base_dir = '/home/rtodorov/maps-shifts'
    mapsize = (1024, 0.1)
    deep_clean = True
    freqs = [8.1, 8.4, 12.1, 15.4]
    freqs = np.sort(np.array(freqs))
    
    with open("sourse_date_list.txt") as f:
        lines = f.readlines()
    
    data_dict = {'sourse':[],
              'core_shift_dec_81':[],
              'core_shift_ra_81':[],
              'core_shift_dec_84':[],
              'core_shift_ra_84':[],
              'core_shift_dec_121':[],
              'core_shift_ra_121':[],
              'img_shift_dec_81':[],
              'img_shift_ra_81':[],
              'img_shift_dec_84':[],
              'img_shift_ra_84':[],
              'img_shift_dec_121':[],
              'img_shift_ra_121':[],
              'core_loc_dec_81':[],
              'core_loc_ra_81':[],
              'core_loc_dec_84':[],
              'core_loc_ra_84':[],
              'core_loc_dec_121':[],
              'core_loc_ra_121':[],
              'core_loc_dec_154':[],
              'core_loc_ra_154':[]}

    for line in lines:
        arr = line.split()
        if arr[0] == 'skip':
            continue
        sourse = arr[0]
        data_dict['sourse'].append(sourse)
        date = arr[1]
        # getting images, core parameters and masks according to cores
        imgs = []
        cores = []
        masks = []
        beams = []
        beam = None
        uv_range = create_individual_script(sourse, date, freqs, base_dir, deep_clean=deep_clean, outdir=base_dir+'/scripts')

        for freq in freqs:
            img, core, mask, beam = get_core_img_mask(load_data(sourse, date, freq), mapsize, [1], 
                                                path_to_default_script=os.path.join(base_dir, 'script_clean_rms.txt'),
                                                base_dir=base_dir, beam=beam, uv_range=uv_range,
                                                path_to_individual_script=os.path.join(base_dir+'/scripts','script_clean_rms_{}.txt'.format(sourse)))
            imgs.append(img)
            cores.append(core)
            masks.append(mask)
            beams.append(beam)
        
        # correlate maps with the 15.4 GHz and shift if nesessary
        shifted_imgs = []
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            shift_arr =  registrate_images(imgs[-1], img, masks[-1], mask)

            img_shift_dict = {'dec':shift_arr[0]*mapsize[1], 'ra':-shift_arr[1]*mapsize[1]}
            shifted_imgs.append(create_shifted_img(load_data(sourse, date, freqs[i]), mapsize, os.path.join(base_dir, 'script_clean_rms.txt'), \
                                    [0., 0.], beam = beams[-1], base_dir=base_dir))
                                    # [img_shift_dict['ra'], img_shift_dict['dec']], beam = beams[-1], base_dir=base_dir))
            for ax in ['dec', 'ra']:
                data_dict['core_loc_{}_{}'.format(ax, freqs[i]).replace('.', '')].append(cores[i][ax])
                if freqs[i] != freqs[-1]:
                    data_dict['core_shift_{}_{}'.format(ax, freqs[i]).replace('.', '')].append(cores[i][ax] 
                                                                        +img_shift_dict[ax]-cores[-1][ax])
                    data_dict['img_shift_{}_{}'.format(ax, freqs[i]).replace('.', '')].append(img_shift_dict[ax])

        beam = beams[-1]
        npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
        spec_ind_map = get_spec_ind(shifted_imgs, freqs, npixels_beam)

        img_toplot = shifted_imgs[-1]
        std = find_image_std(img_toplot, npixels_beam)
        blc, trc = find_bbox(img_toplot, level=3*std, min_maxintensity_mjyperbeam=20*std,
                            min_area_pix=10*npixels_beam, delta=6)
        if blc[0] == 0: blc = (blc[0] + 1, blc[1])
        if blc[1] == 0: blc = (blc[0], blc[1] + 1)
        if trc[0] == img_toplot.shape: trc = (trc[0] - 1, trc[1])
        if trc[1] == img_toplot.shape: trc = (trc[0], trc[1] - 1)
        x = np.linspace(-mapsize[0]/2*mapsize[1]/206265000, mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
        y = np.linspace(mapsize[0]/2*mapsize[1]/206265000, -mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
        colors_mask = img_toplot < 3*std
        spec_ind_map[spec_ind_map < -2] = -2
        spec_ind_map[spec_ind_map > 1.5] = 1.5
        iplot(contours=img_toplot, colors=spec_ind_map, vectors=None, vectors_values=None, x=x,
                y=y, cmap='gist_rainbow', min_abs_level=3*std, colors_mask=colors_mask, beam=(beam[1], beam[0], beam[2]),
                blc=blc, trc=trc, colorbar_label='$\\alpha$', show_beam=True)
        plt.savefig(os.path.join(base_dir, 'index_maps/spec_ind_map_{}.png'.format(sourse)), bbox_inches='tight')
        hdr = fits.Header()
        hdr['FLUX_I'] = 'Source map of I Stokes parameter on the highest frequnacy'
        hdr['SPEC_IND'] = "Spectral index map"
        primary_hdu = fits.PrimaryHDU(header=hdr)
        image_hdu1 = fits.ImageHDU(data=img_toplot, name="FLUX_I")
        image_hdu2 = fits.ImageHDU(data=spec_ind_map, name="SPEC_IND")
        hdul = fits.HDUList([primary_hdu, image_hdu1, image_hdu2])
        hdul.writeto(os.path.join(base_dir, 'index_maps/spec_ind_map_{}.fits'.format(sourse)), overwrite=True)
        plt.close()

        for i, img in enumerate(imgs):
            iplot(contours=img, colors=masks[i], vectors=None, vectors_values=None, x=x,
                    y=y, cmap='coolwarm', min_abs_level=3*std, colors_mask=None, beam=(beam[1], beam[0], beam[2]),
                    blc=blc, trc=trc, colorbar_label='$mask$', show_beam=True)
            plt.savefig(os.path.join(base_dir, 'index_maps/core_map_{}_{}GHz.png'.format(sourse, freqs[i])), bbox_inches='tight')
            plt.close()

    data = pd.DataFrame(data_dict)
    with open(os.path.join(base_dir, 'un_shift_data.txt'), 'w') as fo:
        fo.write(data.to_string())
