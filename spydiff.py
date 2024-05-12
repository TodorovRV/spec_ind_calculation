import glob
import os
import datetime
import copy
import json
from tempfile import TemporaryDirectory
import pandas as pd
from numpy.linalg import inv, svd
from math import atan2
import numpy as np
from utils import degree_to_rad, get_beam_params_from_CCFITS
from components import DeltaComponent, CGComponent, EGComponent
from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.io.fits import getheader, getdata
import astropy.io.fits as pf
import astropy.units as u
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy.stats import normaltest, chi2, scoreatpercentile
from skimage.measure import regionprops
from from_fits import (create_clean_image_from_fits_file,
                       create_image_from_fits_file, create_model_from_fits_file)
import calendar
from uv_data import UVData, downscale_uvdata_by_freq
from astropy.time import Time

months_dict = {v: k for k, v in enumerate(calendar.month_abbr)}
months_dict_inv = {k: v for k, v in enumerate(calendar.month_abbr)}

deg2mas = u.deg.to(u.mas)
deg2rad = u.deg.to(u.rad)


def FWHM_ell_beam_slice(bmin, bmaj, PA_diff_bpa):
    """
    FWHM of the elliptical beam slice.

    :param bmin:
        Minor FWHM.
    :param bmaj:
        Major FWHM.
    :param PA_diff_bpa:
        Difference between beam BPA (i.e. PA of the beam major axis) and slice PA. [-np.pi/2, np.pi/2], [rad].
    """
    return bmaj*bmin*np.sqrt((1+np.tan(PA_diff_bpa)**2)/(bmin**2+bmaj**2*np.tan(PA_diff_bpa)**2))


def find_nw_beam(uvfits, stokes="I", mapsize=(1024, 0.1), uv_range=None, working_dir=None):
    """
    :return:
        Beam parameters (bmin[mas], bmaj[mas], bpa[deg]).
    """
    if working_dir is None:
        working_dir = os.getcwd()

    original_dir = os.getcwd()
    os.chdir(working_dir)

    # Find and remove all log-files
    previous_logs = glob.glob("difmap.log*")
    for log in previous_logs:
        os.unlink(log)

    stamp = datetime.datetime.now()
    command_file = os.path.join(working_dir, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits + "\n")
    difmapout.write("select " + stokes.lower() + "\n")
    difmapout.write("uvw 0,-2\n")
    if uv_range is not None:
        difmapout.write("uvrange " + str(uv_range[0]) + ", " + str(uv_range[1]) + "\n")
    difmapout.write("mapsize " + str(int(2*mapsize[0])) + ", " + str(mapsize[1]) + "\n")
    difmapout.write("invert\n")
    difmapout.write("quit\n")
    difmapout.close()

    shell_command = "difmap < " + command_file + " 2>&1"
    shell_command += " >/dev/null"
    os.system(shell_command)

    log = os.path.join(working_dir, "difmap.log")
    with open(log, "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Estimated beam:" in line][-1]
    bmin = float(line.split(" ")[3][5:])
    bmaj = float(line.split(" ")[5][5:])
    bpa = float(line.split(" ")[7][4:])

    # Remove command and log file
    os.unlink(command_file)
    os.unlink("difmap.log")
    os.chdir(original_dir)

    return bmin, bmaj, bpa


def check_bbox(blc, trc, image_size):
    """
    :note:
        This can make quadratic image rectangular.
    """
    # If some bottom corner coordinate become negative
    blc = list(blc)
    trc = list(trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    # If some top corner coordinate become large than image size
    if trc[0] > image_size:
        delta = abs(trc[0]-image_size)
        blc[0] -= delta
        # Check if shift have not made it negative
        if blc[0] < 0 and trc[0] > image_size:
            blc[0] = 0
        trc[0] -= delta
    if trc[1] > image_size:
        delta = abs(trc[1]-image_size)
        blc[1] -= delta
        # Check if shift have not made it negative
        if blc[1] < 0 and trc[1] > image_size:
            blc[1] = 0
        trc[1] -= delta
    return tuple(blc), tuple(trc)


def filter_CC(ccfits, mask, out_ccfits=None, out_dfm=None, show=False,
              plotsave_fn=None, axes=None):
    """
    :param mask:
        Mask with region of source flux being True.
    :param out_ccfits:
    """
    import matplotlib.pyplot as plt
    mask = np.array(mask, dtype=bool)
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    data_ = data.copy()
    deg2mas = u.deg.to(u.mas)

    header = pf.getheader(ccfits)
    imsize = header["NAXIS1"]
    wcs = make_wcs_from_ccfits(ccfits)

    xs = list()
    ys = list()
    xs_del = list()
    ys_del = list()
    fs_del = list()
    for flux, x_orig, y_orig in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
        # print("FLux = {}, x = {} deg, y = {} deg".format(flux, x_orig, y_orig))
        x, y = wcs.world_to_array_index(x_orig*u.deg, y_orig*u.deg)
        # print("x = {}, y = {}".format(x, y))
        if x >= imsize:
            x = imsize - 1
        if y >= imsize:
            y = imsize - 1
        if mask[x, y]:
            # print("Mask = {}, keeping component".format(mask[x, y]))
            # Keep this component
            xs.append(x)
            ys.append(y)
        else:
            # print("Mask = {}, removing component".format(mask[x, y]))
            # Remove row from rec_array
            xs_del.append(x_orig)
            ys_del.append(y_orig)
            fs_del.append(flux)

    for (x, y, f) in zip(xs_del, ys_del, fs_del):
        local_mask = ~np.logical_and(np.logical_and(data_["DELTAX"] == x, data_["DELTAY"] == y),
                                     data_["FLUX"] == f)
        data_ = data_.compress(local_mask, axis=0)
    print("Deleted {} components".format(len(xs_del)))

    # if plotsave_fn is not None:
    a = data_['DELTAX']*deg2mas
    b = data_['DELTAY']*deg2mas
    a_all = data['DELTAX']*deg2mas
    b_all = data['DELTAY']*deg2mas

    if show or plotsave_fn is not None:
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        im = axes.scatter(a, b, c=np.log10(1000*data_["FLUX"]), s=5, cmap="binary")
        # axes.scatter(a_all, b_all, color="gray", alpha=0.25, s=2)
        # axes.scatter(a, b, color="red", alpha=0.5, s=1)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.00)
        # cb = fig.colorbar(im, cax=cax)
        # cb.set_label("CC Flux, Jy")
        axes.invert_xaxis()
        axes.set_aspect("equal")
        axes.set_xlabel("RA, mas")
        axes.set_ylabel("DEC, mas")
        if plotsave_fn is not None:
            plt.savefig(plotsave_fn, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()

    hdus[1].data = data_
    hdus[1].header["NAXIS2"] = len(data_)
    if out_ccfits is not None:
        hdus.writeto(out_ccfits, overwrite=True)


def find_bbox(array, level, min_maxintensity_mjyperbeam, min_area_pix,
              delta=0.):
    """
    Find bounding box for part of image containing source.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_maxintensity_mjyperbeam:
        Minimum of the maximum intensity in the region to include.
    :param min_area_pix:
        Minimum area for region to include.
    :param delta: (optional)
        Extra space to add symmetrically [pixels]. (default: ``0``)
    :return:
        Tuples of BLC & TRC.

    :note:
        This is BLC, TRC for numpy array (i.e. transposed source map as it
        conventionally seen on VLBI maps).
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    for prop in props:
        if prop.max_intensity > min_maxintensity_mjyperbeam/1000 and prop.area > min_area_pix:
            signal_props.append(prop)

    # Sometimes no regions are found. In that case return full image
    if not signal_props:
        return (0, 0,), (array.shape[1], array.shape[1],)

    blcs = list()
    trcs = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)

    min_blc_0 = min([blc[0] for blc in blcs])
    min_blc_1 = min([blc[1] for blc in blcs])
    max_trc_0 = max([trc[0] for trc in trcs])
    max_trc_1 = max([trc[1] for trc in trcs])
    blc_rec = (min_blc_0-delta, min_blc_1-delta,)
    trc_rec = (max_trc_0+delta, max_trc_1+delta,)

    blc_rec_ = blc_rec
    trc_rec_ = trc_rec
    blc_rec_, trc_rec_ = check_bbox(blc_rec_, trc_rec_, array.shape[0])

    # Enlarge 10% each side
    delta_ra = abs(trc_rec[0]-blc_rec[0])
    delta_dec = abs(trc_rec[1]-blc_rec[1])
    blc_rec = (blc_rec[0] - int(0.1*delta_ra), blc_rec[1] - int(0.1*delta_dec))
    trc_rec = (trc_rec[0] + int(0.1*delta_ra), trc_rec[1] + int(0.1*delta_dec))

    blc_rec, trc_rec = check_bbox(blc_rec, trc_rec, array.shape[0])

    return blc_rec, trc_rec


def find_image_std(image_array, beam_npixels, min_num_pixels_used_to_estimate_std=100):
    # Robustly estimate image pixels std
    std = mad_std(image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)
    print("Found bounding box : ", blc, trc)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    if mask.shape[0]*mask.shape[1] - np.count_nonzero(mask) < min_num_pixels_used_to_estimate_std:
        return mad_std(image_array)
        # raise Exception("Too small area outside found box with source emission to estimate std - try decrease beam_npixels!")
    outside_icn = np.ma.array(image_array, mask=mask)
    return mad_std(outside_icn)


def is_prop_normally_distributed(prop, array):
    res = normaltest(array[prop.slice].ravel())
    if res.pvalue < 0.05:
        return False
    return True


def is_prop_inside_area(prop, array_shape, blc_target, trc_target):

    bbox_target = [blc_target[1], blc_target[0], trc_target[1], trc_target[0]]
    # print("bbox_target {}".format(bbox_target))
    slice_target = (slice(bbox_target[0], bbox_target[2], None), slice(bbox_target[1], bbox_target[3], None))
    # print("slice target {}".format(slice_target))

    sum = np.zeros(array_shape)
    sum[slice_target] += 1
    sum[prop.slice] += 1

    twos = np.count_nonzero(sum == 2)
    # print("Number of intersections = ", twos)
    # print("Prop.area = ", prop.area)
    if twos > 0.25*prop.area:
        return True
    return False


def find_patches(array, level, min_area_pix):
    """
    Find bounding box for part of image containing uncleaned flux.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_area_pix:
        Minimum area for region to include.
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    # TODO: Simulate what threshold value for solidity is/
    # TODO: Using 10 beams as min area pix
    # TODO: Normally distribution check does not play much role here
    for prop in props:
        if prop.area > min_area_pix and prop.solidity < 0.75 and not is_prop_normally_distributed(prop, array):
            signal_props.append(prop)

    return signal_props


def check_dmap(darray, array, npixels_beam):
    std = find_image_std(array, beam_npixels=npixels_beam)
    patches = find_patches(darray, level=1*std, min_area_pix=10*npixels_beam)
    blc, trc = find_bbox(array, level=4*std, min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*npixels_beam, delta=0)
    uncleaned = list()
    for patch in patches:
        if is_prop_inside_area(patch, array.shape, blc, trc):
            uncleaned.append(patch)

    uncleaned_flux = list()
    for patch in uncleaned:
        uncleaned_flux.append(np.sum(darray[patch.slice].ravel())/npixels_beam)
    uncleaned_flux = np.sum(uncleaned_flux)
    print("Estimated uncleaned flux = {} Jy/beam".format(uncleaned_flux))
    return uncleaned_flux, uncleaned


def clean_n(fname, outfname, stokes, mapsize_clean, niter=100,
            path_to_script=None, mapsize_restore=None, beam_restore=None,
            outpath=None, shift=None, show_difmap_output=False,
            txt_windows=None, clean_box=None):
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    difmapout = open("difmap_commands", "w")
    difmapout.write("observe " + fname + "\n")
    if txt_windows is not None:
        difmapout.write("rwin " + str(txt_windows) + "\n")

    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")

    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + ", " + str(niter) + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
                    str(mapsize_restore[1]) + "\n")
    if outpath is None:
        outpath = os.getcwd()
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < difmap_commands 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


def time_average(uvfits, outfname, time_sec=120, show_difmap_output=True,
                 reweight=True):
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())

    difmapout = open(command_file, "w")
    if reweight:
        difmapout.write("observe " + uvfits + ", {}, true\n".format(time_sec))
    else:
        difmapout.write("observe " + uvfits + ", {}, false\n".format(time_sec))
    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)


def convolve_difmap_model_with_nonuniform_beam(difmap_model_file, stokes, mapsize, uvfits_template,
                                               original_beam, image_std_jyperbeam, out_ccfits="tmp_cc.fits",
                                               show_difmap_output=False):
    npixels_beam = np.pi*original_beam[0]*original_beam[1]/(4*np.log(2)*mapsize[1]**2)
    image_std_jyperpixel = image_std_jyperbeam/npixels_beam
    # Flux, r, theta
    cc_comps = np.loadtxt(difmap_model_file, comments="!")
    # Container for convolved images
    result_image = np.zeros((mapsize[0], mapsize[0]))
    for i in range(len(cc_comps)):
        # Create difmap model file for current component
        np.savetxt("difmap_tmp.mdl", cc_comps[i].reshape(1, -1))

        # Find SNR of current CC
        snr = abs(cc_comps[i][0])/image_std_jyperpixel
        print("CC Flux[Jy] = ", cc_comps[i][0], "SNR = ", snr)

        # Find beam to convolve with
        bmaj = original_beam[0]/np.log10(snr)
        local_beam = (bmaj, bmaj, 0)

        stamp = datetime.datetime.now()
        command_file = "difmap_commands_{}".format(stamp.isoformat())
        difmapout = open(command_file, "w")
        difmapout.write("observe " + uvfits_template + "\n")
        difmapout.write("select " + stokes + "\n")
        difmapout.write("rmodel " + difmap_model_file + "\n")
        difmapout.write("mapsize " + str(int(2*mapsize[0])) + ","+str(mapsize[1])+"\n")
        print("Restoring difmap model with BEAM : bmin = "+str(local_beam[1])+", bmaj = "+str(local_beam[0])+", "+str(local_beam[2])+" deg")
        # default dimfap: false,true (parameters: omit_residuals, do_smooth)
        difmapout.write("restore "+str(local_beam[1])+","+str(local_beam[0])+","+str(local_beam[2])+
                        ","+"true,false"+"\n")
        difmapout.write("wmap " + out_ccfits + "\n")
        difmapout.write("exit\n")
        difmapout.close()

        shell_command = "difmap < "+command_file+" 2>&1"
        if not show_difmap_output:
            shell_command += " >/dev/null"
        os.system(shell_command)
        # Remove command file
        os.unlink(command_file)

        # Load convolved image
        image = getdata(out_ccfits).squeeze()
        result_image += image

    hdus = pf.open(out_ccfits, mode="update")
    hdus[0].data[0, 0, ...] = result_image
    hdus.flush()


def clean_difmap_and_convolve_nubeam(uvfits, stokes, mapsize, outfile, original_beam, path_to_script):
    clean_difmap(fname=uvfits,
                 outfname=outfile, stokes=stokes.lower(),
                 mapsize_clean=mapsize,
                 path_to_script=path_to_script,
                 # With residuals
                 # show_difmap_output=True, beam_restore=None, omit_residuals=True, do_smooth=False)
                 beam_restore=None, omit_residuals=False, do_smooth=False,
                 save_dfm_model="dfm_cc_{}.mdl".format(stokes), show_difmap_output=True,
                 dmap="dmap_{}.fits".format(stokes.lower()))

    # Filter CCs - keep only those within the source
    pass

    # Find noise
    from uv_data import UVData
    uvdata = UVData(uvfits)
    noise = uvdata.noise(average_freq=True)
    noise = np.mean(list(noise.values()))
    std = noise/np.sqrt(uvdata.n_usable_visibilities_difmap(freq_average=True))
    # dimage = pf.getdata("dmap_{}.fits".format(stokes.lower()))[0, 0, ...]
    # std = mad_std(dimage)
    # print("dimage size = ", dimage.shape)

    print("Image std [Jy/beam] = ", std)

    convolve_difmap_model_with_nonuniform_beam("dfm_cc_{}.mdl".format(stokes), stokes, mapsize, uvfits,
                                               original_beam, std, out_ccfits=outfile)


def CCFITS_to_difmap(ccfits, difmap_mdl_file, shift=None):
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    with open(difmap_mdl_file, "w") as fo:
        for flux, ra, dec in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
            ra *= deg2mas
            dec *= deg2mas
            if shift is not None:
                ra -= shift[0]
                dec -= shift[1]
            theta = np.rad2deg(np.arctan2(ra, dec))
            r = np.hypot(ra, dec)
            fo.write("{} {} {}\n".format(flux, r, theta))


def filter_difmap_CC_model_by_r(old_difmap_mdl_file, new_difmap_model_file,
                                center, r_min_mas):
    new_lines = list()
    with open(old_difmap_mdl_file, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            if line.startswith("!"):
                new_lines.append(line)
                continue
            flux, r, theta = line.split()
            flux = float(flux)
            r = float(r)
            theta = np.deg2rad(float(theta))
            ra = r*np.sin(theta)
            dec = r*np.cos(theta)
            # print("RA = {:.2f}, DEC = {:.2f}".format(ra, dec))
            if np.hypot(ra - center[0], dec - center[1]) > r_min_mas:
                new_lines.append(line)

    with open(new_difmap_model_file, "w") as fo:
        for line in new_lines:
            fo.write(line)

def join_difmap_models(dfm_model_1, dfm_model_2, out_file):
    """
    Join difmap VLBI models.
    """
    added_lines = list()

    with open(dfm_model_1, "r") as fo:
        lines = fo.readlines()
        added_lines = lines

    with open(dfm_model_2, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            if line.startswith("!"):
                continue
            added_lines.append(line)

    with open(out_file, "w") as fo:
        for line in added_lines:
            fo.write(line)


def create_difmap_file_from_single_component(comp, out_fname, freq_hz):
    """
    :param comp:
        (flux, ra, dec[, bmaj, e, bpa]) - [Jy, mas, mas, mas, -, deg]

    """

    with open(out_fname, "w") as fo:
        if len(comp) == 3:
            flux, ra, dec = comp
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type_ = "0"
        elif len(comp) == 4:
            flux, ra, dec, bmaj = comp
            e = "1.00000"
            bpa = "000.000"
            type_ = "1"
            bmaj = "{:.5f}v".format(bmaj)
        elif len(comp) == 6:
            flux, ra, dec, bmaj, e, bpa = comp
            e = "{:.5f}v".format(e)
            bpa = "{:.5f}v".format((bpa - np.pi / 2) / deg2rad)
            bmaj = "{:.5f}v".format(bmaj)
            type_ = "1"
        else:
            raise Exception
        # mas
        r = np.hypot(ra, dec)
        # rad
        theta = np.rad2deg(np.arctan2(ra, dec))
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        fo.write("{:.5f}v {:.5f}v {:.5f}v {} {} {} {} {:.2E} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type_,
                                                         freq_hz))
        # fo.write("{:>11.7f}v {:>13.7f}v {:>13.5f}v {:>13} {:>13} {:>13} {:>3} {:>12.5e} {:>12d}\n".format(flux, r, theta,
        #                                                                                                   bmaj, e, bpa, type_,
        #                                                                                                   freq_hz, 0))

def remove_residuals_from_CLEAN_map(ccfits, uvfits, out_ccfits, mapsize, stokes="i", restore_beam=None, shift=None,
                                    show_difmap_output=True, working_dir=None):

    if working_dir is None:
        working_dir = os.getcwd()
    tmp_difmap_mdl_file = os.path.join(working_dir, "dfm.mdl")
    CCFITS_to_difmap(ccfits, tmp_difmap_mdl_file)
    if restore_beam is None:
        # Obtain convolving beam from FITS file
        restore_beam = get_beam_params_from_CCFITS(ccfits)
    convert_difmap_model_file_to_CCFITS(tmp_difmap_mdl_file, stokes, mapsize, restore_beam, uvfits, out_ccfits,
                                        shift, show_difmap_output)


# FIXME: Beam BPA in degrees!
def convert_difmap_model_file_to_CCFITS(difmap_model_file, stokes, mapsize,
                                        restore_beam, uvfits_template,
                                        out_ccfits, shift=None,
                                        show_difmap_output=True):
    """
    Using difmap-formated model file (e.g. flux, r, theta) obtain convolution of
    your model with the specified beam.

    :param difmap_model_file:
        Difmap-formated model file. Use ``JetImage.save_image_to_difmap_format`` to obtain it.
    :param stokes:
        Stokes parameter.
    :param mapsize:
        Iterable of image size and pixel size (mas).
    :param restore_beam:
        Beam to restore: bmaj(mas), bmin(mas), bpa(deg).
    :param uvfits_template:
        Template uvfits observation to use. Difmap can't read model without having observation at hand.
    :param out_ccfits:
        File name to save resulting convolved map.
    :param shift: (optional)
        Shift to apply. Need this because wmodel doesn't apply shift. If
        ``None`` then do not apply shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show Difmap output? (default: ``True``)
    """
    from subprocess import Popen, PIPE

    cmd = "observe " + uvfits_template + "\n"
    cmd += "select " + stokes + "\n"
    cmd += "rmodel " + difmap_model_file + "\n"
    cmd += "mapsize " + str(mapsize[0] * 2) + "," + str(mapsize[1]) + "\n"
    if shift is not None:
        # Here we need shift, because in CLEANing shifts are not applied to
        # saving model files!
        cmd += "shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n"
    cmd += "uvw 0, -2\n"
    print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[0]) + ", bmaj = " + str(restore_beam[1]) + ", " + str(restore_beam[2]) + " deg")
    # default dimfap: false,true (parameters: omit_residuals, do_smooth)
    cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
    cmd += "wmap " + out_ccfits + "\n"
    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)


def flag_baseline(uvfits, outfname, ta, tb, show_difmap_output=False):
    """
    Flag specified baseline.
    """
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits + "\n")
    difmapout.write("select i\n")

    difmapout.write("flag 1:{}-{}, true\n".format(ta, tb))

    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)
    # Remove command file
    os.unlink(command_file)


def flag_baseline_scan(uvfits, outfname, ta, tb=None, start_time=None, stop_time=None, except_time_range=False,
                       show_difmap_output=False):
    """
    Flag visibilities on baselines/antenna in specified time range (or all except specified time range).

    :param uvfits:
        Path to UVFITS file.
    :param outfname:
        Path to UVFITS save file.
    :param ta:
        First telescope to flag.
    :param tb: (optional)
        Second telescope to flag. If ``None``, then flag all with ``ta``. (default: ``None``)
    :param start_time: (optional)
        Astropy Time object. Start of the flagging interval. If ``None`` then flag from the beginning.
        (default: ``None``)
    :param stop_time: (optional)
        Astropy Time object. Stop of the flagging interval. If ``None`` then flag till the end.
        (default: ``None``)
    :param except_time_range: (optional)
        Flag outside of the time interval? (default: ``False``)
    """
    if start_time is not None:
        y, month, d, h, m, s = start_time.ymdhms
        # dd-mmm-yyyy-mm:hh:mm:ss
        t_start_difmap = "{}-{}-{}:{}:{}:{}".format(str(d).zfill(2), months_dict_inv[month].lower(), y, str(h).zfill(2),
                                                    str(m).zfill(2), str(int(round(s-1, 0))).zfill(2))
    else:
        t_start_difmap = ""

    if stop_time is not None:
        y, month, d, h, m, s = stop_time.ymdhms
        # dd-mmm-yyyy-mm:hh:mm:ss
        t_stop_difmap = "{}-{}-{}:{}:{}:{}".format(str(d).zfill(2), months_dict_inv[month].lower(), y, str(h).zfill(2),
                                                   str(m).zfill(2), str(int(round(s+1, 0))).zfill(2))
    else:
        t_stop_difmap = ""

    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())
    difmapout = open(command_file, "w")
    difmapout.write("observe " + uvfits + "\n")
    difmapout.write("select i\n")

    if not except_time_range:
        if tb is not None:
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, t_start_difmap, t_stop_difmap))
        else:
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, t_start_difmap, t_stop_difmap))
    else:
        if tb is not None:
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, "", t_start_difmap))
            difmapout.write("flag 1:{}-{}, true, {}, {}\n".format(ta, tb, t_stop_difmap, ""))
        else:
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, "", t_start_difmap))
            difmapout.write("flag 1:{}, true, {}, {}\n".format(ta, t_stop_difmap, ""))

    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)
    # Remove command file
    os.unlink(command_file)


# TODO: add ``shift`` argument, that shifts image before cleaning. It must be
# more accurate to do this in difmap. Or add such method in ``UVData`` that
# multiplies uv-data on exp(-1j * (u*x_shift + v*y_shift)).
# FIXME: BPA in deg!
def clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                 path_to_script=None, mapsize_restore=None, beam_restore=None,
                 outpath=None, shift=None, show_difmap_output=False,
                 command_file=None, clean_box=None, save_dfm_model=None, omit_residuals=False,
                 do_smooth=True, dmap=None, text_box=None,
                 box_rms_factor=None, window_file=None,
                 super_unif_dynam=None, unif_dynam=None,
                 taper_gaussian_value=None, taper_gaussian_radius=None):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size [mas]).
    :param path: (optional)
        Path to uv-data to clean. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use naitive. (default: ``None``)
    :param beam_restore: (optional)
        Beam parameter for restore map (bmaj[mas], bmin[mas], bpa[deg]). If
        ``None`` then use the same beam as in cleaning. (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param shift: (optional)
        Iterable of 2 values - shifts in both directions - East & North [mas].
        If ``None`` then don't shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)
    :param command_file: (optional)
        Script file name to store `difmap` commands. If ``None`` then use
        ``difmap_commands``. (default: ``None``)
    :param clean_box: (optional)
         xa  -   The relative Right-Ascension of either edge of the new window.
         xb  -   The relative Right-Ascension of the opposite edge to 'xa', of
         the new window.
         ya  -   The relative Declination of either edge of the new window.
         yb  -   The relative Declination of the opposite edge to 'ya', of the
         new window.
         If ``None`` than do not use CLEAN windows. (default: ``None``)
    :param text_box: (optional)
        Path to text file with clean box (difmap output) to use. If ``None``
        then do not use text box. (default: ``None``)
    :param save_dfm_model: (optional)
        File name to save difmap-format model with CCs. If ``None`` then do
        not save model. (default: ``None``)
    :param dmap: (optional)
        FIle name to save the residual map. If ``None`` then do not save
        the residual map. (default: ``None``)


    :Note:
           Following 4 parameters are used for alpha-project.

    :param super_unif_dynam: (optional)
        Parameter for minimal dynamic range to use in super uniform weighting.
        Only for some CLEANing scripts!
    :param unif_dynam: (optional)
        Parameter for minimal dynamic range to use in uniform weighting.
        Only for some CLEANing scripts!
    :param taper_gaussian_value: (optional)
        Taper parameter for deep cleaning. Only for some CLEAN scripts!
    :param taper_gaussian_radius: (optional)
        Taper parameter for deep cleaning. Only for some CLEAN scripts!



    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        # command_file = "difmap_commands"
        stamp = datetime.datetime.now()
        command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")

    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")
    if text_box is not None:
        difmapout.write("rwins " + str(text_box) + "\n")

    # This is interface to final_clean_box script
    if box_rms_factor is not None and window_file is not None:
        difmapout.write("@" + path_to_script + " " + stokes + ", " + str(box_rms_factor) + ", " + window_file + "\n")
    elif super_unif_dynam is not None and unif_dynam is not None:
        if taper_gaussian_value is None or taper_gaussian_radius is None:
            # No taper
            taper_gaussian_value = 1.1
            taper_gaussian_radius = 0.0
        difmapout.write("@" + path_to_script + " " + stokes + ", " +
                        str(super_unif_dynam) + ", " + str(unif_dynam) +
                        str(taper_gaussian_value) + ", " + str(taper_gaussian_radius) + "\n")
    # Here boxes are optionally included via text_box argument
    else:
        difmapout.write("@"+path_to_script+" "+stokes+"\n")

    # FIXME: Do I need this?
    # difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
    #                 str(mapsize_clean[1]) + "\n")

    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")

    if beam_restore is not None:
        if omit_residuals:
            omit_residuals = "true"
        else:
            omit_residuals = "false"
        if do_smooth:
            do_smooth = "true"
        else:
            do_smooth = "false"
        difmapout.write("restore " + str(beam_restore[1]) + ', ' +
                        str(beam_restore[0]) + ', ' + str(beam_restore[2]) + ", " + omit_residuals + ", " + do_smooth +
                        "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"

    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    if dmap is not None:
        difmapout.write("wdmap " + os.path.join(outpath, dmap) + "\n")
    if save_dfm_model is not None:
        # FIXME: Difmap doesn't apply shift to model components!
        difmapout.write("wmodel "+os.path.join(outpath, save_dfm_model)+"\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    try:
        os.unlink(command_file)
    except FileNotFoundError:
        pass


def make_wcs_from_ccfits(ccfits):
    header = pf.getheader(ccfits)

    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial

    # Make offset coordinates
    wcs.wcs.crval = 0., 0.
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'mas', 'mas'
    wcs.wcs.cdelt = (wcs.wcs.cdelt * u.deg).to(u.mas)
    return wcs


def convert_boxfile_to_mask(ccfits, boxfile):
    boxes = np.loadtxt(boxfile, comments="!")
    wcs = make_wcs_from_ccfits(ccfits)
    blctrc = list()
    for box in boxes:
        ra_min, ra_max, dec_min, dec_max = box
        blc, trc = convert_radec_ranges_to_bbox(wcs, ra_min, ra_max, dec_min, dec_max)
        blctrc.append((blc, trc))

    header = pf.getheader(ccfits)
    mask = np.ones((header["NAXIS1"], header["NAXIS2"]), dtype=int)
    for blc, trc in blctrc:
        mask[blc[1]:trc[1], blc[0]:trc[0]] = 0
    return np.array(mask, dtype=bool)


def get_rms_from_map_region(ccfits, boxfile):
    mask = convert_boxfile_to_mask(ccfits, boxfile)
    image = pf.getdata(ccfits).squeeze()
    image = np.ma.array(image, mask=mask)
    return np.ma.std(image)

def get_uvrange(uvfits, stokes=None):
    """
    :param stokes: (optional)
        Stokes to consider masking. If ``None`` then ``I``. (default: ``None``)
    :return:
        Minimal and maximal uv-spacing in Ml.
    """
    uvdata = UVData(uvfits)
    if stokes is None:
        # Get RR and LL correlations
        rrll = uvdata.uvdata_freq_averaged[:, :2]
        # Masked array
        I = np.ma.mean(rrll, axis=1)
    else:
        if stokes == "RR":
            I = uvdata.uvdata_freq_averaged[:, 0]
        elif stokes == "LL":
            I = uvdata.uvdata_freq_averaged[:, 1]
    weights_mask = I.mask

    uv = uvdata.uv
    uv = uv[~weights_mask]
    r_uv = np.hypot(uv[:, 0], uv[:, 1])
    return min(r_uv/10**6), max(r_uv/10**6)


# FIXME: Handle cases when no V is available and we need to estimate target rms from 1 asec distant region
# FIXME: Note that I have changed ``save_...`` arguments from boolean to None/filename
def CLEAN_difmap(uvfits, stokes, mapsize, outname, restore_beam=None,
                 boxfile=None, working_dir=None, uvrange=None,
                 box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                 remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                 noise_to_use="F", shift=None):
    if noise_to_use not in ("V", "W", "F"):
        raise Exception("noise_to_use must be V (from Stokes V), W (from weights) or F (from remote region)!")
    print("=== Using target rms estimate from {}".format({"V": "Stokes V", "W": "weights", "F": "remote region"}[noise_to_use]))
    stamp = datetime.datetime.now()
    from subprocess import Popen, PIPE

    if working_dir is None:
        working_dir = os.getcwd()
    current_dir = os.getcwd()
    os.chdir(working_dir)

    # First get noise estimates: from visibility & weights and from Stokes V
    cmd = "wrap_print_output = false\n"
    cmd += "observe "+uvfits+"\n"
    cmd += "select "+stokes+"\n"
    cmd += "mapsize "+str(int(2*mapsize[0]))+","+str(mapsize[1])+"\n"
    cmd += "uvw 0, -2\n"
    if uvrange is not None:
        cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
    cmd += "print \"wtnoise =\", imstat(noise)\n"

    cmd += "select v\n"
    cmd += "uvw 0, -2\n"
    if uvrange is not None:
        cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
    cmd += "print \"vnoise =\", imstat(rms)\n"

    cmd += "select i\n"
    cmd += "shift 10000,10000\n"
    cmd += "print \"farnoise =\", imstat(rms)\n"

    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)

    lines = outs.split("\n")
    line = [line for line in lines if "wtnoise =" in line][-1]
    wtnoise = float(line.split("=")[1])
    line = [line for line in lines if "vnoise =" in line][-1]
    vnoise = float(line.split("=")[1])
    line = [line for line in lines if "farnoise =" in line][-1]
    farnoise = float(line.split("=")[1])
    line = [line for line in lines if "Estimated beam:" in line][-1]
    bmin = float(line.split(" ")[2].split("=")[1])
    bmaj = float(line.split(" ")[4].split("=")[1])
    bpa = float(line.split(" ")[6].split("=")[1])

    # Large weights noise
    if wtnoise > 10*vnoise:
        print("=== Noise from weights ({:.3f} mJy/beam) is much larger then from Stokes V ({:.3f} mJy/beam)".format(1000*wtnoise, 1000*vnoise))
        if noise_to_use == "V":
            target_rms = vnoise
        else:
            target_rms = farnoise
    else:
        if noise_to_use == "V":
            target_rms = vnoise
        elif noise_to_use == "W":
            target_rms = wtnoise
        else:
            target_rms = farnoise

    print("=== Far region rms = {:.3f} mJy/beam".format(1000*farnoise))
    print("=== Weights rms    = {:.3f} mJy/beam".format(1000*wtnoise))
    print("=== V noise rms    = {:.3f} mJy/beam".format(1000*vnoise))

    if restore_beam is None:
        restore_beam = (bmin, bmaj, bpa)

    # CLEAN with SU-weighting
    cmd = "observe "+uvfits+"\n"
    cmd += "select "+stokes+"\n"
    if uvrange is not None:
        cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
    cmd += "mapsize "+str(int(2*mapsize[0]))+","+str(mapsize[1])+"\n"
    cmd += "integer clean_niter;\n float clean_gain;\n clean_gain = {};\n float dynam;\n float flux_peak;\n" \
           "float flux_cutofff;\n float in_rms;\n float target_rms;\n float last_in_rms\n".format(clean_gain)
    cmd += "#+map_residual \
flux_peak = peak(flux);\
flux_cutoff = imstat(rms) * dynam;\
while(abs(flux_peak)>flux_cutoff);\
 clean clean_niter,clean_gain;\
 flux_cutoff = imstat(rms) * dynam;\
 flux_peak = peak(flux);\
end while\n"
    cmd += "dynam = {}\n clean_niter = 10\n uvw 20,-1\n map_residual\n uvw 10,-1\n map_residual\n".format(dynam_su)
    cmd += "uvw 2,-1\n dynam = {}\n map_residual\n".format(dynam_u)

    cmd += "#+deep_map_residual \
in_rms = imstat(rms);\
while(in_rms > {}*target_rms);\
 clean min(100*(in_rms/target_rms),500),clean_gain;\
 last_in_rms = in_rms;\
 in_rms = imstat(rms);\
 if(last_in_rms <= in_rms);\
  in_rms = target_rms;\
 end if;\
end while\n".format(deep_factor)

    if boxfile is None:
        cmd += "uvw 0,-2\n target_rms = imstat(noise)\n deep_map_residual\n"
        # default dimfap: false,true (parameters: omit_residuals, do_smooth)
        if shift is not None:
            cmd += "shift "+str(shift[0])+', '+str(shift[1])+"\n"
        cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "false,true" + "\n"
        cmd += "wmap " + outname + "\n"
        if save_noresid is not None:
            cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
            cmd += "wmap " + save_noresid + "\n"
        if save_resid_only is not None:
            cmd += "wdmap " + save_resid_only + "\n"
        if save_dfm is not None:
            cmd += "wmod " + save_dfm + "\n"
        cmd += "exit\n"
    # If boxes
    else:
        cmd += "rwins {}\n".format(boxfile)
        cmd += "save {}\n".format(stamp.isoformat())
        cmd += "wdmap {}_resid_only.fits\n".format(stamp.isoformat())
        cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)

    if boxfile is not None:
        box_rms = get_rms_from_map_region(os.path.join(working_dir, "{}_resid_only.fits".format(stamp.isoformat())),
                                          boxfile)
        print("INBOX rms = {:.3f} mJy/beam, while TARGET rms = {:.3f} mJy/beam".format(1000*box_rms, 1000*deep_factor*target_rms))
        while box_rms > deep_factor*target_rms:
            print("Current INBOX rms = {:.3f} mJy/beam > TARGET rms = {:.3f} mJy/beam => CLEANing deeper...".format(1000*box_rms, 1000*deep_factor*target_rms))
            cmd = "@{}.par\n".format(stamp.isoformat())
            cmd += "uvw 0,-2\n"
            if uvrange is not None:
                cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
            cmd += "clean {}, {}\n".format(box_clean_nw_niter, clean_gain)
            cmd += "save {}\n".format(stamp.isoformat())
            cmd += "wdmap {}_resid_only.fits\n".format(stamp.isoformat())
            cmd += "exit\n"
            with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
                outs, errs = difmap.communicate(input=cmd)
            box_rms = get_rms_from_map_region(os.path.join(working_dir, "{}_resid_only.fits".format(stamp.isoformat())), boxfile)

        cmd = "@{}.par\n".format(stamp.isoformat())
        if shift is not None:
            cmd += "shift "+str(shift[0])+', '+str(shift[1])+"\n"
        cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "false,true" + "\n"
        cmd += "wmap " + outname + "\n"
        if save_noresid is not None:
            cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
            cmd += "wmap " + save_noresid + "\n"
        if save_resid_only is not None:
            cmd += "wdmap " + save_resid_only + "\n"
        if save_dfm is not None:
            cmd += "wmod " + save_dfm + "\n"
        cmd += "exit\n"
        with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
            outs, errs = difmap.communicate(input=cmd)

        for fn in ("{}_resid_only.fits".format(stamp.isoformat()),
                   "{}.fits".format(stamp.isoformat()),
                   "{}.par".format(stamp.isoformat()),
                   "{}.win".format(stamp.isoformat()),
                   "{}.uvf".format(stamp.isoformat()),
                   "{}.mod".format(stamp.isoformat())):
            os.unlink(os.path.join(working_dir, fn))

    if remove_difmap_logs:
        logs = glob.glob(os.path.join(working_dir, "difmap.log*"))
        for log in logs:
            os.unlink(log)

    os.chdir(current_dir)
    return outs, errs


def rebase_CLEAN_model(target_uvfits, rebased_uvfits, stokes, mapsize, restore_beam, source_ccfits=None,
                       source_difmap_model=None, noise_scale_factor=1.0, need_downscale_uv=None, remove_cc=False):
    if source_ccfits is None and source_difmap_model is None:
        raise Exception("Must specify CCFITS or difmap model file!")
    if source_ccfits is not None and source_difmap_model is not None:
        raise Exception("Must specify CCFITS OR difmap model file!")
    uvdata = UVData(target_uvfits)
    if need_downscale_uv is None:
        need_downscale_uv = downscale_uvdata_by_freq(uvdata)
    noise = uvdata.noise(average_freq=False, use_V=False)
    uvdata.zero_data()
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    if source_ccfits is not None:
        ccmodel = create_model_from_fits_file(source_ccfits)
    if source_difmap_model is not None:
        convert_difmap_model_file_to_CCFITS(source_difmap_model, stokes, mapsize,
                                            restore_beam, target_uvfits, "tmp_cc.fits")
        ccmodel = create_model_from_fits_file("tmp_cc.fits")
        if remove_cc:
            os.unlink("tmp_cc.fits")

    uvdata.substitute([ccmodel])
    uvdata.noise_add(noise)
    uvdata.save(rebased_uvfits, rewrite=True, downscale_by_freq=need_downscale_uv)



def deep_clean_difmap(fname, outfname, stokes, mapsize_clean, path=None,
                      path_to_script=None, mapsize_restore=None, beam_restore=None,
                      outpath=None, shift=None, show_difmap_output=False,
                      command_file=None, clean_box=None, prefix="tmp"):
    """
    Map self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param stokes:
        Stokes parameter 'i', 'q', 'u' or 'v'.
    :param mapsize_clean:
        Parameters of map for cleaning (map size, pixel size [mas]).
    :param path: (optional)
        Path to uv-data to clean. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``clean`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param mapsize_restore: (optional)
        Parameters of map for restoring CC (map size, pixel size). If
        ``None`` then use naitive. (default: ``None``)
    :param beam_restore: (optional)
        Beam parameter for restore map (bmaj, bmin, bpa). If ``None`` then use
        the same beam as in cleaning. (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param shift: (optional)
        Iterable of 2 values - shifts in both directions - East & North [mas].
        If ``None`` then don't shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)
    :param command_file: (optional)
        Script file name to store `difmap` commands. If ``None`` then use
        ``difmap_commands``. (default: ``None``)
    :param clean_box: (optional)
         xa  -   The relative Right-Ascension of either edge of the new window.
         xb  -   The relative Right-Ascension of the opposite edge to 'xa', of
         the new window.
         ya  -   The relative Declination of either edge of the new window.
         yb  -   The relative Declination of the opposite edge to 'ya', of the
         new window.
         If ``None`` than do not use CLEAN windows. (default: ``None``)


    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    if not mapsize_restore:
        mapsize_restore = mapsize_clean

    if command_file is None:
        # command_file = "difmap_commands"
        stamp = datetime.datetime.now()
        command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    # if shift is not None:
    #     difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
                    str(mapsize_clean[1]) + "\n")
    if clean_box is not None:
        difmapout.write("addwin " + str(clean_box[0]) + ', ' +
                        str(clean_box[1]) + ', ' + str(clean_box[2]) + ', ' +
                        str(clean_box[3]) + "\n")
    difmapout.write("@" + path_to_script + " " + stokes + "\n")
    if beam_restore:
        difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                        str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                        "\n")
    # difmapout.write("mapsize " + str(mapsize_restore[0] * 2) + ', ' +
    #                 str(mapsize_restore[1]) + "\n")
    # if outpath is None:
    #     outpath = path
    # elif not outpath.endswith("/"):
    #     outpath = outpath + "/"
    if shift is not None:
        difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
    difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("wdmap " + os.path.join(outpath, "dmap_" + outfname) + "\n")

    if os.path.exists("{}.fits".format(prefix)):
        for ext in ("fits", "mod", "par", "uvf"):
            os.unlink("{}.{}".format(prefix, ext))

    difmapout.write("save " + prefix + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)


    header = getheader(os.path.join(outpath, outfname))
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial
    # Make offset coordinates
    wcs.wcs.crval = 0., 0.
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'mas', 'mas'
    wcs.wcs.cdelt = (wcs.wcs.cdelt * u.deg).to(u.mas)
    print(wcs)

    ccimage = create_clean_image_from_fits_file(os.path.join(outpath, outfname))
    dimage = create_image_from_fits_file(os.path.join(outpath, "dmap_" + outfname))
    beam = ccimage.beam
    npixels_beam = np.pi * beam[0] * beam[1] / mapsize_clean[1] ** 2
    print("Checking uncleaned patches...")
    uncleaned_flux, uncleaned_patches = check_dmap(dimage.image, ccimage.image, npixels_beam)

    # uncleaned_patches = list()

    if uncleaned_patches:
        # Remove old command file
        os.unlink(command_file)

        # TODO: Create CLEAN boxes from patches
        boxes = list()
        for patch in uncleaned_patches:
            bbox = patch.bbox
            blc = (int(bbox[1]), int(bbox[0]))
            trc = (int(bbox[3]), int(bbox[2]))
            print("patch wiht blc,trc ", blc, trc)
            ra_min, ra_max, dec_min, dec_max = convert_bbox_to_radec_ranges(wcs, blc, trc)
            boxes.append((ra_min, ra_max, dec_min, dec_max))

        print(boxes)

        difmapout = open(command_file, "w")
        print("Loading saved state...")
        # difmapout.write("get " + prefix + "\n")
        difmapout.write("@{}.par\n".format(prefix))
        difmapout.write("uvw 0,-2\n")

        for box in boxes:
            difmapout.write("addwin {},{},{},{}\n".format(*box))

        # difmapout.write("select i\n")
        # difmapout.write("mapsize " + str(mapsize_clean[0] * 2) + ', ' +
        #                 str(mapsize_clean[1]) + "\n")
        print("NW CLEAN with 500, 0.03")
        difmapout.write("clean 10000,0.03\n")
        if beam_restore:
            difmapout.write("restore " + str(beam_restore[0]) + ', ' +
                            str(beam_restore[1]) + ', ' + str(beam_restore[2]) +
                            "\n")
        difmapout.write("wmap " + os.path.join(outpath, outfname) + "\n")
        difmapout.write("wdmap " + os.path.join(outpath, "dmap_" + outfname) + "\n")
        difmapout.write("exit\n")
        difmapout.close()

        # TODO: Use subprocess for silent cleaning?
        shell_command = "difmap < " + command_file + " 2>&1"
        os.system(shell_command)

    # Remove command file
    # os.unlink(command_file)
    # Remove saved state if any
    if os.path.exists(os.path.join(outpath, "{}.fits".format(prefix))):
        for ext in ("fits", "mod", "par", "uvf"):
            os.unlink(os.path.join(outpath, "{}.{}".format(prefix, ext)))


def selfcal_difmap(fname, outfname, path=None, path_to_script=None, outpath=None, show_difmap_output=False):
    """
    Self-calibrated uv-data in difmap.
    :param fname:
        Filename of uv-data to clean.
    :param outfname:
        Filename with CCs.
    :param path: (optional)
        Path to uv-data to self-calibrate. If ``None`` then use current directory.
        (default: ``None``)
    :param path_to_script: (optional)
        Path to ``selfcal`` difmap script. If ``None`` then use current directory.
        (default: ``None``)
    :param outpath: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param show_difmap_output: (optional)
        Show difmap output? (default: ``False``)

    """
    if path is None:
        path = os.getcwd()
    if outpath is None:
        outpath = os.getcwd()

    stamp = datetime.datetime.now()
    command_file = os.path.join(outpath, "difmap_commands_{}".format(stamp.isoformat()))

    difmapout = open(command_file, "w")
    difmapout.write("@" + path_to_script + " " + os.path.join(path, fname) + "\n")
    if outpath is None:
        outpath = path
    elif not outpath.endswith("/"):
        outpath = outpath + "/"
    difmapout.write("wobs " + os.path.join(outpath, outfname) + "\n")
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)


def import_difmap_model(mdl_fname, mdl_dir=None, remove_last_char=True):
    """
    Function that reads difmap-format model and returns list of ``Components``
    instances.

    :param mdl_fname:
        File name with difmap model.
    :param mdl_dir: (optional)
        Directory with difmap model. If ``None`` then use CWD. (default:
        ``None``)
    :param remove_last_char: (optional)
        Remove last character (v ~ variable) from value? It works for difmap
        modelfit files, but for difmap CC files should be ``False``. (default:
        ``True``)
    :return:
        List of ``Components`` instances.
    """
    if mdl_dir is None:
        mdl_dir = os.getcwd()
    mdl = os.path.join(mdl_dir, mdl_fname)
    mdlo = open(mdl)
    lines = mdlo.readlines()
    comps = list()
    for line in lines:
        if line.startswith('!'):
            continue
        line = line.strip('\n ')
        try:
            flux, radius, theta, major, axial, phi, type_, freq, spec = line.split()
            # print("Read line:")
            # print(flux, radius, theta, major, axial, phi, type_, freq, spec)
        except ValueError:
            try:
                flux, radius, theta, major, axial, phi, type_ = line.split()
            except ValueError:
                try:
                    flux, radius, theta = line.split()
                except ValueError:
                    print("Problem parsing line :\n")
                    print(line)
                    raise ValueError
                axial = 1.0
                major = 0.0
                type_ = 0

        list_fixed = list()
        if flux[-1] != 'v':
            list_fixed.append('flux')

        if remove_last_char:
            x = -float(radius[:-1]) * np.sin(np.deg2rad(float(theta[:-1])))
            y = -float(radius[:-1]) * np.cos(np.deg2rad(float(theta[:-1])))
            flux = float(flux[:-1])
        else:
            x = -float(radius) * np.sin(np.deg2rad(float(theta)))
            y = -float(radius) * np.cos(np.deg2rad(float(theta)))
            flux = float(flux)

        if int(type_) == 0:
            comp = DeltaComponent(flux, x, y)
        elif int(type_) == 1:

            try:
                bmaj = float(major)
                list_fixed.append('bmaj')
            except ValueError:
                bmaj = float(major[:-1])

            # if float(axial[:-1]) == 1.0:
            try:
                floataxial = float(axial)
            except ValueError:
                floataxial = float(axial[:-1])
            if floataxial == 1.0:
                comp = CGComponent(flux, x, y, bmaj, fixed=list_fixed)
            else:
                try:
                    e = float(axial)
                except ValueError:
                    e = float(axial[:-1])
                try:
                    bpa = np.deg2rad(float(phi)) + np.pi / 2.
                except ValueError:
                    bpa = np.deg2rad(float(phi[:-1])) + np.pi / 2.
                comp = EGComponent(flux, x, y, bmaj, e, bpa, fixed=list_fixed)
        else:
            raise NotImplementedError("Only CC, CG & EG are implemented")
        comps.append(comp)
    return comps


def export_difmap_model_from_tuples(comps, out_fname, freq_hz):
    """
    :param comps:
        Iterable of tuples with (flux, x, y, bmaj).
    :param out_fname:
        Path for saving file.
    """
    with open(out_fname, "w") as fo:
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        for comp in comps:
            if len(comp) == 4:
                # Jy, mas, mas, mas
                flux, x, y, bmaj = comp
                e = "1.00000"
                bpa = "000.000"
                type = "1"
                bmaj = "{:.7f}v".format(bmaj)
            elif len(comp) == 6:
                # Jy, mas, mas, mas, -, deg
                flux, x, y, bmaj, e, bpa = comp
                e = "{}v".format(e)
                bpa = "{}v".format((bpa-np.pi/2)/degree_to_rad)
                bmaj = "{}v".format(bmaj)
                type = "1"
            elif len(comp) == 3:
                flux, x, y = comp
                e = "1.00000"
                bmaj = "0.0000"
                bpa = "000.000"
                type = "0"
            else:
                raise Exception
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(x, y)
            theta /= degree_to_rad
            fo.write("{:>11.7f}v {:>13.7f}v {:>13.5f}v {:>13} {:>13} {:>13} {:>3} {:>12.5e} {:>12d}\n".format(flux, r, theta,
                                                              bmaj, e, bpa, type,
                                                             freq_hz, 0))


def export_difmap_model(comps, out_fname, freq_hz):
    """
    Export iterable of ``Component`` instances to the Difmap-format model file.

    :param comps:
        Iterable of ``Component`` instances.
    :param out_fname:
        Path for saving file.
    """
    with open(out_fname, "w") as fo:
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        for comp in comps:
            if isinstance(comp, EGComponent):
                if len(comp.p_all) == 4:
                    # Jy, mas, mas, mas
                    flux, x, y, bmaj = comp.p_all
                    e = "1.00000"
                    bpa = "000.000"
                    type = "1"
                    if not comp._fixed[3]:
                        bmaj = "{}v".format(bmaj)
                    else:
                        bmaj = "{}".format(bmaj)
                elif len(comp.p_all) == 6:
                    # Jy, mas, mas, mas, -, deg
                    flux, x, y, bmaj, e, bpa = comp.p_all
                    if not comp._fixed[4]:
                        e = "{}v".format(e)
                    else:
                        e = "{}".format(e)
                    if not comp._fixed[5]:
                        bpa = "{}v".format((bpa-np.pi/2)/degree_to_rad)
                    else:
                        bpa = "{}".format((bpa-np.pi/2)/degree_to_rad)
                    if not comp._fixed[3]:
                        bmaj = "{}v".format(bmaj)
                    else:
                        bmaj = "{}".format(bmaj)
                    type = "1"
                else:
                    raise Exception
            elif isinstance(comp, DeltaComponent):
                flux, x, y = comp.p_all
                e = "1.00000"
                bmaj = "0.0000"
                bpa = "000.000"
                type = "0"
            else:
                raise Exception
            # mas
            r = np.hypot(x, y)
            # rad
            theta = np.arctan2(-x, -y)
            theta /= degree_to_rad

            if not comp._fixed[0]:
                flux = "{}v".format(flux)
            else:
                flux = "{}".format(flux)

            if comp._fixed[1]:
                r = "{}".format(r)
            else:
                r = "{}v".format(r)

            if comp._fixed[2]:
                theta = "{}".format(theta)
            else:
                theta = "{}v".format(theta)

            fo.write("{} {} {} {} {} {} {} {} 0\n".format(flux, r, theta,
                                                          bmaj, e, bpa, type,
                                                          freq_hz))


def append_component_to_difmap_model(comp, out_fname, freq_hz):
    """
    :param comp:
        Instance of ``Component``.
    :param fname:
        File with difmap model.
    """
    with open(out_fname, "a") as fo:
        if isinstance(comp, EGComponent):
            if comp.size == 4:
                # Jy, mas, mas, mas
                flux, x, y, bmaj = comp.p
                e = "1.00000"
                bpa = "000.000"
                type = "1"
                bmaj = "{}v".format(bmaj)
            elif comp.size == 6:
                # Jy, mas, mas, mas, -, deg
                flux, x, y, bmaj, e, bpa = comp.p
                e = "{}v".format(e)
                bpa = "{}v".format((bpa - np.pi / 2) / degree_to_rad)
                bmaj = "{}v".format(bmaj)
                type = "1"
            else:
                raise Exception
        elif isinstance(comp, DeltaComponent):
            flux, x, y = comp.p
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type = "0"
        else:
            raise Exception
        # mas
        r = np.hypot(x, y)
        # rad
        theta = np.arctan2(-x, -y)
        theta /= degree_to_rad
        fo.write("{}v {}v {}v {} {} {} {} {} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type,
                                                         freq_hz))


def create_difmap_file_from_single_component(comp, out_fname, freq_hz):
    """
    :param comp:
        (flux, ra, dec[, bmaj, e, bpa]) - [Jy, mas, mas, mas, -, deg]

    """

    with open(out_fname, "a") as fo:
        if len(comp) == 3:
            flux, ra, dec = comp
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type = "0"
        elif len(comp) == 4:
            flux, ra, dec, bmaj = comp
            e = "1.00000"
            bpa = "000.000"
            type = "1"
            bmaj = "{:.5f}v".format(bmaj)
        elif len(comp) == 6:
            flux, ra, dec, bmaj, e, bpa = comp
            e = "{:.5f}v".format(e)
            bpa = "{:.5f}v".format((bpa - np.pi / 2) / degree_to_rad)
            bmaj = "{:.5f}v".format(bmaj)
            type = "1"
        else:
            raise Exception
        # mas
        r = np.hypot(ra, dec)
        # rad
        theta = np.rad2deg(np.arctan2(ra, dec))
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        fo.write("{:.5f}v {:.5f}v {:.5f}v {} {} {} {} {:.2E} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type,
                                                         freq_hz))


def difmap_model_flux(mdl_file):
    """
    Returns flux of the difmap model.

    :param mdl_file:
        Path to difmap model file.
    :return:
        Sum of all component fluxes [Jy].
    """
    comps = import_difmap_model(mdl_file)
    fluxes = [comp.p[0] for comp in comps]
    return sum(fluxes)


def find_ids(comps0, comps1):
    assert len(comps0) == len(comps1)
    ids = list()
    for comp in comps0:
        ids.append(comps1.index(comp))
    return ids


def sort_components_by_distance_from_cj(mdl_path, freq_hz, n_check_for_core=2,
                                        outpath=None, perc_distant=75,
                                        only_indicate=False):
    """
    Function that re-arrange components in a such a way that closest to "counter
    jet" components goes first. If components were already sorted by distance
    from "counter-jet" then does nothing and return ``False``.

    :param mdl_path:
        Path to difmap-format model file.
    :param freq_hz:
        Frequency in Hz.
    :param n_check_for_core: (optional)
        Number of closest to phase center components to check for being on "cj"
        side. (default: ``2``)
    :param outpath: (optional)
        Path to save re-arranged model. If ``None`` then re-write ``mdl_path``.
        (default: ``None``)
    :param perc_distant: (optional)
        Percentile of the component's distance distribution to use as border
        line that defines "distant" component. Position of such components is
        then used to compare polar angles of "cj-candidates". (default: ``75``)
    :param only_indicate: (optional)
        Boolean - ust indicate by the returned value or also do re-arrangement
        of components? (default: ``False``)
    :return:
        Boolean if there any "cj"-components that weren't sorted by distance
        from "CJ".
    """
    mdl_dir, mdl_fname = os.path.split(mdl_path)
    comps = import_difmap_model(mdl_fname, mdl_dir)
    comps = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))
    comps_c = import_difmap_model(mdl_fname, mdl_dir)
    r = [np.hypot(comp.p[1], comp.p[2]) for comp in comps]
    dist = np.percentile(r, perc_distant)
    # Components that are more distant from phase center than ``perc_distant``
    # of components
    remote_comps = [comp for r_, comp in zip(r, comps) if r_ > dist]
    x_mean = np.mean([-comp.p[1] for comp in remote_comps])
    y_mean = np.mean([-comp.p[2] for comp in remote_comps])
    theta_remote = np.arctan2(x_mean, y_mean)/degree_to_rad

    found_cj_comps = list()
    for i in range(0, n_check_for_core):
        comp = comps[i]
        # Don't count EG component
        if len(comp) == 6:
            continue
        # This checks that cj component have different PA
        if abs(np.arctan2(-comp.p[1], -comp.p[2])/degree_to_rad -
               theta_remote) > 90.:
            print("Found cj components - {}".format(comp))
            found_cj_comps.append(comp)

    result_comps = list()
    if found_cj_comps:
        for comp in found_cj_comps:
            comps.remove(comp)

        for comp in sorted(found_cj_comps,
                           key=lambda x: np.hypot(x.p[1], x.p[2])):
            result_comps.append(comp)

        # Now check if cj-components were not first ones
        is_changed = False
        sorted_found_cj_comps = sorted(found_cj_comps,
                                       key=lambda x: np.hypot(x.p[1], x.p[2]))
        for comp1, comp2 in zip(comps_c[:len(sorted_found_cj_comps)],
                                sorted_found_cj_comps[::-1]):
            if comp1 != comp2:
                is_changed = True
        # If all job was already done just return
        if not is_changed:
            print("Components are already sorted")
            return False

    for comp in comps:
        result_comps.append(comp)

    if outpath is None:
        outpath = mdl_path
    if not only_indicate:
        export_difmap_model(result_comps, outpath, freq_hz)
    return bool(found_cj_comps)


def sum_components(comp0, comp1, type="eg"):
    print("Summing components {} and {}".format(comp0, comp1))
    flux = comp0.p[0] + comp1.p[0]
    x = (comp0.p[0] * comp0.p[1] + comp1.p[0] * comp1.p[1]) / flux
    y = (comp0.p[0] * comp0.p[2] + comp1.p[0] * comp1.p[2]) / flux
    bmaj = (comp0.p[0] * comp0.p[3] + comp1.p[0] * comp1.p[3]) / flux
    if type == "cg":
        component = CGComponent(flux, x, y, bmaj)
    elif type == "eg":
        dx = comp0.p[1]-comp1.p[1]
        dy = comp0.p[2]-comp1.p[2]
        bpa = 180*np.arctan(dx/dy)/np.pi
        e = 0.5*(comp0.p[3]+comp1.p[3])/np.hypot(comp0.p[1]-comp1.p[1], comp0.p[2]-comp1.p[2])
        component = EGComponent(flux, x, y, bmaj, e, bpa)
    else:
        raise Exception
    print("Sum = {}".format(component))
    return component


def component_joiner_serial(difmap_model_file, beam_size, freq_hz,
                            distance_to_join_max=1.0, outname=None, new_type="eg"):
    joined = False
    comps = import_difmap_model(difmap_model_file)
    print("Len = {}".format(len(comps)))
    new_comps = list()
    skip_next = False
    for i, comp0 in enumerate(comps):
        print("{} and {}".format(i, i+1))
        if skip_next:
            skip_next = False
            print ("Skipping {}th component".format(i))
            continue
        try:
            comp1 = comps[i+1]
        except IndexError:
            new_comps.append(comp0)
            print("Writing component {} and exiting".format(i))
            break
        print("Fluxes: {} and {}".format(comp0.p[0], comp1.p[0]))

        if i != 0:
            distance_before = max(np.hypot(comp0.p[1]-comps[i-1].p[1], comp0.p[2]-comps[i-1].p[2]),
                                  np.hypot(comp1.p[1]-comps[i-1].p[1], comp1.p[2]-comps[i-1].p[2]))
        else:
            distance_before = 10**10
        print("Distance before = {}".format(distance_before))

        distance_between = np.hypot(comp0.p[1]-comp1.p[1],
                                    comp0.p[2]-comp1.p[2])
        print("Distance between = {}".format(distance_between))

        if i < len(comps)-2:
            distance_after = max(np.hypot(comp0.p[1] - comps[i + 2].p[1],
                                          comp0.p[2] - comps[i + 2].p[2]),
                                 np.hypot(comp1.p[1] - comps[i + 2].p[1],
                                          comp1.p[2] - comps[i + 2].p[2]))
        else:
            distance_after = 10000.0
        print("Distance after = {}".format(distance_after))

        if distance_before > beam_size and\
                        distance_after > beam_size and\
                        distance_between < distance_to_join_max*beam_size and\
                len(comp0) == len(comp1):
            joined = True
            new_comps.append(sum_components(comp0, comp1, type=new_type))
            skip_next = True
        else:
            new_comps.append(comp0)
            skip_next = False

    if outname is None:
            outname = difmap_model_file

    export_difmap_model(new_comps, outname, freq_hz)
    return joined


def remove_furthest_component(difmap_model_file, freq_hz, outname=None):
    comps = import_difmap_model(difmap_model_file)
    # Sort by distance
    furthest_comp = sorted(comps, key=lambda x: np.hypot(x.p[1], x.p[2]))[-1]

    new_comps = [comp for comp in comps if comp != furthest_comp]
    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)
    return True


def remove_small_component(difmap_model_file, freq_hz, size_limit=0.01,
                           outname=None):
    comps = import_difmap_model(difmap_model_file)
    small_comps = [comp for comp in comps if (len(comp.p) ==4 and
                                              comp.p[3] < size_limit)]
    if not small_comps:
        return False
    new_comps = [comp for comp in comps if comp not in small_comps]
    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)
    return True

# FIXME: Check if it works to ``DeltaComponent``!
def transform_component(difmap_model_file, new_type, freq_hz, comp_id=0,
                        outname=None, **kwargs):
    """
    Change component type in difmap model file.

    :param difmap_model_file:
        Path to difmap file with model.
    :param new_type:
        String that defines what is the new component type (delta, cg, eg).
    :param freq_hz:
        Frequency in Hz.
    :param comp_id: (optional)
        Number (ID) of component to transform. ``0`` means transform first
        component (core). (default: ``0``)
    :param outname: (optional)
        Where to save the result. If ``None`` then rewrite
        ``difmap_model_file``.
    :param kwargs: (optional)
        Arguments needed for transforming simple component to more complex.
    """
    if new_type not in ("delta", "cg", "eg"):
        raise Exception
    comps = import_difmap_model(difmap_model_file)
    comp_to_transform = comps[comp_id]
    new_comps = list()
    for comp in comps[:comp_id]:
        new_comps.append(comp)
    if new_type == "delta":
        new_comps.append(comp_to_transform.to_delta())
    elif new_type == "cg":
        new_comps.append(comp_to_transform.to_cirular(**kwargs))
    else:
        new_comps.append(comp_to_transform.to_elliptic(**kwargs))
    for comp in comps[comp_id+1:]:
        new_comps.append(comp)

    if outname is None:
        outname = difmap_model_file
    export_difmap_model(new_comps, outname, freq_hz)



def tb_gaussian(flux_jy, bmaj_mas, freq_ghz, z=0, bmin_mas=None, D=1.0):
    """
    Brightness temperature (optionally corrected for redshift and Doppler
    factor) in K.
    """
    k = 1.38*10**(-16)
    c = 29979245800.0
    mas_to_rad = 4.8481368*1E-09

    bmaj_mas *= mas_to_rad
    if bmin_mas is None:
        bmin_mas = bmaj_mas
    else:
        bmin_mas *= mas_to_rad
    freq_ghz *= 10**9
    flux_jy *= 10**(-23)
    return 2.*np.log(2)*(1.+z)*flux_jy*c**2/(freq_ghz**2*np.pi*k*bmaj_mas*bmin_mas*D)


def sigma_tb_gaussian(sigma_flux_jy, sigma_d_mas, flux_jy, d_mas, freq_ghz, z=0, D=1.0):
    """
    Brightness temperature (optionally corrected for redshift and Doppler
    factor) in K.
    """
    k = 1.38*10**(-16)
    c = 29979245800.0
    mas_to_rad = 4.8481368*1E-09

    d_mas *= mas_to_rad
    sigma_d_mas *= mas_to_rad
    freq_ghz *= 10**9
    flux_jy *= 10**(-23)
    sigma_flux_jy *= 10**(-23)
    return 2.*np.log(2)*(1.+z)*c**2/(freq_ghz**2*np.pi*k*d_mas**2*D)*np.sqrt(sigma_flux_jy**2 + (2*flux_jy/d_mas)**2*sigma_d_mas**2)


def convert_bbox_to_radec_ranges(wcs, blc, trc):
    """
    Given BLC, TRC in coordinates of numpy array return RA, DEC ranges.
    :param wcs:
        Instance of ``astropy.wcs.wcs.WCS``.
    :return:
        RA_min, RA_max, DEC_min, DEC_max
    """
    blc_deg = wcs.all_pix2world(blc[0], blc[1], 1)
    trc_deg = wcs.all_pix2world(trc[0], trc[1], 1)

    ra_max = blc_deg[0]
    ra_min = trc_deg[0]
    dec_min = blc_deg[1]
    dec_max = trc_deg[1]

    return ra_min, ra_max, dec_min, dec_max


def convert_radec_ranges_to_bbox(wcs, ra_min, ra_max, dec_min, dec_max):
    blc_deg_0,  blc_deg_1 = wcs.all_world2pix(ra_max, dec_min, 1)
    trc_deg_0,  trc_deg_1 = wcs.all_world2pix(ra_min, dec_max, 1)
    return (int(round(float(blc_deg_0), 0)), int(round(float(blc_deg_1), 0))),\
           (int(round(float(trc_deg_0), 0)), int(round(float(trc_deg_1), 0)))


def components_info(fname, mdl_fname, dmap_size, PA=None, dmap_name=None,
                    out_path=None, stokes="I", show_difmap_output=True,
                    freq_ghz=15.4, size_error_coefficient=0.2):
    # Conversion coefficient from degrees to mas
    deg2mas = 3.6*10**6

    # size_errors = find_size_errors_using_chi2(mdl_fname, fname, nmodelfit=100, use_selfcal=False)
    # size_errors_single = list()
    # for entry in size_errors:
    #     size_errors_single.append(0.5*(entry[0] + entry[1]))

    columns = ["flux", "ra", "dec", "original_major", "original_minor",  "major", "minor", "posangle", "type", "snr", "rms", "flux_err",
               "r_err", "major_unresolved", "minor_unresolved", "major_err", "minor_err",
               # "PA_core", "size_across_PA_core", "resolved_across_PA_core", "size_across_PA_core_err",
               "Tb", "Tb_err", "upper_limit_Tb", "beam"]
    df_dict = dict()
    for column in columns:
        df_dict[column] = list()

    if out_path is None:
        out_path = os.getcwd()

    # Remove old log-file if any
    difmap_log_files = glob.glob("difmap.log*")
    for difmap_log_file in difmap_log_files:
        try:
            os.unlink(difmap_log_file)
        except OSError:
            pass

    stamp = datetime.datetime.now()
    command_file = os.path.join(out_path, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + fname + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + mdl_fname + "\n")
    difmapout.write("mapsize " + str(dmap_size[0]) + "," + str(dmap_size[1]) + "\n")
    difmapout.write("uvw 0,-2\n")
    difmapout.write("invert\n")
    # difmapout.write("print \"postfitrms =\", uvstat(rms)/sqrt(uvstat(nvis))\n")
    difmapout.write("print \"postfitrms =\", uvstat(rms)\n")
    difmapout.write("print \"nvis =\", uvstat(nvis)\n")
    # difmapout.write("print \"maxampl = \",vis_stats(amplitude)(6)\n")
    # Obtain dirty image
    if dmap_name is None:
        dmap_name = "dirty_residuals_map.fits"
    difmapout.write("wdmap " + os.path.join(out_path, dmap_name) + "\n")

    # Get beam info
    difmapout.write("restore\n")

    difmapout.write("exit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Obtain beam info from lof-file:)
    with open("difmap.log", "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Restoring with beam:" in line][-1]
    bmaj = float(line.split()[4])
    bmin = float(line.split()[6])
    if bmaj < bmin:
        bmaj, bmin = bmin, bmaj
    bpa = np.deg2rad(float(line.split()[8]))
    beam_area_mas = np.pi*bmaj*bmin/(4*np.log(2))
    print("Beam area (mas^2) = ", beam_area_mas)
    print("Beam area (pix^2) = ", beam_area_mas/0.01)
    beam_area_pix = beam_area_mas/0.01
    beam_circ = np.sqrt(bmaj*bmin)
    print("BEAM CIRC = ", beam_circ)



    line = [line for line in lines if "! postfitrms" in line][-1]
    postfitrms = float(line.strip().split(" ")[-1])
    print("POSTFITRMS = ", postfitrms)
    line = [line for line in lines if "! nvis" in line][-1]
    nvis = int(line.strip().split(" ")[-1])
    print("NVIS = ", nvis)

    sum_sq_diff_uv = postfitrms**2*nvis
    print("SUM SQ DIFF UV = ", sum_sq_diff_uv)

    std_from_vis = postfitrms/np.sqrt(2*nvis)


    if PA is None:
        theta_beam = np.sqrt(bmaj*bmin)
    else:
        theta_beam = bmaj*bmin*np.sqrt((1+np.tan(PA-bpa)**2)/(bmin**2+bmaj**2*np.tan(PA-bpa)**2))

    # Remove Difmap command and log-file
    os.unlink(command_file)
    # os.unlink("difmap.log")

    hdulist = pf.open(os.path.join(out_path, dmap_name))
    hdulist.verify("ignore")
    header = hdulist[0].header

    # Build coordinate system
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial
    # Make offset coordinates
    wcs.wcs.crval = 0., 0.
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'mas', 'mas'
    wcs.wcs.cdelt = (wcs.wcs.cdelt * u.deg).to(u.mas)

    dimage = hdulist[0].data.squeeze()
    npix = dimage.shape[0]
    npix *= npix
    print("NPIX = ", npix)

    parseval_rms_per_pix = np.sqrt(sum_sq_diff_uv/(npix/beam_area_pix))
    print("PARSEVAL RMS per beam = ", parseval_rms_per_pix)


    if postfitrms > 0.1:
        postfitrms_dimage = np.std(dimage)
        if postfitrms_dimage < postfitrms:
            postfitrms = postfitrms_dimage


    # Characteristic size of the beam in pixels
    beam_pxl = int(np.sqrt(beam_area_mas) / dmap_size[1])
    print("Beam size (pxl) = ", beam_pxl)

    # Read components from difmap dirty image CC-table
    comps = hdulist["AIPS CC "].data
    # comps = pf.getdata(os.path.join(out_path, dmap_name), ext=("AIPS CC ", 1))
    sizes_lim = list()
    unresolved_maj = None
    unresolved_min = None
    upper_limit_Tb = None

    x = np.arange(header["NAXIS1"])
    y = np.arange(header["NAXIS2"])
    X, Y = np.meshgrid(x, y)
    # Arrays of RA, DEC for all pixels (just in case)
    # ra_pix_all, dec_pix_all = wcs.wcs_pix2world(X, Y, 0)

    for comp in comps:



        # Assume core in the phase center
        # PA_core = 0
        # theta_beam_across_component_PA = size_minor*size_major*np.sqrt((1+np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2)/(bmin**2+bmaj**2*np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2))

        # RA, DEC of component center
        ra = comp[1]*deg2mas
        dec = comp[2]*deg2mas
        df_dict["original_major"].append(comp["MAJOR AX"]*deg2mas)
        df_dict["original_minor"].append(comp["MINOR AX"]*deg2mas)
        # FIXME: Currently it is from phase center - not from the core!
        # PA_core = np.arctan2(ra, dec)
        ra_pix, dec_pix = wcs.wcs_world2pix(ra.reshape(-1, 1), dec.reshape(-1, 1), 0)
        ra_pix = ra_pix[0][0]
        dec_pix = dec_pix[0][0]

        mask_radius = 3*beam_pxl
        if comp["TYPE OBJ"] == 1.0 and comp["POSANGLE"] == 0.0:
            size = comp["MAJOR AX"]*deg2mas/0.1
            mask_radius = np.hypot(mask_radius, size)
        elif comp["TYPE OBJ"] == 1.0 and comp["POSANGLE"] != 0.0:
            size = 0.5*(comp["MAJOR AX"]*deg2mas + comp["MINOR AX"]*deg2mas)/0.1
            mask_radius = np.hypot(mask_radius, size)
        else:
            size = 0.0

        # Create mask of size N*beam_pxl with center in (ra_pix, dec_pix)
        mask = np.sqrt((X-ra_pix)**2+(Y-dec_pix)**2) < mask_radius


        rms_dimage_all = np.std(dimage)
        rms_dimage_all = mad_std(dimage)
        print("RMS of ALL DIMAGE (Jy/beam) = ", rms_dimage_all)

        rms_lim = np.std(dimage[mask])
        # print("RMS DIMAGE (Jy/beam) = ", rms)
        # rms_lim = rms_dimage_all
        # rms = postfitrms
        # print("RMS UV (Jy) = ", rms)
        # TODO: This was before and gave good results
        # rms = std_from_vis
        rms = rms_lim


        # For CG
        if comp["TYPE OBJ"] == 1.0 and comp["MAJOR AX"] == comp["MINOR AX"]:
            size = comp["MAJOR AX"]*deg2mas
            print("SIZE = ", size)
            comp_area_mas = np.pi*size**2/(4*np.log(2))
            comp_area_mas_beam_convolved = np.pi*np.hypot(size, beam_circ)**2/(4*np.log(2))
            # Calculate first time for SNR
            peak_flux = comp["FLUX"] * beam_area_mas / (beam_area_mas + comp_area_mas)
            peak_flux = np.hypot(peak_flux, rms)
            print("flux (Jy) = ", comp["FLUX"])
            print("peak flux (Jy/beam) = ", peak_flux)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # This makes few limits, but they are kinda small
            # SNR = peak_flux / rms
            # This makes more limits for NGC315, but besides this all is OK!
            # SNR = peak_flux / sigma_peak
            SNR_lim = peak_flux / rms_lim
            # print("SNR = ", SNR)
            if SNR_lim > 1:
                # factor 2 - for uniform weighting, 1 - for natural
                size_lim = theta_beam*np.sqrt(np.log(SNR_lim/(SNR_lim-1))*4*np.log(2)/np.pi)
                # size_lim = theta_beam_across_component_PA*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            else:
                size_lim = theta_beam/2
                # size_lim = theta_beam_across_component_PA/2
                SNR_lim = 1
            print("SIZE LIM (MAS) = ", size_lim)
            # If it is a limit - re-calculate peak flux
            unresolved_min = None
            if size < size_lim:
                unresolved_maj = True
                size = size_lim
                upper_limit_Tb = True
            else:
                unresolved_maj = False
                upper_limit_Tb = False

            # Re-calculate if sizes are limits
            comp_area_mas = np.pi*size**2/(4*np.log(2))

            # TODO: Should I re-calculate peak_flux for unresolved components? It doesn't change actually.
            peak_flux = comp["FLUX"] * beam_area_mas / (beam_area_mas + comp_area_mas)
            peak_flux = np.hypot(peak_flux, rms)

            # Lee
            # FIXME: This makes size error of large faint components ~ size
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # Schitzel
            # sigma_peak = rms

            # An
            sigma_tot_flux = rms*(comp_area_mas_beam_convolved/beam_area_mas)
            # Lee and Schitzel
            # sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)


            # An
            # This is not good for the extended components, where the size error could be larger than beam. However, we
            # include fractional size error in fitting!
            # sigma_size = beam_circ/SNR
            # sigma_size = np.hypot(beam_circ, size)/SNR
            # sigma_size = beam_circ*sigma_peak/peak_flux

            # FIXME: Conventional size error
            sigma_size = size_error_coefficient*np.hypot(size, beam_circ)*(sigma_peak/peak_flux)
            sigma_pos = 0.5*sigma_size
            # sigma_pos = 0.5*beam_circ*sigma_peak/peak_flux
            # sigma_size = size*sigma_peak/peak_flux
            print("sigma size (mas) = ", sigma_size)
            tb = tb_gaussian(comp["FLUX"], size, freq_ghz)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, sigma_size, comp["FLUX"], size, freq_ghz)

            # tb_values = tb_gaussian(np.random.normal(comp["FLUX"], sigma_tot_flux, 1000),
            #                         np.random.normal(size, sigma_size, 1000),
            #                         freq_ghz)
            # sigma_tb = np.std(tb_values)

            # size_across_PA_core = size
            # resolved_across_PA_core = unresolved_maj
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux

            df_dict["major"].append(size)
            df_dict["minor"].append(size)
            df_dict["major_err"].append(sigma_size)
            df_dict["minor_err"].append(sigma_size)
            df_dict["posangle"].append(None)
            df_dict["type"].append(1)
            df_dict["r_err"].append(sigma_pos)

        # For EG
        elif comp["TYPE OBJ"] == 1.0 and comp["MAJOR AX"] != comp["MINOR AX"]:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   EG   !!!!!!!!!!!!!!!!!!!!!!!")
            upper_limit_Tb = False
            size_major = comp["MAJOR AX"]*deg2mas
            size_minor = comp["MINOR AX"]*deg2mas

            # Assume core at the phase center
            # PA_core = 0
            # Component size in direction perpendicular to the component PA
            # size_across_PA_core = size_minor*size_major*np.sqrt((1+np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2)/(bmin**2+bmaj**2*np.tan(PA_core+np.pi/2-np.deg2rad(comp["POSANGLE"]))**2))

            comp_area_mas = np.pi*size_major*size_minor/(4*np.log(2))
            comp_area_mas_beam_convolved = np.pi*np.hypot(size_minor, beam_circ)*np.hypot(size_major, beam_circ)/(4*np.log(2))

            # Calculate first time for SNR
            peak_flux = comp["FLUX"] * beam_area_mas / (beam_area_mas + comp_area_mas)
            peak_flux = np.hypot(peak_flux, rms)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # SNR = peak_flux / rms
            # This makes more limits for NGC315, but besides this all is OK!
            # SNR = peak_flux / sigma_peak
            SNR_lim = peak_flux / rms_lim

            # Removed ``2`` coefficient
            theta_beam_along_major = bmaj*bmin*np.sqrt((1+np.tan(np.deg2rad(comp["POSANGLE"])-bpa)**2)/(bmin**2 + bmaj**2*np.tan(np.deg2rad(comp["POSANGLE"])-bpa)**2))
            theta_beam_along_minor = bmaj*bmin*np.sqrt((1+np.tan(np.deg2rad(comp["POSANGLE"]+90)-bpa)**2)/(bmin**2 + bmaj**2*np.tan(np.deg2rad(comp["POSANGLE"]+90)-bpa)**2))

            if SNR_lim > 1:
                size_lim_major = theta_beam_along_major*np.sqrt(np.log(SNR_lim/(SNR_lim-1))*4*np.log(2)/np.pi)
                size_lim_minor = theta_beam_along_minor*np.sqrt(np.log(SNR_lim/(SNR_lim-1))*4*np.log(2)/np.pi)
            else:
                size_lim_major = theta_beam/2
                size_lim_minor = theta_beam/2
                SNR_lim = 1

            # If it is a limit - re-calculate peak flux
            if size_major < size_lim_major:
                size_major = size_lim_major
                unresolved_maj = True
                upper_limit_Tb = True
            else:
                unresolved_maj = False

            if size_minor < size_lim_minor:
                size_minor = size_lim_minor
                unresolved_min = True
                upper_limit_Tb = True
            else:
                unresolved_min = False

            # Re-calculate if sizes are limits
            comp_area_mas = np.pi*size_major*size_minor/(4*np.log(2))

            # TODO: Should I re-calculate peak_flux for unresolved components? It doesn't change actually.
            peak_flux = comp["FLUX"] * beam_area_mas / (beam_area_mas + comp_area_mas)
            peak_flux = np.hypot(peak_flux, rms)

            # Lee
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # Schitzel
            # sigma_peak = rms

            # An
            sigma_tot_flux = rms*(comp_area_mas_beam_convolved/beam_area_mas)

            # Lee and Schitzel
            # sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)

            # An
            # This is not good for the extended components, where the size error could be larger than beam. However, we
            # include fractional size error in fitting!
            # sigma_size_major = bmaj/SNR
            # sigma_size_minor = bmin/SNR
            # sigma_size_minor = beam_circ*sigma_peak/peak_flux
            # sigma_size_major = beam_circ*sigma_peak/peak_flux
            # sigma_size_major = np.hypot(beam_circ, size_major)/SNR
            # sigma_size_minor = np.hypot(beam_circ, size_minor)/SNR

            # FIXME: Conventional size error
            sigma_size_major = size_error_coefficient*np.hypot(size_major, beam_circ)*sigma_peak/peak_flux
            sigma_size_minor = size_error_coefficient*np.hypot(size_minor, beam_circ)*sigma_peak/peak_flux
            # sigma_size_major = size_major*sigma_peak/peak_flux
            # sigma_size_minor = size_minor*sigma_peak/peak_flux
            sigma_pos = 0.5*(sigma_size_minor + sigma_size_major)
            # sigma_pos = 0.5*beam_circ*sigma_peak/peak_flux

            tb = tb_gaussian(comp["FLUX"], size_major, freq_ghz, bmin_mas=size_minor)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, 0.5*(sigma_size_minor+sigma_size_major), comp["FLUX"], 0.5*(size_minor+size_major), freq_ghz)

            # Assume core at the phase center
            PA_core = 0
            theta_beam_across_PA_core = bmaj*bmin*np.sqrt((1+np.tan(PA_core-bpa)**2)/(bmin**2 + bmaj**2*np.tan(PA_core-bpa)**2))
            # size_lim_across_PA_core = theta_beam_across_PA_core*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            # if size_across_PA_core > size_lim_across_PA_core:
            #     resolved_across_PA_core = True
            # else:
            #     resolved_across_PA_core = False
            #     size_across_PA_core = size_lim_across_PA_core
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux


            df_dict["major"].append(size_major)
            df_dict["minor"].append(size_minor)
            df_dict["posangle"].append(np.deg2rad(comp["POSANGLE"]))
            df_dict["type"].append(2)
            df_dict["r_err"].append(sigma_pos)
            df_dict["major_err"].append(sigma_size_major)
            df_dict["minor_err"].append(sigma_size_minor)

        # For delta
        elif comp["TYPE OBJ"] == 0.0:
            unresolved_maj = None
            unresolved_min = None
            upper_limit_Tb = True
            peak_flux = comp["FLUX"]
            peak_flux = np.hypot(peak_flux, rms)
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # SNR = peak_flux/rms
            # This makes more limits for NGC315, but besides this all is OK!
            # SNR = peak_flux/sigma_peak
            SNR_lim = peak_flux / rms_lim
            size_lim = theta_beam*np.sqrt(np.log(SNR_lim/(SNR_lim-1))*4*np.log(2)/np.pi)
            # size_lim = theta_beam_across_component_PA*np.sqrt(np.log(SNR/(SNR-1))*4*np.log(2)/np.pi)
            size = size_lim
            comp_area_mas_beam_convolved = np.pi*np.hypot(size, beam_circ)*np.hypot(size, beam_circ)/(4*np.log(2))

            # Lee
            sigma_peak = rms*np.sqrt(1 + peak_flux/rms)
            # Schitzel
            # sigma_peak = rms

            # An
            # sigma_tot_flux = rms
            sigma_tot_flux = rms*(comp_area_mas_beam_convolved/beam_area_mas)

            # Lee and Schitzel
            # sigma_tot_flux = sigma_peak*np.sqrt(1 + comp["FLUX"]**2/peak_flux**2)

            # An
            # This is not good for the extended components, where the size error could be larger than beam. However, we
            # include fractional size error in fitting!
            # sigma_size = beam_circ/SNR
            # sigma_size = beam_circ*sigma_peak/peak_flux

            # FIXME: Conventional size error
            sigma_size = size_error_coefficient*np.hypot(size, beam_circ)*sigma_peak/peak_flux
            # sigma_size = size*sigma_peak/peak_flux
            sigma_pos = 0.5*sigma_size
            # sigma_pos = 0.5*beam_circ*sigma_peak/peak_flux

            tb = tb_gaussian(comp["FLUX"], size, freq_ghz)
            sigma_tb = sigma_tb_gaussian(sigma_tot_flux, sigma_size, comp["FLUX"], size, freq_ghz)
            # size_across_PA_core = size
            # resolved_across_PA_core = None
            # size_across_PA_core_err = size_across_PA_core*sigma_peak/peak_flux

            df_dict["major"].append(size)
            df_dict["minor"].append(size)
            df_dict["posangle"].append(None)
            df_dict["type"].append(0)
            df_dict["r_err"].append(sigma_pos)
            df_dict["major_err"].append(sigma_size)
            df_dict["minor_err"].append(sigma_size)
        else:
            raise Exception

        df_dict["snr"].append(SNR_lim)
        df_dict["rms"].append(rms)
        df_dict["flux"].append(comp["FLUX"])
        df_dict["ra"].append(ra)
        df_dict["dec"].append(dec)
        df_dict["flux_err"].append(sigma_tot_flux)
        df_dict["major_unresolved"].append(unresolved_maj)
        df_dict["minor_unresolved"].append(unresolved_min)
        # df_dict["PA_core"].append(PA_core)
        # df_dict["resolved_across_PA_core"].append(resolved_across_PA_core)
        # df_dict["size_across_PA_core"].append(size_across_PA_core)
        # df_dict["size_across_PA_core_err"].append(size_across_PA_core_err)
        df_dict["Tb"].append(tb)
        df_dict["Tb_err"].append(sigma_tb)
        df_dict["upper_limit_Tb"].append(upper_limit_Tb)
        df_dict["beam"].append(beam_circ)

    # try:
    #     os.unlink(os.path.join(out_path, "dirty_residuals_map.fits"))
    # except FileNotFoundError:
    #     pass

    return pd.DataFrame.from_dict(df_dict)


# TODO: Add iteration till chi-squared is increasing for some number of
# iterations
# TODO: Add estimating rms in dirty residual map at the components positions
def modelfit_difmap(fname, mdl_fname, out_fname, niter=50, stokes='i',
                    path=None, mdl_path=None, out_path=None,
                    show_difmap_output=False,
                    save_dirty_residuals_map=False,
                    dmap_name=None, dmap_size=(1024, 0.1)):
    """
    Modelfit self-calibrated uv-data in difmap.

    :param fname:
        Filename of uv-data to modelfit.
    :param mdl_fname:
        Filename with model.
    :param out_fname:
        Filename with output file with model.
    :param stokes: (optional)
        Stokes parameter 'i', 'q', 'u' or 'v'. (default: ``i``)
    :param path: (optional)
        Path to uv-data to modelfit. If ``None`` then use current directory.
        (default: ``None``)
    :param mdl_path: (optional)
        Path file with model. If ``None`` then use current directory.
        (default: ``None``)
    :param out_path: (optional)
        Path to file with CCs. If ``None`` then use ``path``.
        (default: ``None``)
    :param save_dirty_residuals_map: (optional)
        Boolean. Whever to save dirty residuals map? (default: ``False``)
    :param dmap_name: (optional)
        Name of the FITS file (in ``out_path``) where to save dirty residual
        map. If ``None`` then dirty_residuals_map.fits``. (default: ``None``)
    :param dmap_size: (optional)
        Size of the dirty residuals map. (default: ``(1024, 0.1)``
    """
    if path is None:
        path = os.getcwd()
    if mdl_path is None:
        mdl_path = os.getcwd()
    if out_path is None:
        out_path = os.getcwd()

    # Remove old log-file if any
    try:
        os.unlink(os.path.join(os.getcwd(), "difmap.log"))
    except OSError:
        pass

    stamp = datetime.datetime.now()
    command_file = os.path.join(out_path, "difmap_commands_{}".format(stamp.isoformat()))
    difmapout = open(command_file, "w")
    difmapout.write("observe " + os.path.join(path, fname) + "\n")
    difmapout.write("select " + stokes + "\n")
    difmapout.write("rmodel " + os.path.join(mdl_path, mdl_fname) + "\n")
    difmapout.write("modelfit " + str(niter) + "\n")
    difmapout.write("wmodel " + os.path.join(out_path, out_fname) + "\n")

    if save_dirty_residuals_map:
        difmapout.write("mapsize "+ str(2*dmap_size[0]) + "," + str(dmap_size[1])
                        + "\n")
        if dmap_name is None:
            dmap_name = "dirty_residuals_map.fits"
        difmapout.write("wdmap "+ os.path.join(out_path, dmap_name)+"\n")

    difmapout.write("exit\n")
    difmapout.close()

    # TODO: Use subprocess?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Get final reduced chi_squared
    try:
        log = os.path.join(os.getcwd(), "difmap.log")
        with open(log, "r") as fo:
            lines = fo.readlines()
        line = [line for line in lines if "Reduced Chi-squared=" in line][-1]
        rchisq = float(line.split(" ")[4].split("=")[1])
    except (FileNotFoundError, IndexError):
        rchisq = None

    # Remove command file
    os.unlink(command_file)

    return rchisq


def find_stat_of_difmap_model(dfm_model_file, uvfits, stokes="I", working_dir=None,
                              show_difmap_output=False, unselfcalibrated_uvfits=None,
                              nmodelfit=20, use_pselfcal=False, use_apselfcal=False,
                              out_dfm_model=None):

    if use_apselfcal and use_pselfcal:
        raise Exception("Only P or AP selfcal!")

    if unselfcalibrated_uvfits is None:
        uvfile = uvfits
    else:
        uvfile = unselfcalibrated_uvfits

    original_dir = os.getcwd()
    if working_dir is None:
        working_dir = os.getcwd()
    os.chdir(working_dir)

    # Find and remove all log-files
    previous_logs = glob.glob("difmap.log*")
    for log in previous_logs:
        os.unlink(log)

    # stamp = datetime.datetime.now()
    # command_file = os.path.join(working_dir, "difmap_commands_{}".format(stamp.isoformat()))
    # difmapout = open(command_file, "w")
    # difmapout.write("observe " + uvfile + "\n")
    # difmapout.write("select " + stokes + "\n")
    # difmapout.write("rmodel " + dfm_model_file + "\n")
    # if use_pselfcal:
    #     difmapout.write("selfcal\n")
    #     difmapout.write("modelfit {}\n".format(nmodelfit))
    #     difmapout.write("selfcal\n")
    # difmapout.write("modelfit {}\n".format(nmodelfit))
    # if out_dfm_model:
    #     difmapout.write("wmod {}\n".format(out_dfm_model))
    # difmapout.write("quit\n")
    # difmapout.close()
    #
    # # TODO: Use subprocess?
    # shell_command = "difmap < " + command_file + " 2>&1"
    # if not show_difmap_output:
    #     shell_command += " >/dev/null"
    # os.system(shell_command)

    from subprocess import Popen, PIPE

    # print(uvfile)
    # print(dfm_model_file)

    cmd = "observe " + uvfile + "\n"
    cmd += "select " + stokes + "\n"
    cmd += "rmodel " + dfm_model_file + "\n"
    if use_pselfcal:
        cmd += "selfcal\n"
        cmd += "modelfit {}\n".format(nmodelfit)
        cmd += "selfcal\n"
    if use_apselfcal:
        cmd += "selfcal true\n"
        cmd += "modelfit {}\n".format(nmodelfit)
        cmd += "selfcal true\n"
    cmd += "modelfit {}\n".format(nmodelfit)
    if out_dfm_model:
        cmd += "wmod {}\n".format(out_dfm_model)
    cmd += "quit\n"

    # print(cmd)

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)

    # Get final reduced chi_squared
    log = os.path.join(working_dir, "difmap.log")
    with open(log, "r") as fo:
        lines = fo.readlines()
    line = [line for line in lines if "Reduced Chi-squared=" in line][-1]
    rchisq = float(line.split(" ")[4].split("=")[1])

    line = [line for line in lines if "degrees of freedom" in line][-1]
    dof = int(float(line.split(" ")[-4]))

    # Remove command file
    # os.unlink(command_file)
    # os.unlink("difmap.log")
    os.chdir(original_dir)

    return {"rchisq": rchisq, "dof": dof}


# def find_stat_of_difmap_model_ap(dfm_model_file, uvfits, stokes="I", working_dir=None,
#                                  show_difmap_output=False, unselfcalibrated_uvfits=None,
#                                  nmodelfit=100, use_apselfcal=True, out_dfm_model=None):
#
#     if unselfcalibrated_uvfits is None:
#         uvfile = uvfits
#     else:
#         uvfile = unselfcalibrated_uvfits
#
#     original_dir = os.getcwd()
#     if working_dir is None:
#         working_dir = os.getcwd()
#     os.chdir(working_dir)
#
#     # Find and remove all log-files
#     previous_logs = glob.glob("difmap.log*")
#     for log in previous_logs:
#         os.unlink(log)
#
#     stamp = datetime.datetime.now()
#     command_file = os.path.join(working_dir, "difmap_commands_{}".format(stamp.isoformat()))
#     difmapout = open(command_file, "w")
#     difmapout.write("observe " + uvfile + "\n")
#     difmapout.write("select " + stokes + "\n")
#     difmapout.write("rmodel " + dfm_model_file + "\n")
#     if use_apselfcal:
#         difmapout.write("selfcal true\n")
#         difmapout.write("modelfit {}\n".format(nmodelfit))
#         difmapout.write("selfcal true\n")
#     difmapout.write("modelfit {}\n".format(nmodelfit))
#     if out_dfm_model:
#         difmapout.write("wmod {}\n".format(out_dfm_model))
#     difmapout.write("quit\n")
#     difmapout.close()
#
#     # TODO: Use subprocess?
#     shell_command = "difmap < " + command_file + " 2>&1"
#     if not show_difmap_output:
#         shell_command += " >/dev/null"
#     os.system(shell_command)
#
#     # Get final reduced chi_squared
#     log = os.path.join(working_dir, "difmap.log")
#     with open(log, "r") as fo:
#         lines = fo.readlines()
#     line = [line for line in lines if "Reduced Chi-squared=" in line][-1]
#     rchisq = float(line.split(" ")[4].split("=")[1])
#
#     line = [line for line in lines if "degrees of freedom" in line][-1]
#     dof = int(float(line.split(" ")[-4]))
#
#     # Remove command file
#     os.unlink(command_file)
#     # os.unlink("difmap.log")
#     os.chdir(original_dir)
#
#     return {"rchisq": rchisq, "dof": dof}


def find_size_errors_using_chi2(dfm_model_file, uvfits, working_dir=None,
                                delta_t_sec=10.0, gain_phase_tcoh=30.0, gain_amp_tcoh=60.0,
                                show_difmap_output=False, rho_IF = 0.75,
                                use_selfcal=True, nmodelfit_cycle=200,
                                freq=15.4E+09):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff_ph = gain_phase_tcoh/delta_t_sec
    rho_ph = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)
    # The same for amplitudes
    t_eff_amp = gain_amp_tcoh/delta_t_sec
    rho_amp = np.exp(-(delta_t_sec/gain_amp_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits, verify_option="ignore")
    all_stokes = uvdata.stokes
    if "RR" in all_stokes and "LL" in all_stokes:
        stokes = "I"
    else:
        if "RR" in all_stokes:
            stokes = "RR"
        else:
            stokes = "LL"
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    n_ant = len(uvdata.antennas)

    # Coefficient of correlation between gains in close IFs
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/10.
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff_ph
    n_eff_gain_amps = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff_amp
    n_eff_gain = n_eff_gain_amps + n_eff_gain_phases
    if stokes == "I":
        n_eff_gain *= 2

    import dlib
    # Just best rchisq (w/o any fit or self-cal)
    if use_selfcal:
        nmodelfit = 200
    else:
        nmodelfit = 0
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir,
                                          nmodelfit=nmodelfit, out_dfm_model="selfcaled.mdl",
                                          use_apselfcal=use_selfcal, use_pselfcal=False)
    rchisq0 = stat_dict["rchisq"]
    print("Initial rchisq = ", rchisq0)
    if use_selfcal:
        dof = stat_dict["dof"] - n_eff_gain
    else:
        dof = stat_dict["dof"]
    print("DoF = ", dof)
    delta_rchisq = 1.0/dof
    required_chisq = rchisq0 + delta_rchisq
    print("Required chi2 = ", required_chisq)
    # original_comps = import_difmap_model(dfm_model_file)
    original_comps = import_difmap_model("selfcaled.mdl")
    # original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    if not use_selfcal:
        comps = import_difmap_model(dfm_model_file)
    else:
        comps = import_difmap_model("selfcaled.mdl")

    errors = dict()

    for i, comp in enumerate(comps):
        index = 3

        # If size is fixed - skip that component
        if comp._fixed[index]:
            errors[i] = (0.0, 0.0)
            continue

        par0 = comp.p[index]

        def min_func_lower(delta):
            ccomp = copy.copy(comp)
            ccomp.set_value(par0 - delta, index)
            # Hack to fix size
            ccomp._fixed[3] = True
            new_comps = comps.copy()
            new_comps.insert(i, ccomp)
            new_comps.pop(i+1)
            export_difmap_model(new_comps, "dfm.mdl", freq)
            sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                              nmodelfit=nmodelfit_cycle, use_apselfcal=use_selfcal, use_pselfcal=False)
            print("For delta = {} mas the differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
            return (required_chisq - sdict["rchisq"])**2

        def min_func_upper(delta):
            ccomp = copy.copy(comp)
            ccomp.set_value(par0 + delta, index)
            # Hack to fix size
            ccomp._fixed[3] = True
            new_comps = comps.copy()
            new_comps.insert(i, ccomp)
            new_comps.pop(i+1)
            export_difmap_model(new_comps, "dfm.mdl", freq)
            sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                              nmodelfit=nmodelfit_cycle, use_apselfcal=use_selfcal, use_pselfcal=False)
            print("For delta = {} mas the differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
            return (required_chisq - sdict["rchisq"])**2

        # First find upper bound
        lower_bounds = [1E-4]
        upper_bounds = [10.0]
        n_fun_eval = 200
        delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

        print("===================")

        # First find upper bound
        lower_bounds = [1E-4]
        upper_bounds = [par0]
        n_fun_eval = 200
        delta_lower, _ = dlib.find_min_global(min_func_lower, lower_bounds, upper_bounds, n_fun_eval)

        print("Parameters = {} - {} + {}".format(par0, delta_lower, delta_upper))
        errors[i] = (delta_lower, delta_upper)
        # sys.exit(0)
    return errors


def find_flux_errors_using_chi2(dfm_model_file, uvfits, working_dir=None,
                                delta_t_sec=30.0, gain_phase_tcoh=30.0, gain_amp_tcoh=60.0,
                                show_difmap_output=False, nmodelfit=20, rho_IF = 0.75,
                                use_selfcal=True, freq=15.4E+09, nmodelfit_cycle=200):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff_ph = gain_phase_tcoh/delta_t_sec
    rho_ph = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)
    # The same for amplitudes
    t_eff_amp = gain_amp_tcoh/delta_t_sec
    rho_amp = np.exp(-(delta_t_sec/gain_amp_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits, verify_option="ignore")
    all_stokes = uvdata.stokes
    if "RR" in all_stokes and "LL" in all_stokes:
        stokes = "I"
    else:
        if "RR" in all_stokes:
            stokes = "RR"
        else:
            stokes = "LL"
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    n_ant = len(uvdata.antennas)

    # Coefficient of correlation between gains in close IFs
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/10.
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff_ph
    n_eff_gain_amps = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff_amp
    n_eff_gain = n_eff_gain_amps + n_eff_gain_phases
    if stokes == "I":
        n_eff_gain *= 2

    import dlib
    # Just best rchisq (w/o any fit or self-cal)
    if use_selfcal:
        nmodelfit = 200
    else:
        nmodelfit = 0
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir, nmodelfit=nmodelfit,
                                          out_dfm_model="selfcaled.mdl", use_apselfcal=use_selfcal, use_pselfcal=False)
    rchisq0 = stat_dict["rchisq"]
    print("Initial rchisq = ", rchisq0)
    if use_selfcal:
        dof = stat_dict["dof"] - n_eff_gain
    else:
        dof = stat_dict["dof"]
    print("DoF = ", dof)
    delta_rchisq = 1.0/dof
    required_chisq = rchisq0 + delta_rchisq
    print("Required chi2 = ", required_chisq)
    # original_comps = import_difmap_model(dfm_model_file)
    original_comps = import_difmap_model("selfcaled.mdl")
    # original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    if not use_selfcal:
        comps = import_difmap_model(dfm_model_file)
    else:
        comps = import_difmap_model("selfcaled.mdl")

    errors = dict()

    for i, comp in enumerate(comps):
        index = 0
        par0 = comp.p[index]

        def min_func_lower(delta):
            ccomp = copy.copy(comp)
            ccomp.set_value(par0 - delta, index)
            # Hack to fix flux
            ccomp._fixed[0] = True
            new_comps = comps.copy()
            new_comps.insert(i, ccomp)
            new_comps.pop(i+1)
            export_difmap_model(new_comps, "dfm.mdl", freq)
            sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, nmodelfit=nmodelfit_cycle,
                                              use_apselfcal=use_selfcal, use_pselfcal=False)
            print("For delta = {} Jy the differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
            return (required_chisq - sdict["rchisq"])**2

        def min_func_upper(delta):
            ccomp = copy.copy(comp)
            ccomp.set_value(par0 + delta, index)
            # Hack to fix the flux
            ccomp._fixed[0] = True
            new_comps = comps.copy()
            new_comps.insert(i, ccomp)
            new_comps.pop(i+1)
            export_difmap_model(new_comps, "dfm.mdl", freq)
            sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, nmodelfit=nmodelfit_cycle,
                                              use_apselfcal=use_selfcal, use_pselfcal=False)
            print("For delta = {} Jy the differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
            return (required_chisq - sdict["rchisq"])**2

        # First find upper bound
        lower_bounds = [1E-4]
        upper_bounds = [10.0]
        n_fun_eval = 200
        delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

        print("===================")

        # Find low bound
        lower_bounds = [1E-4]
        upper_bounds = [par0]
        n_fun_eval = 200
        delta_lower, _ = dlib.find_min_global(min_func_lower, lower_bounds, upper_bounds, n_fun_eval)

        print("Parameters = {} - {} + {}".format(par0, delta_lower, delta_upper))
        errors[i] = (delta_lower, delta_upper)
        # sys.exit(0)
    return errors


def find_position_errors_using_chi2(dfm_model_file, uvfits, stokes="I", working_dir=None,
                                    delta_t_sec=30.0, gain_phase_tcoh=30.0,
                                    show_difmap_output=False, nmodelfit=20,
                                    unselfcalibrated_uvfits=None, freq=15.4E+09):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff = gain_phase_tcoh/delta_t_sec
    rho = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits)
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    n_ant = len(uvdata.antennas)

    # Coefficient of correlation between gains in close IFs
    rho_IF = 0.75
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/10.
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff
    if stokes == "I":
        n_eff_gain_phases *= 2

    import dlib
    # Just best rchisq (w/o any fit or self-cal)
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir, nmodelfit=100, use_pselfcal=True,
                                          out_dfm_model="selfcaled.mdl")
    rchisq0 = stat_dict["rchisq"]
    dof = stat_dict["dof"] - n_eff_gain_phases
    delta_rchisq = 1.0/dof
    required_chisq = rchisq0 + delta_rchisq
    # original_comps = import_difmap_model(dfm_model_file)
    original_comps = import_difmap_model("selfcaled.mdl")
    original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    # comps = import_difmap_model(dfm_model_file)
    comps = import_difmap_model("selfcaled.mdl")

    errors = list()

    for i, comp in enumerate(comps):
        # x, y
        for index in (1, 2):

            par0 = comp.p[index]

            def min_func_lower(delta):
                ccomp = copy.copy(comp)
                ccomp.set_value(par0 - delta, index)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", freq)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                                  nmodelfit=nmodelfit, use_pselfcal=True)
                # print("For delta = {} differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
                return (required_chisq - sdict["rchisq"])**2

            def min_func_upper(delta):
                ccomp = copy.copy(comp)
                ccomp.set_value(par0 + delta, index)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", freq)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir,
                                                  nmodelfit=nmodelfit, use_pselfcal=True)
                return (required_chisq - sdict["rchisq"])**2

            # First find upper bound
            lower_bounds = [1E-4]
            upper_bounds = [1.0]
            n_fun_eval = 50
            delta_lower, _ = dlib.find_min_global(min_func_lower, lower_bounds, upper_bounds, n_fun_eval)
            delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

            print("Parameters = {} - {} + {}".format(par0, delta_lower, delta_upper))
            errors.append((delta_lower, delta_upper))
            # sys.exit(0)
    return errors


def find_2D_position_errors_using_chi2(dfm_model_file, uvfits, stokes="I", working_dir=None,
                                       delta_t_sec=10.0, gain_phase_tcoh=30.0,
                                       show_difmap_output=False, use_gain_dofs=False,
                                       freq=15.4E+09, nmodelfit_cycle=200):

    # Nuber of time stamps for gain phases to "forget" their values
    t_eff = gain_phase_tcoh/delta_t_sec
    rho = np.exp(-(delta_t_sec/gain_phase_tcoh)**2)

    # Find gains DoF
    uvdata = UVData(uvfits, verify_option="ignore")
    n_use_vis = uvdata.n_usable_visibilities_difmap(stokes=stokes)
    n_IF = uvdata.nif
    print(f"# IF = {n_IF}")
    n_ant = len(uvdata.antennas)
    print(f"# ant = {n_ant}")

    # Coefficient of correlation between gains in close IFs
    rho_IF = 0.75
    n_IF_eff = n_IF/(1+(n_IF-1)*rho_IF)


    baseline_scan_times = uvdata.baselines_scans_times
    n_scans = np.argmax(np.bincount([len(a) for a in baseline_scan_times.values()]))
    print(f"# scans = {n_scans}")

    n_measurements = 0
    time_of_scan = list()
    for bl, scans in baseline_scan_times.items():
        for scan in scans:
            delta_t_scan_sec = (Time(scan[-1], format="jd") - Time(scan[0], format="jd")).sec
            time_of_scan.append(delta_t_scan_sec)
            n_measurements += delta_t_scan_sec/delta_t_sec
    time_of_scan = np.median(time_of_scan)

    n_eff_gain_phases = n_scans * time_of_scan * n_IF_eff / delta_t_sec * (n_ant-1) / t_eff
    if stokes == "I":
        n_eff_gain_phases *= 2

    print("N_eff phases = ", n_eff_gain_phases)

    import dlib
    if use_gain_dofs:
        use_pselfcal = True
        nmodelfit = 200
    else:
        use_pselfcal = False
        n_eff_gain_phases = 0
        nmodelfit = 0
    stat_dict = find_stat_of_difmap_model(dfm_model_file, uvfits, stokes, working_dir, nmodelfit=nmodelfit,
                                          use_pselfcal=use_pselfcal, out_dfm_model="selfcaled.mdl",
                                          show_difmap_output=True)
    print("STAT DICT : ", stat_dict)
    rchisq0 = stat_dict["rchisq"]
    dof = stat_dict["dof"] - n_eff_gain_phases
    print("DoF = ", dof)
    # delta_rchisq = 2.0/dof
    delta_rchisq = chi2.ppf(1 - 0.32, 2)/dof
    print(f"Delta rchisq = {delta_rchisq}")
    required_chisq = rchisq0 + delta_rchisq
    print(f"Required rchisq = {required_chisq}")

    if not use_gain_dofs:
        original_comps = import_difmap_model(dfm_model_file)
    else:
        original_comps = import_difmap_model("selfcaled.mdl")
    original_comps = sorted(original_comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    if not use_gain_dofs:
        comps = import_difmap_model(dfm_model_file)
    else:
        comps = import_difmap_model("selfcaled.mdl")
    comps = sorted(comps, key=lambda c: np.hypot(c.p[1], c.p[2]))

    errors = dict()

    for i, comp in enumerate(comps):
        errors[i] = list()
        # Cycle through the polar angle in coordinate system with origin at the best position
        for PA_cur in np.arange(0, np.deg2rad(360), np.deg2rad(30)):

            x0, y0 = comp.p[1], comp.p[2]
            # Convert to RA, DEC
            dec0 = -y0
            ra0 = -x0

            print("Searching along PA = {:.1f} deg around RA = {:.3f}, DEC = {:.3f}".format(np.rad2deg(PA_cur), ra0, dec0))

            def rot_shift(ra, dec, PA, delta):
                """
                RA&DEC after rotating on PA (rad) counter-clockwise around (ra, dec) and shifting on delta (mas)
                """
                return ra + delta*np.sin(PA), dec + delta*np.cos(PA)

            def min_func(delta):
                ra_new, dec_new = rot_shift(ra0, dec0, PA_cur, delta)
                ccomp = copy.copy(comp)
                # Set x (-RA)
                ccomp.set_value(-ra_new, 1)
                # Set y (-DEC)
                ccomp.set_value(-dec_new, 2)
                # Hack to fix r & theta
                ccomp._fixed[1] = True
                ccomp._fixed[2] = True
                new_comps = comps.copy()
                new_comps.insert(i, ccomp)
                new_comps.pop(i+1)
                export_difmap_model(new_comps, "dfm.mdl", freq)
                sdict = find_stat_of_difmap_model("dfm.mdl", uvfits, stokes, working_dir, show_difmap_output,
                                                  nmodelfit=nmodelfit_cycle, use_apselfcal=False, use_pselfcal=use_gain_dofs)
                # print("For delta = {} differece = {}".format(delta, abs(required_chisq - sdict["rchisq"])))
                return (required_chisq - sdict["rchisq"])**2

            # First find upper bound
            lower_bounds = [1E-7]
            upper_bounds = [1.0]
            n_fun_eval = 200
            delta_bound, _ = dlib.find_min_global(min_func, lower_bounds, upper_bounds, n_fun_eval)
            # delta_upper, _ = dlib.find_min_global(min_func_upper, lower_bounds, upper_bounds, n_fun_eval)

            print("Parameters: PA = {:.1f} deg, delta = {:.7f}".format(np.rad2deg(PA_cur), delta_bound[0]))
            errors[i].append((PA_cur, delta_bound[0]))
        # sys.exit(0)
    return errors


def convert_2D_position_errors_to_ell_components(dfm_model_file, errors, include_shfit=True, filter_by_r=False):
    """
    :param dfm_model_file:
    :param errors:
        Dictionary with keys - component numbers and values lists of tuples with
        first element - theta [rad], second element - distance in this direction
        [mas]. Angle ``theta`` counts from North to East (as usually).
        Components are sorted by r.
    :return:
        Iterable of elliptical components representing 2D positional errors.
        Major axis are full widths (2*a).
    """
    from components import EGComponent
    comps = import_difmap_model(dfm_model_file)
    comps = sorted(comps, key=lambda c: np.hypot(c.p[1], c.p[2]))
    error_comps = list()
    for i, comp in enumerate(comps):
        error_list = errors[i]
        theta = list()
        r = list()
        for error_entry in error_list:
            theta.append(error_entry[0])
            r.append(error_entry[1])

        if filter_by_r:
            # Remove outliers
            IQR = scoreatpercentile(r, 75) - scoreatpercentile(r, 25)
            upper = scoreatpercentile(r, 75) + 1.5*IQR
            filtered_r = list()
            filtered_theta = list()
            for i in range(len(r)):
                if r[i] < upper:
                    filtered_r.append(r[i])
                    filtered_theta.append(theta[i])
        else:
            filtered_r = r
            filtered_theta = theta


        fit = fit_ellipse_to_2D_position_errors(filtered_theta, filtered_r, use="skimage")
        dRA = fit["xc"]
        dDEC = fit["yc"]
        a = fit["a"]
        b = fit["b"]
        bmaj = max(a, b)
        bmin = min(a, b)
        e = bmin/bmaj
        bpa = fit["theta"]
        print("Fitted BPA = ", np.rad2deg(bpa))
        # FIXME: Hack to fix skimage result
        # if bpa < np.pi/2:
        #     print("Addin 90 deg")
        # bpa += np.pi/2
        print("Component #{}, BPA = {}".format(i, np.rad2deg(bpa)))
        comp_ra = -comp.p[1]
        comp_dec = -comp.p[2]
        if include_shfit:
            error_comp = EGComponent(0.0, -comp_ra-dRA, -comp_dec-dDEC, 2*bmaj, e, bpa)
        else:
            error_comp = EGComponent(0.0, -comp_ra, -comp_dec, 2*bmaj, e, bpa)
        error_comps.append(error_comp)
    return error_comps


# This return semi-axes lengths
def fit_ellipse_to_2D_position_errors(PA, dr, use="skimage"):
    """
    :param PA:
    :param dr:
    :param use:
    :return:
        Returns semi-axis (a & b).
    """

    t = np.array(PA)
    dr = np.array(dr)

    # Filter small values
    indx = dr > 1E-04
    t = t[indx]
    dr = dr[indx]

    import matplotlib.pyplot as plt
    # FIXME:
    # t -= np.pi/2
    plt.polar(t, dr, ".k")
    plt.show()

    x = -dr*np.sin(t)
    y = dr*np.cos(t)
    # x = dr*np.cos(t)
    # y = dr*np.sin(t)

    if use == "skimage":
        from skimage.measure import EllipseModel
        ell = EllipseModel()
        ell.estimate(np.dstack((x, y))[0])
        try:
            xc, yc, a, b, theta = ell.params
        except TypeError:
            print("Failed to estimate Ellipsoid!")
            print("t = ", t)
            print("dr = ", dr)
            return {"xc": 0, "yc": 0, "a": np.mean(dr), "b": np.mean(dr), "theta": 0}
        # xc - adds to DEC, yc - adds to RA, PA = theta + 90 deg
        return {"xc": xc, "yc": yc, "a": a, "b": b, "theta": theta}

    if use == "dlib":
        import dlib

        def min_func(a, b, theta):
            xt = a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
            yt = a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
            return np.sum((x - xt)**2 + (y - yt)**2)

        lower_bounds = [1E-07, 1E-07, -np.pi]
        upper_bounds = [1.0, 1.0, np.pi]
        n_fun_eval = 1000
        res, _ = dlib.find_min_global(min_func, lower_bounds, upper_bounds, n_fun_eval)
        return {"a": res[0], "b": res[1], "theta": res[2]}

    if use == "simple":
        a, b, x_c, y_c, theta = fit_ellipse(x, y)
        # cen = ellipse_center(res)
        # lengths = ellipse_axis_length(res)
        # theta = ellipse_angle_of_rotation(res)
        return {"xc": x_c, "yc": y_c, "a": a, "b": b, "theta": np.deg2rad(theta)}

    else:
        raise Exception("Methods: skimage or dlib")


def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:
        M - major axis length
        m - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    a = __fit_ellipse(x, y)
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m
    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return [M, m, centre[0], centre[1], phi]






# def fitEllipse(x,y):
#     from numpy import linalg
#     x = x[:,np.newaxis]
#     y = y[:,np.newaxis]
#     D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
#     S = np.dot(D.T,D)
#     C = np.zeros([6,6])
#     C[0,2] = C[2,0] = 2; C[1,1] = -1
#     E, V =  linalg.eig(np.dot(linalg.inv(S), C))
#     n = np.argmax(np.abs(E))
#     a = V[:,n]
#     return a
#
# def ellipse_center(a):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     num = b*b-a*c
#     x0=(c*d-b*f)/num
#     y0=(a*f-b*d)/num
#     return np.array([x0,y0])
#
# def ellipse_angle_of_rotation( a ):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     return 0.5*np.arctan(2*b/(a-c))
#
# def ellipse_axis_length( a ):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
#     down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#     down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#     res1=np.sqrt(up/down1)
#     res2=np.sqrt(up/down2)
#     return np.array([res1, res2])
#
# def find_ellipse(x, y):
#     xmean = x.mean()
#     ymean = y.mean()
#     x -= xmean
#     y -= ymean
#     a = fitEllipse(x,y)
#     center = ellipse_center(a)
#     center[0] += xmean
#     center[1] += ymean
#     phi = ellipse_angle_of_rotation(a)
#     axes = ellipse_axis_length(a)
#     x += xmean
#     y += ymean
#     return center, phi, axes




# def fitEllipse(x,y):
#     from numpy.linalg import eig, inv
#     x = x[:,np.newaxis]
#     y = y[:,np.newaxis]
#     D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
#     S = np.dot(D.T,D)
#     C = np.zeros([6,6])
#     C[0,2] = C[2,0] = 2; C[1,1] = -1
#     E, V =  eig(np.dot(inv(S), C))
#     n = np.argmax(np.abs(E))
#     a = V[:,n]
#     return a
#
# def ellipse_center(a):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     num = b*b-a*c
#     x0=(c*d-b*f)/num
#     y0=(a*f-b*d)/num
#     return np.array([x0,y0])
#
# def ellipse_angle_of_rotation( a ):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     return 0.5*np.arctan(2*b/(a-c))
#
# def ellipse_axis_length( a ):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
#     down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#     down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
#     res1=np.sqrt(up/down1)
#     res2=np.sqrt(up/down2)
#     return np.array([res1, res2])
#
#
# def ellipse_angle_of_rotation2( a ):
#     b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
#     if b == 0:
#         if a > c:
#             return 0
#         else:
#             return np.pi/2
#     else:
#         if a > c:
#             return np.arctan(2*b/(a-c))/2
#         else:
#             return np.pi/2 + np.arctan(2*b/(a-c))/2



def fit_core(uvdata):
    from scipy.optimize import curve_fit
    uv = uvdata.uv
    vis_obs = uvdata.uvdata_freq_averaged.compressed()

    assert uv.shape[0] == vis_obs.shape[0]

    def min_func(uv, *p):
        comp = CGComponent(*p)
        vis = comp.ft(uv)
        return np.sum((abs(vis - vis_obs)**2))

    p0 = [1.0, 0, 0, 0.25]
    res = curve_fit(min_func, uv, vis_obs, p0=p0, maxfev=50000)
    flux, along, across, bmaj = res[0]

    return CGComponent(flux, along, across, bmaj)


def modelfit_core_wo_extending(uvfits, mapsize_clean, beam_fractions, path_to_script, use_elliptical=False,
                               use_brightest_pixel_as_initial_guess=True, save_dir=None,
                               dump_json_result=True):


    result = dict()

    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean,
                     path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=None, shift=None,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)

        uvdata = UVData(uvfits)
        freq_hz = uvdata.frequency
        # Find beam
        bmin, bmaj, bpa = find_nw_beam(uvfits, stokes="i", mapsize=mapsize_clean, uv_range=None, working_dir=working_dir)
        print("NW beam : {:.2f} mas, {:.2f} mas, {:.2f} deg".format(bmaj, bmin, bpa))

        # Find the brightest pixel
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
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

        for beam_fraction in beam_fractions:
            # Filter CCs
            filter_difmap_CC_model_by_r(os.path.join(working_dir, "cc.mdl"),
                                        os.path.join(working_dir, "filtered_cc.mdl"),
                                        r_c, bmaj*beam_fraction)
            # Add single gaussian component model to CC model
            join_difmap_models(os.path.join(working_dir, "filtered_cc.mdl"),
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
                    result.update({beam_fraction: {"flux": flux, "ra": ra,
                                                   "dec": dec, "size": size,
                                                   "rms": np.nan}})
                else:
                    flux, ra, dec, size, e, bpa = components[0]
                    result.update({beam_fraction: {"flux": flux, "ra": ra,
                                                   "dec": dec, "size": size,
                                                   "e": e, "bpa": bpa,
                                                   "rms": np.nan}})
        # if save_dir is not None:
        #     for fn in (os.path.join(working_dir, "filtered_cc.mdl"),
        #                os.path.join(working_dir, "1.mdl"),
        #                os.path.join(working_dir, "test_cc.fits"),
        #                os.path.join(working_dir, "hybrid.mdl"),
        #                os.path.join(working_dir, "hybrid_fitted.mdl")):
        #         fname = os.path.split(fn)[-1]
        #         shutil.move(fn, os.path.join(save_dir, fname))

    base = os.path.split(uvfits)[-1]
    base = base.split(".")[:-1]
    base = ".".join(base)
    if dump_json_result:
        with open(os.path.join(save_dir, f"{base}_core_modelfit_result.json"), "w") as fo:
            json.dump(result, fo)

    return result


def make_map_with_core_at_zero(mdl_file, uv_fits_fname, mapsize_clean,
                               path_to_script, outfname="shifted_cc.fits",
                               stokes="I"):
    """
    Function that shifts map in a way that core in new map will have zero
    coordinates.

    :param mdl_file:
        Path to difmap model file with the core being the first component.
    :param uv_fits_fname:
        Path to UV FITS data.
    :param mapsize_clean:
        Iterable of number of pixels and pixel size [mas].
    :param path_to_script:
        Path to D.Homan difmap final CLEAN script.
    :param outfname:
        Where to save new FITS file with map.
    :param stokes: (optional)
        Stokes parameter. (default: ``I``)
    """
    mdl_dir, mdl_fn = os.path.split(mdl_file)
    uv_dir, uv_fn = os.path.split(uv_fits_fname)
    core = import_difmap_model(mdl_file, mdl_dir)[0]
    ra_mas = -core.p[1]
    dec_mas = -core.p[2]
    shift = (-ra_mas, -dec_mas)
    clean_difmap(uv_fn, outfname, stokes, mapsize_clean, uv_dir, path_to_script,
                 shift=shift)


def make_uvfits_with_core_at_zero(in_uv_fits, out_uv_fits, core_ra, core_dec, show_difmap_output=False):
    """
    Shift phase center and brings it to core. Core RA = - core.p[1] and core
    DEC = -core.p[2] in ``Component`` class.
    """
    from subprocess import Popen, PIPE

    cmd = "observe " + in_uv_fits + "\n"
    cmd += "shift " + str(-core_ra) + ", " + str(-core_dec) + "\n"
    cmd += "wobs " + out_uv_fits + ", true\n"
    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)


def reformat_errors_from_Tb_for_silke(path, epochs_to_skip=()):
    dfm_models = glob.glob(os.path.join(path, "new_*.mod"))
    for dfm_model in dfm_models:
        fn = os.path.split(dfm_model)[-1]
        epoch = fn[4:-4]
        if epoch in epochs_to_skip:
            continue
        print(f"Reformatting errors for epoch {epoch}!")
        comps = import_difmap_model(fn, path)
        flux_errors = np.loadtxt(os.path.join(path, f"{epoch}_flux_error.txt"))
        pos_errors = np.loadtxt(os.path.join(path, f"{epoch}_pos_error.txt"))
        size_errors = np.loadtxt(os.path.join(path, f"{epoch}_size_error.txt"))
        with open(os.path.join(path, f"{epoch}_reformatted_errors.txt"), "w") as fo:
            fo.write("# flux[Jy] flux_err[Jy] pos_err[mas] size_err[mas]\n")
            for i, comp in enumerate(comps):
                flux = comp.p[0]
                flux_err = flux_errors[i]
                pos_err = pos_errors[i]
                size_err = size_errors[i]
                fo.write(f"{flux} {flux_err} {pos_err} {size_err}\n")


def reformat_profile_errors_for_silke(models_dir, errors_dir, save_dir):
    dfm_models = glob.glob(os.path.join(models_dir, "*_exp.mod"))
    for dfm_model in dfm_models:
        fn = os.path.split(dfm_model)[-1]
        epoch = fn[:10]
        print(f"Reformatting errors for epoch {epoch}!")
        comps = import_difmap_model(fn, models_dir)
        flux_errors = np.loadtxt(os.path.join(errors_dir, f"{epoch}_flux_errors.txt"))[:, 1]
        pos_errors = np.loadtxt(os.path.join(errors_dir, f"{epoch}_pos_errors.txt"))[:, 2]
        size_errors = np.loadtxt(os.path.join(errors_dir, f"{epoch}_size_errors.txt"))[:, 1]
        with open(os.path.join(save_dir, f"{epoch}_reformatted_errors.txt"), "w") as fo:
            fo.write("# flux[Jy] flux_err[Jy] pos_err[mas] size_err[mas]\n")
            for i, comp in enumerate(comps):
                flux = comp.p[0]
                flux_err = flux_errors[i]
                flux_err = np.hypot(flux_err, 0.1*flux)
                pos_err = pos_errors[i]
                size_err = size_errors[i]
                fo.write(f"{flux} {flux_err} {pos_err} {size_err}\n")


if __name__ == "__main__":

    # old_difmap_file = "/home/ilya/github/bk_transfer/pics/flares/tmp/cc.mdl"
    # new_difmap_file = "/home/ilya/github/bk_transfer/pics/flares/tmp/new.mdl"
    # gauss_model = "/home/ilya/github/bk_transfer/pics/flares/tmp/in1.mdl"
    # hybrid_model = "/home/ilya/github/bk_transfer/pics/flares/tmp/hybrid.mdl"
    # for fn in (new_difmap_file, gauss_model, hybrid_model):
    #     os.unlink(fn)
    #
    # filter_difmap_CC_model_by_r(old_difmap_file, new_difmap_file, (0, 0), 2.5)
    # comp = (1., 0., 0., 0.25)
    # create_difmap_file_from_single_component(comp, gauss_model, 2.3e+09)
    # join_difmap_models(new_difmap_file, gauss_model, hybrid_model)

    # uvfits = "/home/ilya/github/bk_transfer/pics/flares/tmp/template_S_1800.0.uvf"
    # # CLEAN_difmap(uvfits, "i", (512, 0.5), "test_cc.fits", restore_beam=None,
    # #              boxfile="/home/ilya/github/bk_transfer/pics/flares/tmp/wins.txt",
    # #              working_dir="/home/ilya/github/bk_transfer/pics/flares/tmp", uvrange=None,
    # #              box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=0.3,
    # #              remove_difmap_logs=True, save_noresid=None, save_resid_only=None,
    # #              save_dfm="/home/ilya/github/bk_transfer/pics/flares/tmp/cc.mdl",
    # #              noise_to_use="F", shift=None)
    #
    # clean_difmap(uvfits, "test_cc.fits", "i", (512, 0.5),
    #              path="/home/ilya/github/bk_transfer/pics/flares/tmp",
    #              path_to_script="/home/ilya/github/bk_transfer/scripts/script_clean_rms", mapsize_restore=None, beam_restore=None,
    #              outpath="/home/ilya/github/bk_transfer/pics/flares/tmp", shift=None, show_difmap_output=False,
    #              command_file=None, clean_box=None, save_dfm_model="cc.mdl", omit_residuals=False,
    #              do_smooth=True, dmap=None, text_box=None,
    #              box_rms_factor=None, window_file=None,
    #              super_unif_dynam=None, unif_dynam=None,
    #              taper_gaussian_value=None, taper_gaussian_radius=None)


#     # uvfits_files = ["template_S_1936.6.uvf", "template_S_2085.5.uvf" ,
#     #                 "template_S_2234.5.uvf", "template_S_2383.4.uvf"]
#     # test_dir = "/home/ilya/github/bk_transfer/pics/flares/test"
#     # path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
#     #
#     # beam_fractions = [1.0]
#     # mapsize_clean = (2048, 0.05)
#     # results_1 = list()
#     # results_2 = list()
#     #
#     # for uvfits in uvfits_files:
#     #     results = modelfit_core_wo_extending(uvfits,
#     #                                          beam_fractions, path=test_dir,
#     #                                          mapsize_clean=mapsize_clean,
#     #                                          path_to_script=path_to_script,
#     #                                          niter=500,
#     #                                          out_path=test_dir,
#     #                                          use_brightest_pixel_as_initial_guess=True,
#     #                                          estimate_rms=True,
#     #                                          stokes="i",
#     #                                          use_ell=False)
#     #     results_2.append(results)
#     #
#     #     results = modelfit_core_wo_extending(uvfits,
#     #                                          beam_fractions, path=test_dir,
#     #                                          mapsize_clean=mapsize_clean,
#     #                                          path_to_script=path_to_script,
#     #                                          niter=500,
#     #                                          out_path=test_dir,
#     #                                          use_brightest_pixel_as_initial_guess=True,
#     #                                          estimate_rms=True,
#     #                                          stokes="i",
#     #                                          use_ell=False,
#     #                                          two_stage=False)
#     #     results_1.append(results)
#     #
#     #
#     # for result_1, result_2 in zip(results_1, results_2):
#     #     print("Two stages")
#     #     print(result_2)
#     #     print("One stage")
#     #     print(result_1)
#     #     print("==================")
#     #
#     # import sys; sys.exit(0)
#
#     # import pickle
#     # data_dir = "/home/ilya/data/Mkn501/difmap_models"
#     # pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
#     # for pkl_file in pkl_files:
#     #     epoch = os.path.split(pkl_file)[-1][7:17]
#     #     print("Processing epoch ", epoch)
#     #     mdl_file = os.path.join(data_dir, "{}.mod".format(epoch))
#     #     with open(pkl_file, "rb") as fo:
#     #         errors = pickle.load(fo)
#     #     errors_comps = convert_2D_position_errors_to_ell_components(os.path.join(data_dir, mdl_file),
#     #                                                                 errors, include_shfit=False)
#     #     pos_errors = [0.5*errors_comps[i].p[3]*(1+errors_comps[i].p[4]) for i in range(len(errors_comps))]
#     #     bpas = [np.rad2ded(errors_comps[i].p[5]) for i in range(len(errors_comps))]
#     #     bmajs = [errors_comps[i].p[3] for i in range(len(errors_comps))]
#     #     es = [errors_comps[i].p[4] for i in range(len(errors_comps))]
#     #     with open(os.path.join(data_dir, "{}_posistion_ellipse_errors_chi2_errors.txt".format(epoch)), "w") as fo:
#     #         for err in pos_errors:
#     #             fo.write("{}\n".format(err))
#
#
# # ============
#
#     # import matplotlib.pyplot as plt
#     # import pickle
#     # from image import plot as iplot
#     #
#     #
#     # pixsize_mas = 0.1
#     # data_dir = "/home/ilya/data/Mkn501/difmap_models"
#     # save_dir = os.path.join(data_dir, "test")
#     # ccfits_files = sorted(glob.glob(os.path.join(data_dir, "*.icn.fits.gz")))
#     # ccfits_files = [os.path.split(path)[-1] for path in ccfits_files]
#     # epochs = [fn.split(".")[2] for fn in ccfits_files]
#     # mdl_files = ["{}.mod".format(epoch) for epoch in epochs]
#     #
#     # for ccfits_file, mdl_file, epoch in zip(ccfits_files, mdl_files, epochs):
#     #     # Problematic epochs
#     #     if epoch in ["1997_03_13", "2001_12_30", "2003_08_23", "2004_05_29"]:
#     #         continue
#     #
#     #     print(mdl_file, ccfits_file)
#     #
#     #     uvfits_file = "1652+398.u.{}.uvf".format(epoch)
#     #     uvdata = UVData(os.path.join(data_dir, uvfits_file))
#     #     all_stokes = uvdata.stokes
#     #     if "RR" in all_stokes and "LL" in all_stokes:
#     #         stokes = "I"
#     #     else:
#     #         if "RR" in all_stokes:
#     #             stokes = "RR"
#     #         else:
#     #             stokes = "LL"
#     #     print("Stokes parameter: ", stokes)
#     #
#     #     # Find errors if they are not calculated
#     #     if not os.path.exists(os.path.join(data_dir, "errors_{}.pkl".format(epoch))):
#     #         errors = find_2D_position_errors_using_chi2(os.path.join(data_dir, mdl_file),
#     #                                                     os.path.join(data_dir, uvfits_file),
#     #                                                     stokes=stokes,
#     #                                                     show_difmap_output=False)
#     #         with open(os.path.join(data_dir, "errors_{}.pkl".format(epoch)), "wb") as fo:
#     #             pickle.dump(errors, fo)
#     #     # Or just load already calculated
#     #     else:
#     #         with open(os.path.join(data_dir, "errors_{}.pkl".format(epoch)), "rb") as fo:
#     #             errors = pickle.load(fo)
#     #
#     #     # Make dummy elliptical components for plotting errors
#     #     error_comps = convert_2D_position_errors_to_ell_components(os.path.join(data_dir, mdl_file),
#     #                                                                errors, include_shfit=False)
#     #
#     #     comps = import_difmap_model(os.path.join(data_dir, mdl_file))
#     #     ccimage = create_clean_image_from_fits_file(os.path.join(data_dir, ccfits_file))
#     #     beam = ccimage.beam
#     #     npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*pixsize_mas**2)
#     #     std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
#     #     blc, trc = find_bbox(ccimage.image, level=4*std, min_maxintensity_mjyperbeam=6*std,
#     #                          min_area_pix=4*npixels_beam, delta=10)
#     #     fig, axes = plt.subplots(1, 1, figsize=(10, 15))
#     #     fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std,
#     #                 blc=blc, trc=trc, beam=beam, show_beam=True, show=False,
#     #                 close=True, contour_color='black',
#     #                 plot_colorbar=False, components=comps, components_errors=error_comps,
#     #                 outfile="{}_original_model_errors2D_test".format(epoch), outdir=save_dir, fig=fig)
#     #
#     #
#     #     if not os.path.exists(os.path.join(data_dir, "size_errors_{}.pkl".format(epoch))):
#     #         size_errors = find_size_errors_using_chi2(os.path.join(data_dir, mdl_file),
#     #                                                   os.path.join(data_dir, uvfits_file),
#     #                                                   show_difmap_output=False)
#     #         with open(os.path.join(data_dir, "size_errors_{}.pkl".format(epoch)), "wb") as fo:
#     #             pickle.dump(size_errors, fo)
#     #
#     #     if not os.path.exists(os.path.join(data_dir, "flux_errors_{}.pkl".format(epoch))):
#     #         flux_errors = find_flux_errors_using_chi2(os.path.join(data_dir, mdl_file),
#     #                                                   os.path.join(data_dir, uvfits_file),
#     #                                                   show_difmap_output=False)
#     #         with open(os.path.join(data_dir, "flux_errors_{}.pkl".format(epoch)), "wb") as fo:
#     #             pickle.dump(flux_errors, fo)
#     #
#     #     #
#     #     # stat_dict = find_stat_of_difmap_model(os.path.join(data_dir, mdl_file),
#     #     #                                       os.path.join(data_dir, uvfits_file),
#     #     #                                       stokes, data_dir, nmodelfit=100, use_pselfcal=True,
#     #     #                                       out_dfm_model="selfcaled.mdl")
#     #     # selfcaled_comps = import_difmap_model(os.path.join(data_dir, "selfcaled.mdl"))
#     #     # fig, axes = plt.subplots(1, 1, figsize=(10, 15))
#     #     # fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std,
#     #     #             blc=blc, trc=trc, beam=beam, show_beam=True, show=False,
#     #     #             close=True, contour_color='black',
#     #     #             plot_colorbar=False, components=selfcaled_comps,
#     #     #             outfile="{}_selfcaled_model".format(epoch), outdir=data_dir, fig=fig)
#     #     #
#     #     # sys.exit(0)
#     #
#     #
#     #
#     # import sys
#     # sys.exit(0)
#
#
    # ==========================================================================

    import matplotlib
    matplotlib.use('TkAgg')
    import glob
    # new_path = "/home/ilya/data/silke/1215/last/0d2/"
    # new_path = "/home/ilya/data/Mkn501/difmap_models/redone_epochs"
    # new_path = "/home/ilya/Downloads/TXS0506/tberrors"
    # new_path = "/home/ilya/data/silke/0735/15GHz"
    # new_path = "/home/ilya/data/silke/0506_old"
    # new_path = "/home/ilya/Downloads/3C454.3"
    # new_path = "/home/ilya/Downloads/3C454.3/Boston/lost"
    # new_path = "/home/ilya/Downloads/TXS8"
    new_path = "/home/ilya/Downloads/pks1717draftmodelfitfiles"
    # dfm_models = glob.glob("/home/ilya/data/silke/1215/*.mod")
    # dfm_models = glob.glob("/home/ilya/data/Mkn501/difmap_models/redone_epochs/*.mod")
    # dfm_models = glob.glob("/home/ilya/Downloads/TXS0506/*.mod")

    # rm old
    torm_dfm_models = glob.glob(os.path.join(new_path, "new*.mod"))
    for fn in torm_dfm_models:
        os.unlink(fn)

    dfm_models = glob.glob(os.path.join(new_path, "*.mod"))
    print(dfm_models)
    # freq_ghz = 43.0
    freq_ghz = 15.4
    # sys.exit(0)
    problems = list()

    epochs = list()

    # ACTUAL ERRORS CALCULATION ################################################
    for dfm_model in dfm_models:
        fn = os.path.split(dfm_model)[-1]
        if fn != "2013_02_28.mod":
            continue
        epoch = fn[:10]
        epochs.append(epoch)
        print(f"EPOCH = {epoch} =====================")

        comps = import_difmap_model(dfm_model, new_path)
        new_dfm_model = os.path.join(new_path, "new_{}".format(os.path.split(dfm_model)[-1]))
        export_difmap_model(comps, new_dfm_model, 1E+09*freq_ghz)
        # uvfits = "/home/ilya/data/silke/1215/1215+303.u.{}.uvf".format(epoch)
        # uvfits = "/home/ilya/data/Mkn501/difmap_models/1652+398.u.{}.uvf".format(epoch)
        # uvfits = "/home/ilya/Downloads/TXS0506/0506+056.u.{}.uvf".format(epoch)
        # uvfits = "{}/0506+056.u.{}.uvf".format(new_path, epoch)
        # uvfits = "{}/0506+056.u.{}.uvf".format(new_path, epoch)
        # uvfits = "{}/J2253+1608_Q_{}_mar_vis.fits".format(new_path, epoch)
        # if epoch == "2014_06_09":
        #     uvfits = "{}/J0509+0541_X_{}_pus_vis.fits".format(new_path, epoch)
        # elif epoch == "2010_11_13":
        #     uvfits = "{}/J0509+0541_U_{}_moj_vis.fits".format(new_path, epoch)
        # else:
        #     uvfits = "{}/J0509+0541_X_{}_pet_vis.fits".format(new_path, epoch)
        uvfits = os.path.join(new_path, "1717+178.u.{}.uvf".format(epoch))
        if epoch not in ():
            try:
                df = components_info(uvfits, new_dfm_model, dmap_size=(1024, 0.1), PA=None,
                                     size_error_coefficient=0.35)
            except:
                problems.append(epoch)
                continue
            np.savetxt(new_path + "/{}_pos_error.txt".format(epoch), 0.5*df["major_err"])
            np.savetxt(new_path + "/{}_size_error.txt".format(epoch), df["major_err"])
            flux_err = np.hypot(df["flux_err"], 0.05*df["flux"])
            np.savetxt(new_path + "/{}_flux_error.txt".format(epoch), flux_err)

    ##############################################

    # Skip all except the target one
    epochs_to_skip = ()
    # epochs_to_skip.remove("2010_11_13")

    reformat_errors_from_Tb_for_silke(new_path, epochs_to_skip=epochs_to_skip)
                                      # epochs_to_skip=("2010_11_13",))
                                      # epochs_to_skip=problems)
    print("Problem epochs : ", problems)