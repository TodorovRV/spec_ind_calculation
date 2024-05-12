import warnings
from astropy import wcs
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
import os
import glob
import datetime
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from astropy.convolution import convolve, Gaussian2DKernel
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
from astropy.io import fits as pf
from astropy import units as u
from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.modeling import models, fitting
import matplotlib
import matplotlib.patches as patches
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData, downscale_uvdata_by_freq
from from_fits import create_model_from_fits_file


matplotlib.use("Agg")
label_size = 16
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams["contour.negative_linestyle"] = 'dotted'
import matplotlib.pyplot as plt

deg2mas = u.deg.to(u.mas)


def get_file_basename(fn):
    names = os.path.split(fn)[-1].split(".")
    n = len(names)-1
    basename = names[0]
    for i in range(1, n):
        basename += ("."+names[i])
    return basename


def downscale_uvdata_by_freq(uvdata):
    if abs(uvdata.hdu.data[0][0]) > 1:
        downscale_by_freq = True
    else:
        downscale_by_freq = False
    return downscale_by_freq


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
    print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[1]) + ", bmaj = " + str(restore_beam[0]) + ", " + str(restore_beam[2]) + " deg")
    # default dimfap: false,true (parameters: omit_residuals, do_smooth)
    cmd += "restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
    cmd += "wmap " + out_ccfits + "\n"
    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)


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


def find_nw_beam(uvfits, stokes="I", mapsize=(1024, 0.1), uv_range=None, working_dir=None):
    """
    :return:
        Beam parameters (bmaj[mas], bmin[mas], bpa[deg]).
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

    # Get final reduced chi_squared
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


def get_beam_params_from_CCFITS(ccfits):
    """
    :return:
        Beam parameters (bmaj[mas], bmin[mas], bpa[deg]).
    """
    bmaj, bmin, bpa = None, None, None
    hdulist = pf.open(ccfits)
    pr_header = hdulist[0].header
    try:
        # BEAM info in ``AIPS CG`` table
        idx = hdulist.index_of('AIPS CG')
        data = hdulist[idx].data
        bmaj = float(data['BMAJ'])*deg2mas
        bmin = float(data['BMIN'])*deg2mas
        bpa = float(data['BPA'])
    # In Petrov's data it in PrimaryHDU header
    except KeyError:
        try:
            bmaj = pr_header['BMAJ']*deg2mas
            bmin = pr_header['BMIN'] *deg2mas
            bpa = pr_header['BPA']
        except KeyError:
            # In Denise data it is in PrimaryHDU ``HISTORY``
            # TODO: Use ``pyfits.header._HeaderCommentaryCards`` interface if
            # any
            try:
                for line in pr_header['HISTORY']:
                    if 'BMAJ' in line and 'BMIN' in line and 'BPA' in line:
                        bmaj = float(line.split()[3])*deg2mas
                        bmin = float(line.split()[5])*deg2mas
                        bpa = float(line.split()[7])
            except KeyError:
                pass
        if not (bmaj and bmin and bpa):
            raise Exception("Beam info absent!")

    return bmin, bmaj, bpa


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


def find_bbox(array, level, min_maxintensity_mjyperbeam, min_area_pix):
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
        List of tuples with BLC and TRC for each region.

    :note:
        These are BLC, TRC for numpy array (i.e. transposed source map as it
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

    blcs = list()
    trcs = list()

    bounding_boxes = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)
        bounding_boxes.append((blc, trc))

    return bounding_boxes


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


def filter_CC(ccfits, mask, out_ccfits=None, out_dfm=None, show=False,
              plotsave_fn=None, axes=None):
    """
    :param mask:
        Mask with region of source flux being True.
    :param out_ccfits:
    """
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


# def create_clean_boxes(ccfits, n_std_bbox=3, n_std_lowest_contour=3,
#                        n_std_min_maxintensity=4.0, min_area_pix=100,
#                        enlarge_fractional_factor=0.1):
#     wcs = make_wcs_from_ccfits(ccfits)
#
#     icn = pf.getdata(ccfits)
#     # Remove one-sized dimensions (RA, DEC, ...)
#     icn = icn.squeeze()
#     # Robustly estimate image pixels std
#     std = mad_std(icn)
#
#     # Find preliminary bounding box
#     bounding_boxes = find_bbox(icn, level=n_std_bbox*std,
#                                min_maxintensity_mjyperbeam=n_std_min_maxintensity*std,
#                                min_area_pix=min_area_pix)
#
#     # Now mask out source emission using found bounding box and estimate std
#     # more accurately
#     mask = np.zeros(icn.shape)
#     for blc, trc in bounding_boxes:
#         mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
#     outside_icn = np.ma.array(icn, mask=mask)
#     std = mad_std(outside_icn)
#
#     # Final bounding box estimation
#     bounding_boxes = find_bbox(icn, level=n_std_bbox*std,
#                                min_maxintensity_mjyperbeam=n_std_min_maxintensity*std,
#                                min_area_pix=min_area_pix)
#
#     # Enlarge 10% each side
#     enlarged_bounding_boxes = list()
#     for blc, trc in bounding_boxes:
#         delta_ra = abs(trc[0]-blc[0])
#         delta_dec = abs(trc[1]-blc[1])
#         blc = (blc[0] - int(enlarge_fractional_factor*delta_ra), blc[1] - int(enlarge_fractional_factor*delta_dec))
#         trc = (trc[0] + int(enlarge_fractional_factor*delta_ra), trc[1] + int(enlarge_fractional_factor*delta_dec))
#         enlarged_bounding_boxes.append((blc, trc))
#
#
#     fig = plt.figure(figsize=(8, 8))
#     axes = fig.add_axes([0.1, 0.1, 0.9, 0.9], projection=wcs, aspect='equal')
#     axes.coords[0].set_axislabel('Relative Right Ascension (mas)', size='large')
#     axes.coords[1].set_axislabel('Relative Declination (mas)', size='large')
#
#     lev0 = n_std_lowest_contour*std
#     levels = lev0 * np.logspace(0, 30, num=31, base=np.sqrt(2))
#     levels = np.hstack(([-lev0], levels))
#     levels = levels[levels < icn.max()]
#
#     axes.contour(icn, levels=levels, colors='k', linewidths=0.75)
#
#
#     # Plot each BBOX
#     for blc, trc in enlarged_bounding_boxes:
#         print("BLC = ", blc)
#         print("TRC = ", trc)
#         ra_min, ra_max, dec_min, dec_max = convert_bbox_to_radec_ranges(wcs, blc, trc)
#         print("RA : ", ra_min, ra_max)
#         print("DEC : ", dec_min, dec_max)
#         # rect = patches.Rectangle((blc[1], trc[1]), trc[0]-blc[0], -trc[1]+blc[1], linewidth=1, edgecolor='r', facecolor='none')
#         # axes.add_patch(rect)
#     plt.show()


# FIXME: Handle cases when no V is available and we need to estimate target rms from 1 asec distant region
# FIXME: Note that I have changed ``save_...`` arguments from boolean to None/filename
def CLEAN_difmap(uvfits, stokes, mapsize, outname, restore_beam=None,
                 boxfile=None, working_dir=None, uvrange=None,
                 box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                 remove_difmap_logs=True, save_noresid=None, save_resid_only=None, save_dfm=None,
                 noise_to_use="F"):
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


# Same as CLEAN_difmap, but uvw 5,-1 always
def CLEAN_difmap_RA(uvfits, stokes, mapsize, outname, restore_beam=None,
                 boxfile=None, working_dir=None, uvrange=None,
                 box_clean_nw_niter=1000, clean_gain=0.03, dynam_su=20, dynam_u=6, deep_factor=1.0,
                 remove_difmap_logs=True, save_noresid=False, save_resid_only=False, save_dfm=False,
                 noise_to_use="F"):
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
    cmd += "uvw 5, -1\n"
    if uvrange is not None:
        cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
    cmd += "print \"wtnoise =\", imstat(noise)\n"

    cmd += "select v\n"
    cmd += "uvw 5, -1\n"
    if uvrange is not None:
        cmd += "uvrange {}, {}\n".format(uvrange[0], uvrange[1])
    cmd += "print \"vnoise =\", imstat(rms)\n"

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
    cmd += "dynam = {}\n clean_niter = 10\n uvw 5,-1\n map_residual\n uvw 5,-1\n map_residual\n".format(dynam_su)
    cmd += "uvw 5,-1\n dynam = {}\n map_residual\n".format(dynam_u)

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
        cmd += "uvw 5,-1\n target_rms = imstat(noise)\n deep_map_residual\n"
        # default dimfap: false,true (parameters: omit_residuals, do_smooth)
        cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "false,true" + "\n"
        cmd += "wmap " + outname + ".fits\n"
        if save_noresid:
            cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
            cmd += "wmap " + outname + "_noresid.fits\n"
        if save_resid_only:
            cmd += "wdmap " + outname + "_resid_only.fits\n"
        if save_dfm:
            cmd += "wmod " + outname + ".dfm\n"
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
            cmd += "uvw 5,-1\n"
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
        cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "false,true" + "\n"
        cmd += "wmap " + outname + ".fits\n"
        if save_noresid:
            cmd += "restore " + str(restore_beam[0]) + "," + str(restore_beam[1]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
            cmd += "wmap " + outname + "_noresid.fits\n"
        if save_resid_only:
            cmd += "wdmap " + outname + "_resid_only.fits\n"
        if save_dfm:
            cmd += "wmod " + outname + ".dfm\n"
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


# def CLEAN_difmap(uvfits, stokes, mapsize, outname, restore_beam=None,
#                  show_difmap_output=True, boxfile=None, box_rms_deep_factor=10):
#     from subprocess import Popen, PIPE
#     from os import read
#     import fcntl
#
#     difmap = Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
#     fcntl.fcntl(difmap.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
#
#     # First get noise estimates: from visibility & weights and from Stokes V
#     # difmap.stdin.write("wrap_print_output = false\n")
#     a = difmap.stdin.write("observe "+uvfits+"\n")
#     print(a)
#     # difmap.stdin.flush()
#     print(repr(difmap.stdout.readline()))
#
#     # cmd += "select "+stokes+"\n"
#     # cmd += "mapsize "+str(int(2*mapsize[0]))+","+str(mapsize[1])+"\n"
#     # cmd += "uvw 0, -2\n"
#     # cmd += "print \"wtnoise =\", imstat(noise)\n"
#     # cmd += "select v\n"
#     # cmd += "uvw 0, -2\n"
#     # cmd += "print \"vnoise =\", imstat(rms)\n"
#     # cmd += "exit\n"
#
#     # "cat" will exit when you close stdin.  (Not all programs do this!)
#     difmap.stdin.close()
#     print('Waiting for difmap to exit')
#     difmap.wait()
#     print('difmap finished with return code %d' % difmap.returncode)
#
#     # default dimfap: false,true (parameters: omit_residuals, do_smooth)
#     # cmd += "restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
#     # cmd += "wmap " + outname + ".fits" + "\n"
#
#     # with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
#     #     outs, errs = difmap.communicate(input=cmd)
#     # if show_difmap_output:
#     #     print(outs)
#     #     print(errs)
#     # return outs, errs


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


def get_significance_mask_from_mc(original_ccfits, mc_ccfits, mask=None, perc=2.5):
    """
    :param mask:
        Pre-determined mask. Consider only unmasked pixesl.
    """
    from scipy.stats import percentileofscore
    original = pf.getdata(original_ccfits).squeeze()
    result = np.ones(original.shape, dtype=bool)
    mc = [pf.getdata(ccfits).squeeze() for ccfits in mc_ccfits]
    mc_cube = np.dstack(mc)
    if mask is None:
        mask = np.ones(original.shape, dtype=bool)
    for (i, j), value in np.ndenumerate(original):
        if mask[i, j]:
            continue
        low = percentileofscore(mc_cube[i, j, :] - np.median(mc_cube[i, j, :]) + original[i, j], 0.0)
        if low < perc:
            result[i, j] = False
    return result


def get_significance_mask_from_mc2(original_ccfits, mc_ccfits, mask=None, n_sigma_min=3):
    """
    :param mask:
        Pre-determined mask. Consider only unmasked pixesl.
    """
    from scipy.stats import percentileofscore, norm
    original = pf.getdata(original_ccfits).squeeze()
    result = np.ones(original.shape, dtype=bool)
    mc = [pf.getdata(ccfits).squeeze() for ccfits in mc_ccfits]
    mc_cube = np.dstack(mc)
    if mask is None:
        mask = np.ones(original.shape, dtype=bool)
    for (i, j), value in np.ndenumerate(original):
        if mask[i, j]:
            continue
        sign = percentileofscore(mc_cube[i, j, :] - np.median(mc_cube[i, j, :]) + original[i, j], 0.0)/100.
        n_sigma = norm.isf(sign)

        if n_sigma > n_sigma_min or sign == 0.0:
            result[i, j] = False

    return result


def get_errors_from_mc(mc_ccfits, mask=None):
    from scipy.stats import scoreatpercentile
    mc = [pf.getdata(ccfits).squeeze() for ccfits in mc_ccfits]
    result = np.nan*np.ones(mc[0].shape, dtype=float)
    mc_cube = np.dstack(mc)
    if mask is None:
        mask = np.ones(result.shape, dtype=bool)
    for (i, j), value in np.ndenumerate(mc[0]):
        if mask[i, j]:
            continue
        low, up = scoreatpercentile(mc_cube[i, j, :], [16, 84])
        result[i, j] = 0.5*(up - low)
    return result


def create_mask(shape, region):
    """
    :param region:
        Tuple (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
        None,) or (center[0][pix], center[1][pix], bmaj[pix], e, bpa[deg])
    :return:
        Numpy 2D bool array.
    """
    if region[3] is None:
        # Creating a disc shaped mask with radius r
        a, b = region[0], region[1]
        n = min(shape)
        r = region[2]
        y, x = np.ogrid[-a: n - a, -b: n - b]
        mask = x ** 2 + y ** 2 <= r ** 2

    elif len(region) == 4:
        # Creating rectangular mask
        y, x = np.ogrid[0: shape[0], 0: shape[1]]
        mask = (region[0] < x) & (x < region[2]) & (region[1] < y) & (y < region[3])

    elif len(region) == 5:
        # Create elliptical mask
        a, b = region[0], region[1]
        n = min(shape)
        y, x = np.ogrid[-a: n - a, -b: n - b]
        bmaj = region[2]
        e = region[3]
        bpa = np.deg2rad(region[4])
        bmin = bmaj * e
        # This brings PA to VLBI-convention (- = from North counterclocwise)
        bpa += np.pi/2
        a = np.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
            np.sin(bpa) ** 2. / (2. * bmin ** 2.)
        b = np.sin(2. * bpa) / (2. * bmaj ** 2.) - \
            np.sin(2. * bpa) / (2. * bmin ** 2.)
        c = np.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
            np.cos(bpa) ** 2. / (2. * bmin ** 2.)
        mask = a * x ** 2 + b * x * y + c * y ** 2 <= 1
    else:
        raise Exception
    return mask


def registrate_images(image1, image2, mask1, mask2, fit_gaussian=True, n=9):
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
    :return:
        A tuple of shifts [pixels] (DEC, RA).
    """
    corr_matrix = cross_correlate_masked(image1, image2, mask1, mask2)
    max_pos = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

    if fit_gaussian:
        # Grab a part pf image around maximal correlation coefficient
        sub = corr_matrix[max_pos[0]-n : max_pos[0]+n, max_pos[1]-n : max_pos[1]+n]
        x, y = np.mgrid[:2*n, :2*n]
        p_init = models.Gaussian2D(1, n, n, n/2, n/2, 0)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, x, y, sub)
        # TODO: Check +1
        result = p.x_mean.value-n+1, p.y_mean.value-n+1
    else:
        result = max_pos[0]-image1.shape[0]+1, max_pos[1]-image2.shape[1]+1

    return result


if __name__ == "__main__":

    # Debugging image registration #############################################
    # working_dir = "/home/ilya/data/alpha/reanalysis/shifts"
    # uvfits_x = "/home/ilya/data/alpha/reanalysis/shifts/1228+126.x.2006_06_15.uvf"
    # uvfits_u = "/home/ilya/data/alpha/reanalysis/shifts/1228+126.u.2006_06_15.uvf"
    # uvrange_x = get_uvrange(uvfits_x)
    # uvrange_u = get_uvrange(uvfits_u)
    # uvrange = (uvrange_u[0], uvrange_x[1])
    # common_beam = find_nw_beam(uvfits_x, mapsize=(8192, 0.03), uv_range=uvrange)
    # boxfile = os.path.join(working_dir, "wins.txt")
    # outs, errs = CLEAN_difmap(uvfits_x, "i", (8192, 0.03), "x", deep_factor=1.0,
    #                           working_dir=working_dir, restore_beam=common_beam,
    #                           boxfile=boxfile)
    # outs, errs = CLEAN_difmap(uvfits_u, "i", (8192, 0.03), "u", deep_factor=1.0,
    #                           working_dir=working_dir, restore_beam=common_beam,
    #                           boxfile=boxfile)
    # ccfits_x = os.path.join(working_dir, "x.fits")
    # ccfits_u = os.path.join(working_dir, "u.fits")
    # mask = convert_boxfile_to_mask(ccfits_x, boxfile)
    # mask_core = ~create_mask((8192, 8192), (4096, 4096, 100, 0.5, np.deg2rad(-2)))
    # mask = np.logical_and(mask, mask_core)
    #
    # image_u = pf.getdata(ccfits_u).squeeze()
    # image_x = pf.getdata(ccfits_x).squeeze()
    # result = registrate_images(image_u, image_x, mask, mask)
    # # ccfits = "/home/ilya/github/fb7ac8ce8a072ba114fb47b464bf063b/0333+321_original_stack_I.fits"
    # # create_clean_boxes(ccfits, min_area_pix=200, n_std_min_maxintensity=10, n_std_bbox=3, enlarge_fractional_factor=0.25)


    # # Rebase U-band CLEAN model to X-band uv-coverage for BK145 data ###########
    # target_uvfits = "/home/ilya/data/M87Lesha/to_ilya/1228+126.X.2009_05_23.uvf_cal"
    # source_difmap_model = "/home/ilya/data/M87Lesha/to_ilya/1228+126.U.2009_05_23C.cc"
    # stokes = "I"
    # mapsize = (2048, 0.1)
    # restore_beam = (1, 1, 0)
    # rebase_CLEAN_model(target_uvfits, "/home/ilya/data/M87Lesha/to_ilya/U_cc_rebased_to_X_uv.uvf",
    #                    stokes, mapsize, restore_beam, source_difmap_model=source_difmap_model,
    #                    noise_scale_factor=0.1, need_downscale_uv=True)
    # uvdata = UVData("/home/ilya/data/M87Lesha/to_ilya/U_cc_rebased_to_X_uv.uvf")
    # uvdata.uvplot()


    # Rebase U-band CLEAN model to X-band uv-coverage for artificial BK145 data ###########
    target_uvfits = "/home/ilya/data/M87Lesha/bias_compensation/BK145_8GHz_artificial_KH.uvf"
    source_difmap_model = "/home/ilya/data/M87Lesha/bias_compensation/U.mod"
    stokes = "I"
    mapsize = (2048, 0.1)
    restore_beam = (1, 1, 0)
    rebase_CLEAN_model(target_uvfits, "/home/ilya/data/M87Lesha/bias_compensation/U_cc_rebased_to_X_uv_KH.uvf",
                       stokes, mapsize, restore_beam, source_difmap_model=source_difmap_model,
                       noise_scale_factor=0.1, need_downscale_uv=True)
    uvdata = UVData("/home/ilya/data/M87Lesha/bias_compensation/U_cc_rebased_to_X_uv_KH.uvf")
    uvdata.uvplot()