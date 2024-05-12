from itertools import combinations, chain
import warnings
import collections
import re
import math
import numpy as np
import astropy.io.fits as pf
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.stats import mad_std
import matplotlib as mpl
import string
from math import floor
from scipy import optimize
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from scipy.stats import scoreatpercentile
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn import svm
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
try:
    from skimage import transform
    have_skimage = True
except ImportError:
    have_skimage = False
if have_skimage:
    try:
        from skimage.filters import gaussian_filter
    except ImportError:
        from skimage.filters import gaussian as gaussian_filter
from scipy.ndimage import interpolation
try:
    # Python 3 moved reduce to the functools module
    from functools import reduce
except ImportError:
    # Python 2 reduce is a built-in
    pass


vcomplex = np.vectorize(complex)
v_int = np.vectorize(int)
v_round = np.vectorize(round)

deg2mas = u.deg.to(u.mas)
deg2rad = u.deg.to(u.rad)
mas2rad = u.mas.to(u.rad)
degree_to_mas = deg2mas
degree_to_rad = deg2rad
mas_to_rad = mas2rad

stokes_dict = {-8: 'YX', -7: 'XY', -6: 'YY', -5: 'XX',
               -4: 'LR', -3: 'RL', -2: 'LL', -1: 'RR',
               1: 'I', 2: 'Q', 3: 'U', 4: 'V'}


# Dictionary of GRT latitudes and longitudes (sign "-" mean West of Greenwich.
GRT_coordinates = {'AR': (18.344167, -66.752778),
                   'GBT': (38.433056, -79.839722),
                   'EB': (50.524722, 6.882778),
                   'EFF': (50.524722, 6.882778),
                   'WSRT': (52.914722, 6.603333),
                   'SC': (17.75652, -64.58376),
                   'HN': (42.93362, -71.98681),
                   'NL': (41.77165, -91.574133),
                   'FD': (30.635214, -103.944826),
                   'LA': (35.7752887, -106.2455897),
                   'PT': (34.30107, -108.11912),
                   'KP': (31.956253, -111.612361),
                   'OV': (37.23176, -118.27714),
                   'BR': (48.13117, -119.68325),
                   'MK': (19.80159, -155.45581),
                   'Y':  (34.078749, -107.617728)}


class SyncArray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __setitem__(self, coords, value):
        super(SyncArray, self).__setitem__(coords, value)


def to_boolean_array(numbers, length):
    """
    Convert iterable of integers to boolean array.

    :param numbers:
        Iterable of integers.
    :param length:
        Integer. Length of the resulting array.
    :return:
        Numpy boolean array with indexes of ``numbers`` being ``True``.
    """
    numbers = list(numbers)
    result = np.zeros(length, dtype=bool)
    result[numbers] = np.ones(len(numbers))
    return result


def from_boolean_array(bool_array):
    """
    Convert boolean array to array of integers.
    :param bool_array:
        Boolean array.
    :return:
        Numpy integer array with length equal to number of ``True`` elements in
        original boolean array.
    """
    return np.where(bool_array)[0]


def mask_boolean_with_boolean(bool_array, bool_mask):
    """
    Mask boolean array with other boolean array.
    :param bool_array:
        Boolean numpy array.
    :param bool_mask:
        Boolean mask with size equal to number of ``True`` elements in
        ``bool_array``.
    :return:
        Boolean array with shape of ``bool_array`` where some of the original
        ``True`` elements are now ``False`` depending on there boolean value in
        ``bool_mask``.
    """
    assert np.count_nonzero(bool_array) == len(bool_mask)
    int_indxs = from_boolean_array(bool_array)
    int_indxs = int_indxs[bool_mask]
    return to_boolean_array(int_indxs, len(bool_array))


def check_issubset(to_check, original):
    """
    Check that ``to_check`` is among iterable ``original``.

    :param to_check:
        This must be among ``original`` elements.
    :param original:
        Iterable with elements to check .
    :return:
        Boolean.
    """
    try:
        s = set((to_check,))
    # If to_check is a list (unhashable)
    except TypeError:
        s = set(to_check)
    return s.issubset(original)


def nested_ddict():
    """
    Defaultdict with arbitrary number of levels.
    """
    return collections.defaultdict(nested_ddict)


def fit_2d_gmm(cdata, n_max=5):
    """
    Fit complex array on complex plane with number of gaussians.

    :param cdata:
        Complex numpy array.
    :param n_max: (optional)
        Maximum number of gaussian components in mixture. (default: ``5``)
    :return:
        Instance of ``sklearn.mixture.GMM`` class with best model.
    """
    re = cdata.real
    im = cdata.imag
    reim = np.vstack((re, im)).T
    clf_dict = dict()
    for n in range(1, n_max + 1):
        clf = GMM(n_components=n, covariance_type='full', min_covar=0.00001)
        clf.fit(reim)
        clf_dict.update({n: clf})
    n_mixture = sorted(clf_dict, key=lambda x: clf_dict[x].bic(reim))[0]
    print("Best n_mixture = {}".format(n_mixture))
    return clf_dict[n_mixture]


def fit_kde(data):
    """
    Fit 1D density, representing number of points (Re/Im of visibility) with
    gaussian KDE.

    :param data:
        Numpy array.
    :return:
        Instance of ``sklearn.neighbors.KernelDensity`` class with best density
        estimate choosen by 5-fold CV.
    """

    params = {'bandwidth': np.logspace(-3, 1, 10)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data[:, np.newaxis])

    return grid.best_estimator_


def fit_2d_kde(cdata):
    """
    Fit 2D density, representing number of points on complex plane with gaussian
    KDE.

    :param cdata:
        Complex numpy array.
    :return:
        Instance of ``sklearn.neighbors.KernelDensity`` class with best density
        estimate choosen by 5-fold CV.
    """

    re = cdata.real
    im = cdata.imag
    reim = np.vstack((re, im)).T
    params = {'bandwidth': np.logspace(-3, 1, 10)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(reim)

    return grid.best_estimator_


def find_outliers_2d_svm(data, outliers_fraction):
    """
    Found outliers in complex data using one class SVM.

    :param data:
        Complex numpy array.
    :param outliers_fraction:
        Fraction of outliers suggested.
    :return:
        Boolean numpy array with outliers.
    """
    re = data.real
    im = data.imag
    reim = np.vstack((re, im)).T
    clf = svm.OneClassSVM(nu=outliers_fraction,
                          kernel='rbf', gamma=0.1)
    clf.fit(reim)
    y_pred = clf.decision_function(reim).ravel()
    threshold = scoreatpercentile(y_pred,
                                  100 * outliers_fraction)
    return y_pred < threshold


def find_outliers_2d_mincov(data, mahalanobis_max):
    """
    Found outliers in complex data using Minimum Covariance Determinant.

    :param data:
        Complex numpy array.
    :param mahalanobis_max:
        Maximum mahalanobis distance to count data point as outlier.
    :return:
        Boolean numpy array with outliers.
    """
    re = data.real
    im = data.imag
    reim = np.vstack((re, im)).T
    clf = MinCovDet().fit(reim)
    mahalanobis_dist = clf.mahalanobis(reim)

    return mahalanobis_dist > mahalanobis_max


# FIXME: Choose ``eps`` using robust std estimate
def find_outliers_2d_dbscan(data, eps, min_samples):
    """
    Found outliers in complex data using DBSCAN clustering algorithm.

    :param data:
        Complex numpy array.
    :param eps:
        The maximum distance between two samples for them to be considered as in
        the same neighborhood.
    :param min_samples:
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself.

    :return:
        Boolean numpy array with outliers.

    :note:
        Common sense suggests that ``min_samples`` parameter should be ~ the
        number of visibilities in one scan (~15) if outliers are searched in
        baseline data. If outliers are searched in single scan's data then it
        is also should be the number of scan's visibilities.
    """
    re = data.real
    im = data.imag
    reim = np.vstack((re, im)).T
    X = StandardScaler().fit_transform(reim)
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='brute').fit(X)

    return db.labels_ == -1


# FIXME: Choose ``eps`` using robust std estimate
def find_outliers_dbscan(data, eps, min_samples):
    """
    Found outliers in real/imag data using DBSCAN clustering algorithm.

    :param data:
        Numpy array with real/imaginary visibility parts.
    :param eps:
        The maximum distance between two samples for them to be considered as in
        the same neighborhood.
    :param min_samples:
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself.

    :return:
        Boolean numpy array with outliers.

    :note:
        Common sense suggests that ``min_samples`` parameter should be ~ the
        number of visibilities in one scan (~15) if outliers are searched in
        baseline data. If outliers are searched in single scan's data then it
        is also should be the number of scan's visibilities.
    """
    data = np.asarray(data)
    data = data.reshape(-1, 1)
    X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='brute').fit(X)

    return db.labels_ == -1


def make_ellipses(gmm, ax, colors="rgbyk"):
    """
    Add ellipses representing components of Gaussian Mixture Model.

    :param gmm:
        Fitted instance of ``sklearn.mixture.GMM`` class.
    :param ax:
        Matplotlib axes object.
    """
    for n in range(gmm.n_components):
        color = colors[n]
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        print("before sqrt - {}".format(v))
        v = np.sqrt(v)
        print("after sqrt - {}".format(v))
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def get_fits_image_info(fname):
    """
    Returns image parameters from FITS-file.

    :param fname:
        Fits-file name.

    :return:
        Dictionary with following information:
        ``imsize`` [pix, pix] - size of image,
        ``pixref`` [pix, pix] - reference pixel numbers,
        ``pixrefval`` [rad, rad] - value of coordinates at reference pixels,
        ``(bmaj, bmin, bpa,)`` [rad, rad, rad] - beam parameters (if any). If no
        beam parameters found => ``(None, None, None,)``,
        ``pixsize`` [rad, rad]- size of pixel dimensions,
        ``stokes`` (I, Q, U or V) - stokes parameter that image does describe,
        ``freq`` [Hz] - sky frequency.

    """
    hdulist = pf.open(fname)
    return get_fits_image_info_from_hdulist(hdulist)


def get_fits_image_info_from_hdulist(hdulist):
    """
    Returns image parameters from instance of ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.

    :return:
        Dictionary with following information:
        ``imsize`` [pix, pix] - size of image,
        ``pixref`` [pix, pix] - reference pixel numbers,
        ``pixrefval`` [rad, rad] - value of coordinates at reference pixels,
        ``(bmaj, bmin, bpa,)`` [rad, rad, rad] - beam parameters (if any). If no
        beam parameters found => ``(None, None, None,)``,
        ``pixsize`` [rad, rad]- size of pixel dimensions,
        ``stokes`` (I, Q, U or V) - stokes parameter that image does describe,
        ``freq`` [Hz] - sky frequency.

    """
    bmaj, bmin, bpa = None, None, None
    pr_header = hdulist[0].header
    imsize = (pr_header['NAXIS1'], pr_header['NAXIS2'],)
    pixref = (int(pr_header['CRPIX1']), int(pr_header['CRPIX2']),)
    pixrefval = (pr_header['CRVAL1'] * degree_to_rad,
                 pr_header['CRVAL2'] * degree_to_rad,)
    pixsize = (pr_header['CDELT1'] * degree_to_rad,
               pr_header['CDELT2'] * degree_to_rad,)
    # Find stokes info
    stokes_card = find_card_from_header(pr_header, value='STOKES')[0]
    indx = stokes_card.keyword[-1]
    stokes = stokes_dict[pr_header['CRVAL' + indx]]
    # Find frequency info
    freq_card = find_card_from_header(pr_header, value='FREQ')[0]
    indx = freq_card.keyword[-1]
    freq = pr_header['CRVAL' + indx]

    try:
        # BEAM info in ``AIPS CG`` table
        idx = hdulist.index_of('AIPS CG')
        data = hdulist[idx].data
        bmaj = float(data['BMAJ']) * degree_to_rad
        bmin = float(data['BMIN']) * degree_to_rad
        bpa = float(data['BPA']) * degree_to_rad
    # In Petrov's data it in PrimaryHDU header
    except KeyError:
        try:
            bmaj = pr_header['BMAJ'] * degree_to_rad
            bmin = pr_header['BMIN'] * degree_to_rad
            bpa = pr_header['BPA'] * degree_to_rad
        except KeyError:
            # In Denise data it is in PrimaryHDU ``HISTORY``
            # TODO: Use ``pyfits.header._HeaderCommentaryCards`` interface if
            # any
            try:
                for line in pr_header['HISTORY']:
                    if 'BMAJ' in line and 'BMIN' in line and 'BPA' in line:
                        bmaj = float(line.split()[3]) * degree_to_rad
                        bmin = float(line.split()[5]) * degree_to_rad
                        bpa = float(line.split()[7]) * degree_to_rad
            except KeyError:
                pass
        if not (bmaj and bmin and bpa):
            warnings.warn("Beam info absent!")

    return {"imsize": imsize, "pixref": pixref, "pixrefval": pixrefval,
            "bmaj": bmaj, "bmin": bmin, "bpa": bpa, "pixsize": pixsize,
            "stokes": stokes, "freq": freq}


def get_hdu(fname, extname=None, ver=1):
    """
    Function that returns instance of ``PyFits.HDU`` class with specified
    extension and version from specified file.

    :param fname:
        Path to FITS-file.

    :param extname: (optional)
        Header's extension. If ``None`` then return first from
        ``PyFits.HDUList``. (default: ``None``)

    :param ver: (optional)
        Version of ``HDU`` with specified extension. (default: ``1``)

    :return:
        Instance of ``PyFits.HDU`` class.
    """

    hdulist = pf.open(fname)
    return get_hdu_from_hdulist(hdulist, extname, ver)


def get_hdu_from_hdulist(hdulist, extname=None, ver=1):
    """
    Function that returns instance of ``PyFits.HDU`` class with specified
    extension and version from instance of ``PyFits.HDUList``.

    :param hdulist:
        Instance of ``PyFits.HDUList``.

    :param extname: (optional)
        Header's extension. If ``None`` then return first from
        ``PyFits.HDUList``. (default: ``None``)

    :param ver: (optional)
        Version of ``HDU`` with specified extension. (default: ``1``)

    :return:
        Instance of ``PyFits.HDU`` class.

    """
    if extname:
        try:
            indx = hdulist.index_of((extname, ver,))
            hdu = hdulist[indx]
        except:
            raise AbsentHduExtensionError('No {} binary table'
                                          ' found'.format(extname))
    else:
        hdu = hdulist[0]

    return hdu


def find_card_from_header(header, value=None, keyword=None,
                          comment_contens=None):
    """
    Find card from header specified be several possible ways.

    :param header:
        Instance of ``astropy.io.fits.Header`` class.
    :param value:
        Value of header's card that specifies card.
    :param keyword:
        Keyword of header's card that specifies card.
    :param comment_contens:
        Comment of header's card that specifies card.

    :return:
        Instance of ``astropy.io.fits.card.Card`` class.
    """
    # HISTORY often holds shitty stuff
    try:
        del header["HISTORY"]
    except KeyError:
        pass
    if comment_contens is not None:
        search = [card for card in header.cards if comment_contens in
                  card.comment]
    else:
        search = header.cards

    # Fix just in case
    [card.verify("fix") for card in search]

    if value is not None and keyword is None:
        result = [card for card in search if card.value == value]
    elif value is None and keyword is not None:
        result = [card for card in search if card.keyword == keyword]
    elif value is not None and keyword is not None:
        result = [card for card in search if (card.keyword == keyword and
                                              card.value == value)]
    else:
        result = search

    return result


def get_key(header, value, keyword):
    """
    Get some keyword value from header.

    :param header:
        Instance of ``astropy.io.fits.Header`` class.
    :param value:
        Value of header's card that specifies parameter.
    :param keyword:
        Key to value to return.

    :return:
        Value for specified key.
    """
    freq_card = find_card_from_header(header, value=value)[0]
    return header[keyword + '{}'.format(freq_card[0][-1])]


class AbsentHduExtensionError(Exception):
    pass


class AbsentVersionOfBinTableError(Exception):
    pass


class EmptyImageFtError(Exception):
    pass

# TODO: convert utils to using arrays instead of lists


# numpy.lib.recfunctions.append_fields
def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.
    http://stackoverflow.com/questions/1201817/adding-a-field-to-a-structured-numpy-array

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> import numpy
    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


def index_of(ar1, ar2, issubset=True):
    """
    Find indexes of elements of 1d-numpy array-like ar1 in ar2.

    Output:

        list (len = len(ar1)) of arrays with indexes of elements in ar2
        corresponding to current (list[i] -> ar1[i]) element of ar1. If no
        elements are found then i-th element of list is None.
    """
    ar1 = np.array(ar1)
    ar2 = np.array(ar2)

    if issubset:
        # assert that all elements of ar1 are in ar2
        assert np.all(np.intersect1d(ar2, ar1) == np.sort(ar1))
        # assert np.all(np.in1d(ar1, ar2))

    indxs_ar2_sorted = np.argsort(ar2)
    ar1_pos_left = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='left')
    ar1_pos_right = np.searchsorted(ar2[indxs_ar2_sorted], ar1, side='right')

    indxs = list()
    for i in range(len(ar1_pos_left)):
        indxs.append(range(ar1_pos_left[i], ar1_pos_right[i]))

    # indxs = sum(indxs, [])
    result = list()
    for indx in indxs:
        if indx:
            result.append(indxs_ar2_sorted[indx])
        else:
            result.append(None)

    # return indxs_ar2_sorted[indxs]
    return result


def convert_fq_hdu(ohdu):
    new_columns = list()
    for coldef in ohdu.data.columns:
        c = pf.Column(name=coldef.name, format=coldef.format, unit=coldef.unit,
                      array=coldef.array)
        new_columns.append(c)

    hdu = pf.BinTableHDU.from_columns(new_columns, name='AIPS FQ')
    hdu.header.set('EXTVER', value=1, after='EXTNAME')
    return hdu


def convert_an_hdu(ohdu, new_prhdu):
    """
    Create AN table HDU.

    :param ohdu:
        Original AN table HDU.
    :param new_prhdu:
        New primary HDU with possibly some baselines missed.
    :return:
        Instance of ``BinTableHDU`` with AN table.
    """
    new_ants = np.array(baselines_2_ants(new_prhdu.data['BASELINE']))
    new_ants = ohdu.data['ANNAME'][np.array(new_ants)-1]
    old_ants = ohdu.data['ANNAME']
    an_indexes = to_boolean_array(np.array(index_of(new_ants, old_ants)).flatten(),
                                  len(old_ants))

    new_columns = list()
    for coldef in ohdu.data.columns:
        c = pf.Column(name=coldef.name, format=coldef.format, unit=coldef.unit,
                      array=coldef.array[an_indexes])
        new_columns.append(c)

    hdu = pf.BinTableHDU.from_columns(new_columns, name='AIPS AN')
    hdu.header.set('EXTVER', value=1, after='EXTNAME')
    return hdu


def _to_complex_array(struct_array, real_name, imag_name):
    """
    Method that takes structured array and names of 2 fields and returns
    complex numpy.ndarray.
    """

    assert(np.shape(struct_array[real_name]) ==
           np.shape(struct_array[imag_name]))

    return struct_array[real_name] + 1j * struct_array[imag_name]


def _to_one_ndarray(struct_array, *names):
    """
    Method that takes structured array and names of 2 (or more) fields and
    returns numpy.ndarray with expanded shape. Field can be 2-dim array.
    """

    # TODO: can i use struct_array[[name1, name2]] synthax? Yes but you'll get
    # structured array with this 2 fields.

    l = list()
    for name in names:
        if struct_array[name].ndim == 1:
            l.append(struct_array[name][:, None])
        elif struct_array[name].ndim == 2:
            l.extend(np.hsplit(struct_array[name], struct_array[name].shape[1]))

    return np.hstack(l)
    # return np.vstack([struct_array[name] for name in names]).T


def change_shape(_array, _dict1, _dict2):
    """
    Function that takes ndarray and 2 dictionaries with array's shape and
    permuted shape and returns array with permuted shape.

    Inputs:
        _array [numpy.ndarray] - array to change,
        dict1 - shape of array, that will be changed,
        dict2 - dictionary of new shape.
        dict2 can contain more keys then dict1. Only keys in dict2 that are in
            dict1 influence new shape. If dict1 contains some keys that are not
            in dict2, then position of such axes will be changed by other axis
            that contained in both dict1 and dict2.
    """

    dict1 = _dict1.copy()
    dict2 = _dict2.copy()
    array = _array.copy()

    for key in dict1:
        if key not in dict2:
            # Don't alter position of this dimension directly (but it could
            # change it's position because of other dimensions).
            pass
        else:
            if not dict1[key] == dict2[key]:
                array = np.swapaxes(array, dict1[key], dict2[key])
                # Updated values for 2 changed keys in dict1
                dict1[key] = dict2[key]
                for item in dict1.items():
                    # If found other key in dict1 with the same value
                    if (item[1] == dict2[key]) and (item[0] != key):
                        dict1[item[0]] = dict1[key]

    # Assert that altered dict1 (it's part with shapes from dict2) coincide
    # with dict2
    for key in dict2:
        if key in dict1:
            assert(dict1[key] == dict2[key])

    return array


# TODO: if ``min`` or ``max`` key is absent then only upper/lower bound does
# exist.
def get_indxs_from_struct_array(array, **kwargs):
    """
    Function that given structured array ``array`` and specified fields and
    conditions in ``kwargs`` argument returns corresponding indexes.

    Inputs:

        array - numpy structured array,

        kwargs - keyword arguments that specifies conditions:

            {field: value}

            If value is dictionary then use ``min`` & ``max`` keys to bound,
            if value is iterable then use its content.

    Output:

        numpy.array of indexes of ``array``.

    Example:
        >>>get_indxs_from_struct_array(array, time={'min': None, 'max': 0.5},
                                       baseline = [515, 517])
    """

    pass


def aips_bintable_fortran_fields_to_dtype_conversion(aips_type):
    """Given AIPS fortran format of binary table (BT) fields, returns
    corresponding numpy dtype format and shape. Examples:
    4J => array of 4 32bit integers,
    E(4,32) => two dimensional array with 4 columns and 32 rows.
    """

    intv = np.vectorize(int)
    aips_char = None
    dtype_char = None
    repeat = None
    _shape = None

    format_dict = {'L': 'bool', 'I': '>i2', 'J': '>i4', 'A': 'S',  'E': '>f4',
                   'D': '>f8'}

    for key in format_dict.keys():
        if key in aips_type:
            aips_char = key

    if not aips_char:
        raise Exception("aips data format reading problem " + str(aips_type))

    try:
        dtype_char = format_dict[aips_char]
    except KeyError:
        raise Exception("no dtype counterpart for aips data format" +
                        str(aips_char))

    try:
        repeat = int(re.search(r"^(\d+)" + aips_char,
                     aips_type).groups()[0])
        if aips_char == 'A':
            dtype_char = str(repeat) + dtype_char
            repeat = 1
    except AttributeError:
        repeat = None

    if repeat is None:
        _shape = tuple(intv(string.split(re.search(r"^" + aips_char +
                                                   "\((.+)\)$",
                                                   aips_type).groups()[0],
                                         sep=',')))
    else:
        _shape = repeat

    return dtype_char, _shape


def build_dtype_for_bintable_data(header):
    """Builds dtype for recarray from header.
    """

    # substitue = {'UV--SIN': 'u', 'VV--SIN': 'v', 'WW--SIN': 'w', 'BASELINE':
    # 'bl', 'DATE': 't'}
    # assert(header_dict['EXTNAME'] == 'UV_DATA')

    # # # of axis. 2 => matrix
    # naxis = int(header['NAXIS'])
    # # # of fields in a item
    tfields = int(header['TFIELDS'])
    # # # of Bytes in a item (sum of length of tfields elements)
    # naxis1 = int(header['NAXIS1'])
    # # # of items
    # naxis2 = int(header['NAXIS2'])
    # nrecords = naxis2

    # parameters of regular data matrix if in UV_DATA table
    try:
        maxis = int(header['MAXIS'])
    except KeyError:
        print("non UV_DATA")
        maxis = None

    # build np.dtype format
    names = []
    formats = []
    shapes = []
    tuple_shape = []
    array_names = []

    for i in range(1, tfields + 1):
        name = header['TTYPE' + str(i)]
        if name in names:
            name = name * 2
        names.append(name)
        _format, _shape = \
            aips_bintable_fortran_fields_to_dtype_conversion(header['TFORM' +
                                                                    str(i)])

        # building format & names for regular data matrix
        if name == 'FLUX' and maxis is not None:
            for i in range(1, maxis + 1):
                maxisi = int(header['MAXIS' + str(i)])
                if maxisi > 1:
                    tuple_shape.append(int(header['MAXIS' + str(i)]))
                    array_names.append(header['CTYPE' + str(i)])
            formats.append('>f4')
            shapes.append(tuple(tuple_shape))
            array_names = array_names
        else:
            formats.append(_format)
            shapes.append(_shape)

    print(names, formats, shapes, array_names)

    dtype_builder = zip(names, formats, shapes)
    dtype = [(name, _format, shape) for (name, _format, shape) in dtype_builder]

    return dtype, array_names


def baselines_2_ants(baselines):
    """Given list of baseline numbers (fits keyword) returns list of
    corresponding antennas.
    """
    # TODO: CHECK IF OUTPUT/INPUT IS OK!!!
    for baseline in baselines:
        baseline = abs(baseline)
        assert(baseline > 256)

    ants = list()
    for baseline in baselines:
        baseline = abs(baseline)
        ant1 = int(baseline // 256)
        ant2 = int(baseline - ant1 * 256)
        # if ant1 * 256 + ant2 != baseline:
        #     continue
        ants.append(ant1)
        ants.append(ant2)
    ants = list(set(ants))
    ants.sort()

    return ants


# def ants_2_baselines(ants):
#     """Given several antennas returns corresponding baselines.
#     """
#
#     baselines = list()
#     ants_by2 = list(permutations(ants, 2))
#     for ant in ants_by2:
#         baseline = 256 * ant[0] + ant[1]
#         baselines.append(baseline)
#     return baselines


def ant_2_containing_baslines(ant, antennas):
    """
    Given antenna returns list of all baselines among given list with that
    antenna.
    """

    baselines = list()
    for antenna in antennas:
        if antenna < ant:
            baselines.append(256 * antenna + ant)
        elif antenna > ant:
            baselines.append(256 * ant + antenna)
        else:
            pass

    return baselines


def ants_2_baselines(ants):
    baselines = list()
    for ant in ants:
        baselines.extend(ant_2_containing_baslines(ant, ants))
    baselines = list(set(baselines))
    baselines = baselines.sort()

    return baselines


def get_triangles(ref, antennas):
    """
    Find triangles of antennas.

    :param ref:
        Number of antenna to build triangles with.
    :param antennas:
        Iterable of antenna numbers to build triangles with.
    :return:
        Dictionary with keys - ijk of antenna numbers and values - lists of
        3 baseline numbers.
    """

    baselines_list = list()
    assert (len(antennas) >= 2), "Need > 2 antennas for triangle!"
    # antennas must be iterable
    baselines_list.extend(list(antennas))

    # Assert that we don't have the same antennas in ``antennas`` and
    # ``antennas`` keywords
    if len(baselines_list) == 2:
        assert ref not in baselines_list, "Need 3 diff. antennas!"
    else:
        if ref in baselines_list:
            baselines_list.remove(ref)

    # Find triangles (combinations of 3 antenna numbers)
    triangles = list()
    # This 3 triangles complement each triangle in `triangles`` so sum of their
    # closure phase is equal to triangle in ``traingles``
    complement_triangles = list()
    for comb in combinations(baselines_list, 2):
        triangles.append([ref] + list(comb))
    n = 0
    # Convert to baseline numbers
    triangle_baselines = dict()
    for triangle in triangles:
        i, j, k = sorted(triangle)
        triangle_baselines.update({str(i) + '-' + str(j) + '-' + str(k): [j + 256 * i,
                                                                          k + 256 * i,
                                                                          k + 256 * j]})
        n += 1
    print(n)

    return triangle_baselines


def get_quadrangles(ref, antennas):
    baselines_list = list()
    assert (len(antennas) >= 3), "Need > 3 antennas for quadrangle!"
    # antennas must be iterable
    baselines_list.extend(list(antennas))

    # Assert that we don't have the same antennas in ``antennas`` and
    # ``antennas`` keywords
    if len(baselines_list) == 3:
        assert ref not in baselines_list, "Need 4 diff. antennas!"
    else:
        if ref in baselines_list:
            baselines_list.remove(ref)

    # Find triangles (combinations of 3 antenna numbers)
    quadrangles = list()
    for comb in combinations(baselines_list, 3):
        quadrangles.append([ref] + list(comb))

    quadrangles_baselines = dict()
    n = 0
    for quadrangle in quadrangles:
        i, j, k, l = sorted(quadrangle)
        quadrangles_baselines.update({str(i) + str(j) + str(k) + str(l):
                                          ([str(i)+str(j), str(k)+str(l), str(i)+str(k), str(j)+str(l)],
                                           [str(i)+str(k), str(j)+str(l), str(i)+str(l), str(j)+str(k)])})
        n += 2
    print(n)
        # quadrangles_baselines.update({str(i) + str(j) + str(k) + str(l):
        #                                   ([i+j*256, k+l*256, i+256*k, j+256*l],
        #                                    [i+k*256, j+256*l, i+256*l, j+256*k])})

    return quadrangles_baselines


def time_frac_to_dhms(fractime):
    """Converts time in fraction of the day format to time in d:h:m:s
    format."""

    dhms = list()

    for time in fractime:
        day = int(floor(time))
        hour = int(floor(time * 24.))
        minute = int(floor(time * 1440. - hour * 60.))
        second = int(floor(time * 86400. - hour * 3600. - minute * 60.))
        dhms.append(tuple([day, hour, minute, second]))

    return dhms


def time_dhms_to_frac(dhmses):
    """Converts time in format d:h:m:s to time in parameters format =
    fraction of the day.
    """
    fractions = list()

    for dhms in dhmses:
        day, hour, minute, second = dhms
        fraction = day + hour / 24.0 + minute / (24.0 * 60.0) + \
            second / (24.0 * 60.0 * 60.0)
        fractions.append(fraction)

    return fractions


# FIXME: Seems that size of beam here is not the same as in difmap.
def gaussian_beam(size_x, bmaj, bmin, bpa, size_y=None):
    """
    Generate and return a 2D Gaussian function
    of dimensions (size_x,size_y).

    See Briggs PhD (Appendix B) for details.

    :param size_x:
        Size of first dimension [pixels].
    :param bmaj:
        Beam major axis size [pixels].
    :param bmin:
        Beam minor axis size [pixels].
    :param bpa:
        Beam positional angle [deg].
    :param size_y: (optional)
        Size of second dimension. Default is ``size_x``.
    :return:
        Numpy array of shape (``size_x``, ``size_y``,).
    """
    size_y = size_y or size_x
    # FIXME: it depends on where is reference pixel. This works for (512, 513)
    # Or for (256, 257)
    x, y = np.mgrid[-size_x-1: size_x-1, -size_y+1: size_y+1]
    # Constructing parameters of gaussian from ``bmaj``, ``bmin``, ``bpa``.
    a0 = 1. / (0.5 * bmaj) ** 2.
    c0 = 1. / (0.5 * bmin) ** 2.
    # This brings PA to VLBI-convention (- = from North counter-clockwise)
    bpa = -bpa
    theta = math.pi * (bpa + 90.) / 180.
    a = math.log(2) * (a0 * math.cos(theta) ** 2. +
                       c0 * math.sin(theta) ** 2.)
    b = (-(c0 - a0) * math.sin(2. * theta)) * math.log(2.)
    c = math.log(2) * (a0 * math.sin(theta) ** 2. +
                       c0 * math.cos(theta) ** 2.)

    g = np.exp(-a * x ** 2. - b * x * y - c * y ** 2.)
    return g


def infer_gaussian(data):
    """
    Return (amplitude, x_0, y_0, width), where width - rough estimate of
    gaussian width
    """
    amplitude = data.max()
    x_0, y_0 = np.where(data == amplitude)
    row = data[x_0, :]
    column = data[:, y_0]
    x_0 = float(x_0)
    y_0 = float(y_0)
    dx = len(np.where(row - amplitude/2 > 0)[0])
    dy = len(np.where(column - amplitude/2 > 0)[0])
    width = math.sqrt(dx ** 2. + dy ** 2.)

    return amplitude, x_0, y_0, width


# TODO: in components ft methods this repeats - use this method there
def gaussian(height, x0, y0, bmaj, e, bpa):
    """
    Returns a gaussian function with the given parameters.

    :example:
    create grid:
        x, y = np.meshgrid(x, y)
        imshow(gaussian(x, y))

    """
    # This brings PA to VLBI-convention (- = from NOrth counterclocwise)
    bpa = -bpa
    bmaj = bmaj / (2. * np.sqrt(2. * np.log(2)))
    bmin = e * bmaj
    a = math.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.sin(bpa) ** 2. / (2. * bmin ** 2.)
    b = math.sin(2. * bpa) / (2. * bmaj ** 2.) - \
        math.sin(2. * bpa) / (2. * bmin ** 2.)
    c = math.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.cos(bpa) ** 2. / (2. * bmin ** 2.)
    return lambda x, y: height * np.exp(-(a * (x - x0) ** 2 +
                                          b * (x - x0) * (y - y0) +
                                          c * (y - y0) ** 2))


def fit_gaussian(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit.
    """
    # Calculate initial values of circular gaussian + dummy params for
    # ellipticity
    params = list(infer_gaussian(data)) + [1., 0.]
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


# TODO: refactor to select mask type (circular, rectangular or elliptic)
# depending on ``len(region)``
def create_mask(shape, region):
    """
    Function that creates rectangular or circular mask.

    :param region:
        Tuple (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
        None,) or (center[0], center[1], bmaj, e, bpa). Where ``bpa`` [rad].
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
        mask = (region[0] < x) & (x < region[2]) & (region[1] < y) & (y <
                                                                      region[3])
    elif len(region) == 5:
        # Create elliptical mask
        a, b = region[0], region[1]
        n = min(shape)
        y, x = np.ogrid[-a: n - a, -b: n - b]
        bmaj = region[2]
        e = region[3]
        bpa = region[4]
        bmin = bmaj * e
        # This brings PA to VLBI-convention (- = from North counterclocwise)
        bpa = -bpa
        a = math.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
            math.sin(bpa) ** 2. / (2. * bmin ** 2.)
        b = math.sin(2. * bpa) / (2. * bmaj ** 2.) - \
            math.sin(2. * bpa) / (2. * bmin ** 2.)
        c = math.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
            math.cos(bpa) ** 2. / (2. * bmin ** 2.)
        mask = a * x ** 2 + b * x * y + c * y ** 2 <= 1
    else:
        raise Exception
    return mask


def sector_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    http://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx, y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask


def gaussian_filter(data, sigma):
    return gaussian_filter(data, sigma)


def circular_mean(data, radius):
    """
    http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array

    To apply this circular kernel to compute the mean of all the values
    surrounding each point in a 2D matrix.

    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf

    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    mask = x**2 + y**2 <= radius ** 2
    kernel[mask] = 1
    return gf(data, np.mean, footprint=kernel)


def circular_median(data, radius):
    """
    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf
    from skimage.morphology import disk

    kernel = disk(radius)
    return gf(data, np.median, footprint=kernel)


def elliptical_mean(data, bmaj, bmin, bpa):
    """
    To apply this elliptical kernel to compute the mean of all the values
    surrounding each point in a 2D matrix.

    :param data:
    :param bpa:
        Positional angle [rad].
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf

    kernel = np.zeros((4 * bmaj + 1, 4 * bmaj + 1))
    y, x = np.ogrid[-2 * bmaj: 2 * bmaj + 1, -2 * bmaj: 2 * bmaj + 1]
    bpa = -bpa + np.pi/2.
    a = math.cos(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.sin(bpa) ** 2. / (2. * bmin ** 2.)
    b = math.sin(2. * bpa) / (2. * bmaj ** 2.) - \
        math.sin(2. * bpa) / (2. * bmin ** 2.)
    c = math.sin(bpa) ** 2. / (2. * bmaj ** 2.) + \
        math.cos(bpa) ** 2. / (2. * bmin ** 2.)
    mask = a * x ** 2 + b * x * y + c * y ** 2 <= 1
    kernel[mask] = 1

    return gf(data, np.mean, footprint=kernel)


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij

    http://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k, edge_order=2)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def mask_region(data, region):
    """
    Function that masks 2D numpy array.
    :param data:
        Array-like.
    :param region:
        Tuple (blc[0], blc[1], trc[0], trc[1],) or (center[0], center[1], r,
        None,).
    :return:
        Masked 2D numpy array.
    """
    data = np.atleast_2d(data)
    mask = create_mask(data.shape, region)
    masked_array = np.ma.array(data, mask=mask)
    return masked_array


def create_grid(imsize):
    """Create meshgrid of size ``imsize``.

        :param imsize:
            Container of image dimensions
        :return:
            Meshgrid of size (imsize[0], imsize[1])
    """
    xsize, ysize = imsize
    x = np.linspace(0, xsize - 1, xsize)
    y = np.linspace(0, ysize - 1, ysize)
    x, y = np.meshgrid(x, y)
    return (x, y,)


def gen_rand_vecs(dims, number):
    """
    Function that generates random unit vectors.
    :param dims:
        Number of dimensions.
    :param number:
        Number of vectors to generate.
    :return:
        2D numpy array with shape (number, dims,)
    """
    vecs = np.random.normal(size=(number,dims))
    mags = np.linalg.norm(vecs, axis=-1)
    return vecs / mags[..., np.newaxis]


def find_close_regions(data, std_decrease_factor=1.1):
    """
    Function that finds entries of array with close elements (aka scans for time
    domain).
    :param data:
        1D numpy array with data.
    :return:
        list of lists with (first index, last index) of close regions.
    """
    maxs = np.diff(data)[np.argsort(np.diff(data))[::-1]]
    i = 0
    while np.std(maxs[i:])/np.std(maxs[i+1:]) > std_decrease_factor:
        i += 1
    threshold = maxs[i-1]
    borders = np.where((data[1:] - data[:-1]) > maxs[i-1])[0]
    print(len(borders))
    regions_list = list()
    # Append first region
    regions_list.append([data[0], data[borders[0]]])
    # Append others
    for k in range(len(borders) - 1):
        regions_list.append([data[borders[k] + 1], data[borders[k+1]]])
    # Append last
    regions_list.append([data[borders[k+1] + 1], data[-1]])

    return regions_list


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    and ``kwargs``are also included.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("vlbi_errors: Exception while calling your prior pdf:")
            print(" params:", x)
            print(" args:", self.args)
            print(" kwargs:", self.kwargs)
            print(" exception:")
            traceback.print_exc()
            raise


def ln_uniform(x, a, b):
    assert(a < b)
    if not a < x < b:
        return -np.inf
    return -math.log(b - a)


def is_sorted(lst):
    return sorted(lst) == lst

# TODO: Implement ``stokes`` argument - iterable of correlations for which to
# calculate FT.
def get_uv_correlations(uv, models):
    """
    Function that accepts models of stokes parameters in image plane and returns
    cross-correlations (whatever possible) for given instance of ``UVData``
    class.

    :param uv:
        Numpy 2d-array of (u,v)-coordinates used for calculating correlations.
    :param models:
        Iterable of ``Model`` subclass instances. There should be only one (or
        zero) model for each stokes parameter. If there are two, say I-stokes
        models, then sum them firstly using ``Model.__add__``.
    :return:
        Dictionary with keys from 'RR', 'LL', 'RL', 'LR' and values - 1d numpy
        arrays with comlex values of visibilities. Length of array equals to
        number of (u,v)-points specified in argument (that is ``len(uv)``).
    """
    # Create dictionary of type {stokes/hands: model}
    model_dict = {'I': None, 'Q': None, 'U': None, 'V': None}
    model_dict.update({model.stokes: model for model in models})
    # Dictionary with keys - 'RR', 'LL', ... and values - correlations
    uv_correlations = dict()
    # FIXME: Use exceptions (see next ``else``)
    if model_dict['I'] or model_dict['V']:
        if model_dict['I'] and model_dict['V']:
            RR = model_dict['I'].ft(uv) + model_dict['V'].ft(uv)
            LL = model_dict['I'].ft(uv) - model_dict['V'].ft(uv)
        elif not model_dict['V'] and model_dict['I']:
            RR = model_dict['I'].ft(uv)
            LL = RR.copy()
        elif not model_dict['I'] and model_dict['V']:
            RR = model_dict['V'].ft(uv)
            LL = RR.copy()
        else:
            # Actually, we shouldn't get there
            raise EmptyImageFtError('Not enough data for RR&LL visibility'
                                    ' calculation')
        # Setting up parallel hands correlations
        uv_correlations.update({'RR': RR})
        uv_correlations.update({'LL': LL})
    else:
        # Only I, Q, U, V are in model_dict by default
        if 'RR' in model_dict or 'LL' in model_dict:
            try:
                RR = model_dict['RR'].ft(uv)
                uv_correlations.update({'RR': RR})
            except KeyError:
                pass
            try:
                LL = model_dict['LL'].ft(uv)
                uv_correlations.update({'LL': LL})
            except KeyError:
                pass

    if model_dict['Q'] or model_dict['U']:
        if model_dict['Q'] and model_dict['U']:
            RL = model_dict['Q'].ft(uv) + 1j * model_dict['U'].ft(uv)
            LR = model_dict['Q'].ft(uv) - 1j * model_dict['U'].ft(uv)
            # RL = FT(Q + j*U)
            # LR = FT(Q - j*U)
            # Setting up cross hands correlations
            uv_correlations.update({'RL': RL})
            uv_correlations.update({'LR': LR})
        else:
            raise EmptyImageFtError('Not enough data for RL&LR visibility'
                                    ' calculation')

    return uv_correlations


# FIXME: Seems it doesn't work for multimodal densities.
def hdi_of_mcmc(sample, cred_mass=0.95, return_mean_median=False, mask_nan=True):
    """
    Highest density interval of sample.
    """
    assert len(sample), 'need points to find HDI'
    if mask_nan:
        sample_vec = sample[:]
        sample_vec = np.ma.array(sample_vec, mask=np.isnan(sample_vec))
    else:
        sample_vec = sample
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    if return_mean_median:
        mean = np.mean(sample_vec)
        median = np.median(sample_vec)
        return hdi_min, hdi_max, mean, median
    else:
        return hdi_min, hdi_max


def hdi_of_sample(sample, cred_mass=0.95):
    alpha = 1. - cred_mass
    return scoreatpercentile(sample, [100*alpha/2, 100*(1-alpha/2)])


def bc_endpoint(sample_vec, sample_val, alpha):
    """
    Function that calculates Bias Corrected bootstrap confidence interval endpoints.

    :param sample_vec:
        Sample of bootstrapped statistics.
    :param sample_val:
        Value of sample statistic.
    :param alpha:
        Level (0. - 1.)
    :return:
        Endpoint corresponding to ``alpha``.
    """
    n = len(sample_vec)
    import scipy
    z0 = scipy.stats.norm.ppf(float(len(sample_vec[sample_vec < sample_val])) / n)
    f = scipy.stats.norm.cdf(2. * z0 + scipy.stats.norm.ppf(alpha))
    return scipy.stats.scoreatpercentile(sample_vec, 100. * f)


def bc_endpoints(sample_vec, sample_val, alpha):
    """
    Function that calculates Bias Corrected bootstrap confidence interval endpoints.

    :param sample_vec:
        Sample of bootstrapped statistics.
    :param sample_val:
        Value of sample statistic.
    :param alpha:
        Level (0. - 1.)
    :return:
        Endpoint corresponding to ``alpha``.
    """
    n = len(sample_vec)
    import scipy
    z0 = scipy.stats.norm.ppf(float(len(sample_vec[sample_vec < sample_val])) / n)
    f = scipy.stats.norm.cdf(2. * z0 + scipy.stats.norm.ppf(alpha))
    return scipy.stats.scoreatpercentile(sample_vec, 100. * f)


def hdi_of_arrays(arrays, cred_mass=0.68):
    """
    Function that calculates a width of highest density interval for each pixel
    using user supplied arrays.
    :param arrays:
        Iterable of arrays.
    :param cred_mass: (optional)
        Credibility mass. (default: ``0.68``)
    :return:
        Numpy 2D array with widths of hdi in each pixel.
    """
    arrays = [np.atleast_2d(array) for array in arrays]
    # Check that images have the same shape
    assert len(set([array.shape for array in arrays])) == 1

    cube = np.dstack(tuple(array for array in arrays))
    hdis = np.zeros(np.shape(cube[:, :, 0]))
    for (x, y), value in np.ndenumerate(hdis):
        hdi = hdi_of_mcmc(cube[x, y, :], cred_mass=cred_mass)
        hdis[x, y] = hdi[1] - hdi[0]
    return hdis


def percent(xs, perc=None):
    """
    Find ``perc`` % in sorted container xs.
    """

    xs_ = sorted(xs)
    indx = int(math.ceil(len(xs) * perc / 100.))

    return xs_[indx]


def percent(xs, perc=None):
    """
    Find ``perc`` % in sorted container xs.
    """

    xs_ = sorted(xs)
    indx = int(math.ceil(len(xs) * perc / 100.))

    return xs_[indx]


def unwrap_phases(phases):
    """
    Function that bring phases that are subject to +/-pi*N closer.
     """
    # Bring first value to [-pi/2, pi/2] interval
    values = np.array(phases)
    if values[0] > np.pi / 2:
        values[0] -= np.pi
    elif values[0] < -np.pi / 2:
        values[0] += np.pi
    for i, value in enumerate(values[1:]):
        diff = value - values[i]
        # 0 => 0, 1 => -pi, 2 => +pi
        diff_array = np.array([abs(diff), abs(diff - np.pi),
                               abs(diff + np.pi)])
        add_dict = {0: 0, 1: -np.pi, 2: np.pi}
        values[i+1] += add_dict[np.argmin(diff_array)]

    return values


def getFromDict(dataDict, mapList):
    return reduce(lambda d, k: d[k], mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def delInDict(dataDict, mapList):
    del getFromDict(dataDict, mapList[:-1])[mapList[-1]]


def nested_dict_itervalue(nested):
    """
    http://stackoverflow.com/questions/10756427/loop-through-all-nested-dictionary-values

    :param nested:
        Nested dictionary.
    :return:
        Iteratoer of values.

    >>> list(nested_dict_iter({'a': {'b': {'c': 1, 'd': 2},
                                     'e': {'f': 3, 'g': 4}
                                     },
                               'h': {'i': 5, 'j': 6}
                               }))
    [1, 2, 3, 4, 5, 6]

    """
    for value in nested.values():
        if isinstance(value, collections.Mapping):
            for inner_value in nested_dict_itervalue(value):
                yield inner_value
        else:
            yield value


def flatten(iterable, ltypes=collections.abc.Iterable):
    """
    http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python/2158532#2158532
    :param iterable:
    :param ltypes:
    :return:
    """
    remainder = iter(iterable)
    while True:
        first = next(remainder)
        if isinstance(first, ltypes) and not isinstance(first, str):
            remainder = chain(first, remainder)
        else:
            yield first


def slice_2darray(array, pix1, pix2):
    """
    Method that returns slice of image along line.

    :param x1:
        Iterable of cordinates of first pixel.
    :param x2:
        Iterable of cordinates of second pixel.
    :return:
        Numpy array of image values for given slice.
    """
    length = int(round(np.hypot(pix2[0] - pix1[0], pix2[1] - pix1[1])))
    if pix2[0] < pix1[0]:
        x = np.linspace(pix2[0], pix1[0], length)[::-1]
    else:
        x = np.linspace(pix1[0], pix2[0], length)
    if pix2[1] < pix1[1]:
        y = np.linspace(pix2[1], pix1[1], length)[::-1]
    else:
        y = np.linspace(pix1[1], pix2[1], length)

    return array[v_round(x).astype(np.int), v_round(y).astype(np.int)]


def transform_image(image, amplitude, shift_x, shift_y, scale_factor, rotation,
                    cleaning_threshold=10**(-5)):
    if not have_skimage:
        print("Install skimage for using ``transform_image")
    imsize = image.shape[0]
    new_size = int(np.round(imsize*scale_factor))
    if new_size % 2:
        scale_factor = (1.0 + new_size)/imsize
        new_size = int(np.round(imsize * scale_factor))
    zoomed_image = interpolation.zoom(image, scale_factor)
    delta = scale_factor - 1
    if abs(delta) > 0.0001:
        if delta > 0:
            window_large = slice(new_size / 2 - imsize / 2,
                                 new_size / 2 + imsize / 2, None)
            tr_image = zoomed_image[window_large, window_large]
        else:
            tr_image = np.zeros((imsize, imsize), dtype=float)
            window_small = slice(imsize / 2 - new_size / 2,
                                 imsize / 2 + new_size / 2, None)
            tr_image[window_small, window_small] = zoomed_image
    else:
        tr_image = image.copy()
    tr_image = transform.rotate(tr_image, rotation/degree_to_rad)
    tf_shift = transform.SimilarityTransform(translation=[shift_x, shift_y])

    tr_image = transform.warp(tr_image, tf_shift)
    tr_image *= amplitude
    tr_image[tr_image < cleaning_threshold] = 0.0
    return tr_image


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
        sub = corr_matrix[max_pos[0]-n:max_pos[0]+n, max_pos[1]-n:max_pos[1]+n]
        x, y = np.mgrid[:2*n, :2*n]
        p_init = models.Gaussian2D(1, n, n, n/2, n/2, 0)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, x, y, sub)
        # TODO: Check +1
        result = p.x_mean.value-n+1, p.y_mean.value-n+1
    else:
        result = max_pos[0]-image1.shape[0]+1, max_pos[1]-image2.shape[1]+1

    return result


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


def blnoise(sefd1, sefd2, tint, bw):
    """From eht-imaging code:
        Determine the standard deviation of Gaussian thermal noise on a baseline
       This is the noise on the rr/ll/rl/lr product, not the Stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """

    noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88