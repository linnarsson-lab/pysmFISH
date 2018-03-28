from sympy import Point, Line, Segment
from skimage import feature, measure
import numpy as np
from scipy import ndimage as nd

def thr_calculator(filtered_img,min_distance,stringency):

    """
    Function used to calculate the threshold to use for the dots
    counting in a 2D image. 

    Parameters:
    -----------

    filtered_img: np.array float64
        preprocessed image used to count the dots.
    min_distance: int
        minimum distance that two maxima need to have in order to be defined as 
        separete peaks.
    stringency: int
        integer used to select the stringency of the generated
        threshold. By adding stringency to the thr_idx we can select a Thr with higher
        value from the thr_array.

    Returns:
    -----------

    counting_dict : dict 
        dictionary containing all the counting infos:
        selected_thr: float64
            Thr used for counting after application of the stringency.
        calculated_thr: float64 
            Calculated Thr
        selected_peaks: int64 
            2D coords of the peaks defined using the selected_thr.
        thr_array: float64 
            Thr array of 100 points distributed between (Img.min(),Img.max()).
        peaks_coords: float64 
            list of all the 3D coords calculated using the Thr array.
        total_peaks: list of int 
            List of the peaks counts.
        thr_idx: int64 
            index of the calculated threshold.
        stringency: int64 
            stringency used for the identification of the selected_peaks
    """
    
    # List with the total peaks calculated for each threshold
    total_peaks = []
    
    # List of ndarrays with the coords of the peaks calculated for each threshold
    peaks_coords = []

    # Define the Thr array to be tested
    thr_array = np.linspace(filtered_img.min(),filtered_img.max(),num=100)


    # Calculate the number of peaks for each threshold. In this calculation
    # the size of the objects is not considered
    for thr in thr_array:
        # The border is excluded from the counting
        peaks = feature.peak_local_max(filtered_img,min_distance=min_distance,\
            threshold_abs=thr,exclude_border=False, indices=True,\
            num_peaks=np.inf, footprint=None,labels=None)    
        # Stop the counting when the number of peaks detected falls below 3
        if len(peaks)<=3:
            stop_thr = thr # Move in the upper loop so you will stop at the previous thr
            break
        else:
            peaks_coords.append(peaks) 
            total_peaks.append(len(peaks))


    # Consider the case of no detectected peaks or if there is only one Thr
    # that create peaks (list total_peaks have only one element and )
    # if np.array(total_peaks).sum()>0 or len(total_peaks)>1:
    if len(total_peaks)>1:

        # Trim the threshold array in order to match the stopping point
        # used the [0][0] to get the first number and then take it out from list
        thr_array = thr_array[:np.where(thr_array==stop_thr)[0][0]]


        # Calculate the gradient of the number of peaks distribution
        grad = np.gradient(total_peaks)
        
        # Restructure the data in order to avoid to consider the min_peak in the
        # calculations

        # Coord of the gradient min_peak
        grad_min_peak_coord = np.argmin(grad)
        
        # Trim the data to remove the peak.
        trimmed_thr_array = thr_array[grad_min_peak_coord:]
        trimmed_grad = grad[grad_min_peak_coord:]

        if trimmed_thr_array.shape>(1,):

            # Trim the coords array in order to maintain the same length of the 
            # tr and pk
            trimmed_peaks_coords = peaks_coords[grad_min_peak_coord:]
            trimmed_total_peaks = total_peaks[grad_min_peak_coord:]

            # To determine the threshold we will determine the Thr with the biggest
            # distance to the segment that join the end points of the calculated
            # gradient

            # Distances list
            distances = []

            # Calculate the coords of the end points of the gradient
            p1 = Point(trimmed_thr_array[0],trimmed_grad[0])
            p2 = Point(trimmed_thr_array[-1],trimmed_grad[-1])
            
            # Create a line that join the points
            s = Line(p1,p2)
            allpoints = np.arange(0,len(trimmed_thr_array))
            
            # Calculate the distance between all points and the line
            for p in allpoints:
                dst = s.distance(Point(trimmed_thr_array[p],trimmed_grad[p]))
                distances.append(dst.evalf())

            # Remove the end points from the lists
            trimmed_thr_array = trimmed_thr_array[1:-1]
            trimmed_grad = trimmed_grad[1:-1]
            trimmed_peaks_coords = trimmed_peaks_coords[1:-1]
            trimmed_total_peaks = trimmed_total_peaks[1:-1]
            trimmed_distances = distances[1:-1]
        
            # Determine the coords of the selected Thr
            # Converted trimmed_distances to array because it crashed
            # on Sanger.
            if trimmed_distances: # Most efficient way will be to consider the length of Thr list
                thr_idx=np.argmax(np.array(trimmed_distances))
                calculated_thr = trimmed_thr_array[thr_idx]
                # The selected threshold usually causes oversampling of the number of dots
                # I added a stringency parameter (int n) to use to select the Thr+n 
                # for the counting. It selects a stringency only if the trimmed_thr_array
                # is long enough
                if thr_idx+stringency<len(trimmed_thr_array):
                    selected_thr = trimmed_thr_array[thr_idx+stringency]
                    selected_peaks = trimmed_peaks_coords[thr_idx+stringency]
                    thr_idx = thr_idx+stringency
                else:
                    selected_thr = trimmed_thr_array[thr_idx]
                    selected_peaks = trimmed_peaks_coords[thr_idx]

                
                # Calculate the selected peaks after removal of the big and small objects
                
                # Threshold the image using the selected threshold
                if selected_thr>0:
                    img_mask = filtered_img>selected_thr
                
                labels = nd.label(img_mask)[0]
                
                properties = measure.regionprops(labels)
                    
                for ob in properties:
                    if ob.area<6 or ob.area>200:
                        img_mask[ob.coords[:,0],ob.coords[:,1]]=0
                
                labels = nd.label(img_mask)[0]
                selected_peaks = feature.peak_local_max(filtered_img, min_distance=min_distance, threshold_abs=selected_thr, exclude_border=False, indices=True, num_peaks=np.inf, footprint=None, labels=labels)
                
                if selected_peaks.size:
                    # Intensity counting of the max peaks
                    selected_peaks_int = filtered_img[selected_peaks[:,0],selected_peaks[:,1]]
                
                
                else:
                    selected_thr = 0
                    calculated_thr = 0
                    selected_peaks = 0
                    peaks_coords = 0
                    total_peaks = 0
                    thr_idx = 0
                    selected_peaks_int = 0
                    trimmed_thr_array = 0
                    trimmed_peaks_coords = 0
                           
            else:
                selected_thr = 0
                calculated_thr = 0
                selected_peaks = 0
                peaks_coords = 0
                total_peaks = 0
                thr_idx = 0
                selected_peaks_int = 0
                trimmed_thr_array = 0
                trimmed_peaks_coords = 0
        else:
            selected_thr = 0
            calculated_thr = 0
            selected_peaks = 0
            peaks_coords = 0
            total_peaks = 0
            thr_idx = 0
            selected_peaks_int = 0
            trimmed_thr_array = 0
            trimmed_peaks_coords = 0


    else:
        selected_thr = 0
        calculated_thr = 0
        selected_peaks = 0
        peaks_coords = 0
        total_peaks = 0
        thr_idx = 0
        selected_peaks_int = 0
        trimmed_thr_array = 0
        trimmed_peaks_coords = 0

    counting_dict={}

    counting_dict['selected_thr'] = selected_thr
    counting_dict['calculated_thr'] = calculated_thr
    counting_dict['selected_peaks'] = selected_peaks
    counting_dict['thr_array'] = thr_array
    counting_dict['trimmed_thr_array'] = trimmed_thr_array
    counting_dict['peaks_coords'] = peaks_coords
    counting_dict['trimmed_peaks_coords'] = trimmed_peaks_coords
    counting_dict['total_peaks'] = total_peaks 
    counting_dict['thr_idx'] = thr_idx
    counting_dict['stringency'] = stringency
    counting_dict['selected_peaks_int'] = selected_peaks_int
    
    return counting_dict


import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from skimage.feature import peak_local_max
from skimage.filters import rank_order


def peak_thrs_local_max(image, min_distance=1, threshold_abs=None,
                        threshold_rel=None, exclude_border=True, indices=True,
                        num_peaks=np.inf, footprint=None, labels=None):
    """
    Function after modification:
    returns the coordinates for a range of thresholds

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int, optional
        If nonzero, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates.  If False, the output will be a boolean array shaped as
        `image.shape` with peaks present at True elements.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance` (also for `exclude_border`).
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=0)
    array([[10, 10, 10]])

    """

    if type(exclude_border) == bool:
        exclude_border = min_distance if exclude_border else 0

    out = np.zeros_like(image, dtype=np.bool)

    # In the case of labels, recursively build and return an output
    # operating on each label separately
    if labels is not None:
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        # New values for new ordering
        label_values = np.unique(labels)
        for label in label_values[label_values != 0]:
            maskim = (labels == label)
            out += peak_local_max(image * maskim, min_distance=min_distance,
                                  threshold_abs=threshold_abs,
                                  threshold_rel=threshold_rel,
                                  exclude_border=exclude_border,
                                  indices=False, num_peaks=np.inf,
                                  footprint=footprint, labels=None)

        if indices is True:
            return np.transpose(out.nonzero())
        else:
            return out.astype(np.bool)

    if np.all(image == image.flat[0]):
        if indices is True:
            return np.empty((0, 2), np.int)
        else:
            return out

    # Non maximum filter
    if footprint is not None:
        image_max = ndi.maximum_filter(image, footprint=footprint,
                                       mode='constant')
    else:
        size = 2 * min_distance + 1
        image_max = ndi.maximum_filter(image, size=size, mode='constant')
    mask = image == image_max

    if exclude_border:
        # zero out the image borders
        for i in range(mask.ndim):
            mask = mask.swapaxes(0, i)
            remove = (footprint.shape[i] if footprint is not None
                      else 2 * exclude_border)
            mask[:remove // 2] = mask[-remove // 2:] = False
            mask = mask.swapaxes(0, i)

    # find top peak candidates above a threshold
    thresholds = []
    if threshold_abs is None:
        threshold_abs = image.min()
    thresholds.append(threshold_abs)
    if threshold_rel is not None:
        thresholds.append(threshold_rel * image.max())
    if thresholds:
        mask_original = mask  # save the local maxima's of the image
        thrs_coords = {}  # dictionary holds the coordinates correspond for each threshold
        for threshold in thresholds[0]:
            mask = mask_original
            mask &= image > threshold

            # get coordinates of peaks
            coordinates = np.transpose(mask.nonzero())

            if coordinates.shape[0] > num_peaks:
                intensities = image.flat[np.ravel_multi_index(coordinates.transpose(),
                                                              image.shape)]
                idx_maxsort = np.argsort(intensities)[::-1]
                coordinates = coordinates[idx_maxsort][:num_peaks]

            if indices is True:
                thrs_coords[threshold] = coordinates
            else:
                nd_indices = tuple(coordinates.T)
                out[nd_indices] = True
                return out
    if thresholds and thrs_coords:
        return thrs_coords
