from sympy import Point, Line, Segment
from skimage import feature, measure
import numpy as np
from scipy import ndimage as nd

def thr_calculator(filtered_img,min_distance=3,stringency=0):

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
                           
            else:
                selected_thr = 0
                calculated_thr = 0
                selected_peaks = 0
                peaks_coords = 0
                total_peaks = 0
                thr_idx = 0
                selected_peaks_int = 0
        else:
            selected_thr = 0
            calculated_thr = 0
            selected_peaks = 0
            peaks_coords = 0
            total_peaks = 0
            thr_idx = 0
            selected_peaks_int = 0


    else:
        selected_thr = 0
        calculated_thr = 0
        selected_peaks = 0
        peaks_coords = 0
        total_peaks = 0
        thr_idx = 0
        selected_peaks_int = 0

    counting_dict={}

    counting_dict['selected_thr'] = selected_thr
    counting_dict['calculated_thr'] = calculated_thr
    counting_dict['selected_peaks'] = selected_peaks
    counting_dict['thr_array'] = thr_array
    counting_dict['peaks_coords'] = peaks_coords
    counting_dict['total_peaks'] = total_peaks 
    counting_dict['thr_idx'] = thr_idx
    counting_dict['stringency'] = stringency
    counting_dict['selected_peaks_int'] = selected_peaks_int
    
    return counting_dict
