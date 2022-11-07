import numpy as np
import cv2

def draw_major_minor(target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end, lineThickness=2):
    target = cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
    target = cv2.line(
        target,
        (int(major_axis_end[0]), int(major_axis_end[1])),
        (int(major_axis[0]), int(major_axis[1])),
        (0, 0, 255),
        lineThickness,
    )
    target = cv2.line(
        target,
        (int(minor_axis_end[0]), int(minor_axis_end[1])),
        (int(minor_axis[0]), int(minor_axis[1])),
        (0, 255, 0),
        lineThickness,
    )
    return target

def calculate_difference_map(target, base):
        diff = abs(target/255. - base/255.)**2.
        diff[diff < 0.] = (diff[diff < 0.] * 0.5)
        diff[diff < 0.] = 0.
        if len(diff.shape) > 2:
            diff = np.mean(np.abs(diff), axis=-1)
        return diff

def get_contours(target):
    mask = ((np.abs(target) > 0.008) * 255).astype(np.uint8)
    kernel = np.ones((16, 16), np.uint8)
    mask = cv2.erode(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compute_contact_area(contours, contour_threshold):
    cnt_list = []
    for contour in contours:
        if len(contour) > contour_threshold:
            cnt_list.append(contour)
    if len(cnt_list) > 0:
        contour = np.concatenate(cnt_list, axis=0)
        ellipse = cv2.fitEllipse(contour)
        poly = cv2.ellipse2Poly(
            (int(ellipse[0][0]), int(ellipse[0][1])),
            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
            int(ellipse[2]),
            0,
            360,
            5,
        )
        center = np.array([ellipse[0][0], ellipse[0][1]])
        a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
        theta = (ellipse[2] / 180.0) * np.pi
        major_axis = np.array(
            [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
        )
        minor_axis = np.array(
            [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
        )
        major_axis_end = 2 * center - major_axis
        minor_axis_end = 2 * center - minor_axis
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end, center
    else:
        return None
    
def contact_area(target, base, contour_threshold=10):
    diffmap = calculate_difference_map(target, base)
    contours = get_contours(diffmap)
    output = compute_contact_area(contours, contour_threshold)
    if output is None:
        return diffmap, None
    else:
        (
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
            center
        ) = output
        return diffmap, (poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), center)

def compute_multiple_contact_area(contours, contour_threshold):
    cnt_list = []
    print(np.size(contours))
    for contour in contours:
        if len(contour) > contour_threshold:
            cnt_list.append(contour)
    poly = []
    major_axis = []
    major_axis_end = []
    minor_axis = []
    minor_axis_end = []
    centers = []
    if len(cnt_list) > 0: 
        for contour in contours:
            if len(contour) > 75:
                ellipse = cv2.fitEllipse(contour)
                poly.append(cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                ))
                center = np.array([ellipse[0][0], ellipse[0][1]])
                centers.append(np.array([ellipse[0][0], ellipse[0][1]]))
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis.append(np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                ))
                minor_axis.append(np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                ))
                major_axis_end.append(2 * center - major_axis[-1])
                minor_axis_end.append(2 * center - minor_axis[-1])
        
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end, centers
    else:
        return None
    
def multiple_contact_area(target, base, contour_threshold=10):
    diffmap = calculate_difference_map(target, base)
    contours = get_contours(diffmap)
    output = compute_multiple_contact_area(contours, contour_threshold)
    if output is None:
        return diffmap, None
    else:
        (
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
            centers
        ) = output
        return diffmap, (poly, (major_axis, major_axis_end), (minor_axis, minor_axis_end), centers)
