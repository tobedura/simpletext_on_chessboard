import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = './chessboard.avi'
K = np.array([
    [1374.033,    0.0,    978.346],
    [   0.0,   1374.894, 518.438],
    [   0.0,      0.0,     1.0  ]
])
dist_coeff = np.array([[-0.076481, 0.500060, 0.003831, 0.000233, -1.056192]])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D plane for floating text (parallel to chessboard, offset in z)
text_z = -0.05  # floating distance from the board (meters)
text_3d = board_cellsize * np.array([
    [3, 2, text_z / board_cellsize],
    [5, 2, text_z / board_cellsize],
    [5, 5, text_z / board_cellsize],
    [3, 5, text_z / board_cellsize]
], dtype=np.float64)

# Create text image to warp onto the plane
text_img_w, text_img_h = 400, 200
text_img = np.zeros((text_img_h, text_img_w, 4), dtype=np.uint8)  # BGRA with alpha
cv.putText(text_img, 'BOARD', (20, 130), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255, 255), 6)
text_img = cv.rotate(text_img, cv.ROTATE_90_CLOCKWISE)
text_img_h, text_img_w = text_img.shape[:2]
text_corners = np.array([[0, 0], [text_img_w, 0], [text_img_w, text_img_h], [0, text_img_h]], dtype=np.float32)

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Warp floating text onto the image
        projected, _ = cv.projectPoints(text_3d, rvec, tvec, K, dist_coeff)
        dst_corners = projected.reshape(-1, 2).astype(np.float32)
        H = cv.getPerspectiveTransform(text_corners, dst_corners)
        warped = cv.warpPerspective(text_img, H, (img.shape[1], img.shape[0]))
        mask = warped[:, :, 3] > 0
        img[mask] = warped[mask, :3]

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()