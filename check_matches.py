import cv2
import numpy as np
import os

def load_images(left_folder, right_folder):
    left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    return left_images, right_images

def feature_matching(left_img_path, right_img_path):
    # Load images
    img1 = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    print(len(matches))
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def display_images(left_img_path, right_img_path, img_matches):
    # Load images
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    # Concatenate images side by side
    img_concat = np.hstack((img1, img2))

    # Display the images
    cv2.imshow('Stereo Images', img_concat)
    cv2.imshow('Feature Matches', img_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(left_folder, right_folder):
    left_images, right_images = load_images(left_folder, right_folder)

    for left_img, right_img in zip(left_images, right_images):
        img_matches = feature_matching(left_img, right_img)
        display_images(left_img, right_img, img_matches)

if __name__ == "__main__":
    left_folder = 'data/images/left'
    right_folder = 'data/images/right'
    main(left_folder, right_folder)
