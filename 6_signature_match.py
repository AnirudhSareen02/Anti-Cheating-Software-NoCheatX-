import cv2

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Couldn't read image from path: {image_path}")
    
    # Resize for consistency
    img = cv2.resize(img, (500, 250))

    # Threshold to binary
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_bin

def compare_signatures(img1_path, img2_path, show=False):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is None or des2 is None:
        return 0, "Not enough features detected."

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Max possible matches = length of smaller descriptor set
    total_possible_matches = min(len(des1), len(des2))
    good_matches = [m for m in matches if m.distance < 50]  # lower distance = better

    # Calculate percentage similarity
    match_percent = (len(good_matches) / total_possible_matches) * 100

    if show:
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, flags=2)
        cv2.imshow("Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    result = "Matched" if match_percent >= 90 else "Not Matched"
    return match_percent, result

# Example usage
if __name__ == "__main__":
    score, result = compare_signatures(
        r"C:\Users\ANIRUDH\Downloads\aastha_sign.jpg",
        r"C:\Users\ANIRUDH\Downloads\anirudh_signature.jpg",
        show=True
    )
    print(f"Match: {score:.2f}% -> {result}")
