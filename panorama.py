import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def findPoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def matchFeatures(des1, des2, img1, img2):
    topmatches = list()
    idx1 = 0
    for i in des1:
        allDis = list()
        idx2 = 0
        for j in des2:
            d = cv2.norm(i, j)
            item = [d, idx1, idx2]
            idx2 += 1
            allDis.append(item)
        idx1 += 1
        allDis.sort()
        topmatches.append(allDis[0:2])
    return topmatches


def goodMatches(matches):
    good = []
    for m, n in matches:
        # print("m[0]= ", m[0], " ,n[0]= ", n[0])
        if m[0] < 0.5 * n[0]:
            dMatchObj = cv2.DMatch()
            dMatchObj.imgIdx = 0
            dMatchObj.distance = m[0]
            dMatchObj.trainIdx = m[1]
            dMatchObj.queryIdx = m[2]
            good.append(dMatchObj)
    return good


def findHomography(randFour):
    homoList = list()
    # print("randFour values = ", randFour)

    for pt in randFour:
        # print("pt.item(0) ", pt.item(0))
        xVal = [-pt.item(0), -pt.item(1), -1, 0, 0, 0, pt.item(2) * pt.item(0), pt.item(2) * pt.item(1), pt.item(2)]
        yVal = [0, 0, 0, -pt.item(0), -pt.item(1), -1, pt.item(3) * pt.item(0), pt.item(3) * pt.item(1), pt.item(3)]
        homoList.append(xVal)
        homoList.append(yVal)

    homoMat = np.matrix(homoList)
    u, s, v = np.linalg.svd(homoMat)
    h = np.reshape(v[8], (3, 3))
    h = (1 / h.item(8)) * h
    return h


def calcDist(i, homo):
    p1 = np.transpose(np.matrix([i[0].item(0), i[0].item(1), 1]))
    estimatep2 = np.dot(homo, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2
    p2 = np.transpose(np.matrix([i[0].item(2), i[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def ransacAlgo(coorMat):
    maxInlier = []
    flag = 0
    for j in range(0, 1000):
        randFour = []
        for i in range(0, 4):
            randmatch = random.choice(coorMat)
            # if flag == 0:
            #     print("Randmatch = ", randmatch)
            randFour.append(randmatch)
            # print("randFour 1111: ", randFour)
        # if flag == 0:
        #     print("randFour after all appending: ", randFour)
        # flag = 1
        homo = findHomography(randFour)
        # print("Homography function output: ", homo)
        inlier = list()
        for i in coorMat:
            dist = calcDist(i, homo)

            if dist < 5:
                inlier.append(i)

        if len(inlier) > len(maxInlier):
            maxInlier = inlier
            H = homo
    # print("Final H in function = ", H)
    # print("H size: ", H.shape)
    # print("H.item(0)= ", H.item(0))
    return maxInlier, H


def imageStich(H, img_r, imgr):
    dst = cv2.warpPerspective(img_r, H, (img_r.shape[1] + imgr.shape[1], imgr.shape[0]))
    # cv2.imshow("warp_result.jpg", dst)
    dst[0:imgr.shape[0], 0:imgr.shape[1]] = imgr
    return dst


def stitch(imga, imgb, count):
    # if count == 0:
    #     img_r = cv2.resize(imga, (0, 0), fx=0.15, fy=0.15)
    # elif count == 1:
    #     img_r = cv2.resize(imga, (0, 0), fx=0.15, fy=0.15)
    if count == 2:
        img_r = cv2.resize(imga, (0, 0), fx=0.5, fy=1)
        imgr = cv2.resize(imgb, (0, 0), fx=0.5, fy=1)
    else:
        img_r = imga
        imgr = imgb
    img1 = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    # if count == 0:
    #     imgr = cv2.resize(imgb, (0, 0), fx=0.15, fy=0.15)
    # elif count == 1:
    #     imgr = cv2.resize(imgb, (0, 0), fx=0.5, fy=1)



    """............Detecting keypoints and descriptors............."""
    kp1, des1 = findPoints(img1)
    kp2, des2 = findPoints(img2)
    print("Keypoint detected")
    keyImage = cv2.drawKeypoints(img_r, kp1, None)
    cv2.imshow('image_keypoints', keyImage)
    cv2.imwrite('image]_keypoints.jpg', keyImage)

    """............Matching keypoints in both images................"""
    matches = matchFeatures(des1, des2, img_r, imgr)
    print("matching done")

    """........Finding good matches.............."""
    good = goodMatches(matches)
    # draw_good = []
    # for g in good:
        # print("g[0] ", g[0])
        # draw_good.append([g[1], g[2], g[0]])
    # print("Length of good = ", len(good))
    # print("drawGood: ", draw_good)

    """......... Drawing matching lines ............."""
    if count == 0:
        draw_params = dict(matchColor = (255,0,0),
                           singlePointColor = None,
                           flags = 2)
        matchImg = cv2.drawMatches(imgr, kp2, img_r, kp1, good, None, **draw_params)
        cv2.imshow("original_image_drawMatches.jpg", matchImg)
        cv2.imwrite("Matched.jpg", matchImg)

    MIN_MATCH_COUNT = 0
    if len(good) > MIN_MATCH_COUNT:
        coordList = list()
        for m in good:
            (x1, y1) = kp1[m.trainIdx].pt
            (x2, y2) = kp2[m.queryIdx].pt
            coordList.append([x1, y1, x2, y2])
        # print("coordlist[1] = ", coordList[1])
        coordMat = np.matrix(coordList)
        # print("coordMat = ", coordMat[0])

        """........Using Ransac to calculate Final Homography Matrix............."""
        maxInlier, H = ransacAlgo(coordMat)
        print("Homography matrix H = ", H)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    """............Stitching the Images using Homography Matrix............"""
    dst = imageStich(H, img_r, imgr)
    return dst

def main():
    """...........Reading images................."""
    img_left = cv2.imread("pillar_left.jpeg")
    img_middle = cv2.imread("pillar_mid.jpeg")
    img_right = cv2.imread("pillar_right.jpeg")

    def trim(frame):
        # countTrim += 1
        print("entering trim")
        # crop top
        if not np.sum(frame[0]):
            print("top crop")
            return trim(frame[1:])
        # crop bottom
        elif not np.sum(frame[-1]):
            print("bottom crop")
            return trim(frame[:-2])
        # crop left
        elif not np.sum(frame[:, 0]):
            print("left crop")
            return trim(frame[:, 1:],)
        # crop right
        elif not np.sum(frame[:, -1]):
            print("right crop")
            return trim(frame[:, :-73],)
        return frame


    count = 0
    pan1 = stitch(img_middle, img_left, count)
    pan1 = trim(pan1)
    count += 1
    pan2 = stitch(img_right, img_middle, count)
    pan2 = trim(pan2)
    count += 1
    pan3 = stitch(pan2, pan1, count)
    pan3 = trim(pan3)

    # pan3 = stitch(pan2, pan1)
    # dst = stitch(img_right, pan1)



    # countTrim = 0

    cv2.imshow("Pan1.jpg", pan1)
    cv2.imwrite("pan1.jpg", pan1)

    cv2.imshow("Pan2.jpg", pan2)
    cv2.imwrite("pan1.jpg", pan1)

    cv2.imshow("Final Panorama: ", pan3)
    cv2.imwrite("pan3.jpg", pan3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
