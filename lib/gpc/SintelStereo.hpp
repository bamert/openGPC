// Copyright (c) 2018, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Niklaus Bamert (bamertn@ethz.ch)
#ifndef _GPC_SintelStereo
#define _GPC_SintelStereo

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <random>
/* Needed to scan through directory (Sintel set)*/
#include <dirent.h>
#include <fstream>
#include <gpc/buffer.hpp>

using namespace std;

namespace gpc {
namespace datasource {

/**
 *
 @brief      The Sintel dataset stereo dataset

*/
class SintelStereo {
private:
    typedef typename gpc::training::Feature F;
    typedef typename F::GPCPatchTriplet GPCTriplet_t;
    bool canDoExtraction = false;
    /**
     * @brief Checks dataset location sanity.
     *        Not tested on windows
     *
     * @param path
     *
     * @return  true if path points to a directory
     */
    bool isDir(std::string path) {
        struct stat info;
        if (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR))
            return true;
        else
            return false;
    }

public:
    SintelStereo(std::string basePath) {
        if (basePath.back() != '/') basePath += "/";
        // setup paths
        cleanLeftDir = basePath + "training/clean_left";
        cleanRightDir = basePath + "training/clean_right";
        dispDir = basePath + "training/disparities";
        oclDir = basePath + "training/occlusions";
        oofDir = basePath + "training/outofframe";

        // count images in first scene
        numFrames = countImages();
        canDoExtraction = true;
    }
    SintelStereo() { canDoExtraction = false; }
    /**
     * @brief extract training dataset made up of triplets (reference, positive, negative)
     *        image patches.
     *
     * @param numTripletsPerPair  number of triplets to sample per image pair
     * @param radiusLower         lower sampling radius for negative patch
     * @param radiusUpper         upper sampling radius for negative patch
     *
     * @return The triplet set
     */
    std::vector<GPCTriplet_t> extractTrainingData(
        int numTripletsPerPair, int radiusLower, int radiusUpper) {
        std::vector<GPCTriplet_t> trainingData;
        if (canDoExtraction == false) {
            cout << "ERR: No path for Sintel dataset specified" << endl;
            return trainingData;
        }
        // Verify directory structure for Sintel Stereo dataset
        if (!(isDir(cleanLeftDir) && isDir(cleanRightDir) && isDir(dispDir) &&
              isDir(oclDir) && isDir(oofDir))) {
            cout << "ERR: This does not look like the Sintel Stereo dataset. Please "
                    "verify paths."
                 << endl;
            return trainingData;
        }

        // cycle through scenes
        for (int sceneId = 0; sceneId < 20; sceneId++) {
            selectScene(sceneId);
            int numImages = countImages();
            // cycle through images in given scene
            for (int imgId = 1; imgId < numImages - 1; imgId++) {
                std::vector<ndb::Point> kptsL, kptsR, kptsN;
                // Keep the necessary images
                ndb::Buffer<uint8_t> occ, oof, imgL, imgR;
                // ndb::Buffer<ndb::RGBColor> disp;
                ndb::RGBBuffer disp;
                // Get images and disparity
                try {
                    int err = 0;
                    Eigen::MatrixXd u, v;
                    err |= getBW(imgId, imgL, imgR);
                    err |= getDisparity(imgId, disp);
                    err |= getOcclusion(imgId, occ);
                    err |= getInvalid(imgId, oof);
                    if (err)
                        throw std::invalid_argument(
                            "could not open dataset file. Verify paths to Sintel dataset "
                            "are set correctly.");

                    // Get Keypoint coordinate lists for given image pair
                    getGroundTruthMatches(
                        disp, oof, occ, numTripletsPerPair, radiusLower, radiusUpper,
                        kptsL, kptsR, kptsN);
                    // Extract features under Feature requested
                    Feature.extractAllTriplets(
                        imgL, imgR, kptsL, kptsR, kptsN, trainingData);
                } catch (const std::invalid_argument& e) {
                }
            }  // image loop
        }      // scene loop
        std::random_shuffle(trainingData.begin(), trainingData.end());
        return trainingData;
    }

    /**
     * @brief Store a training set to file. Note that the format used
     *        may not be not portable between machines of different endianness.
     *
     * @param data
     * @param path
     */
    void storeTrainingData(std::vector<GPCTriplet_t>& data, std::string path) {
        Feature.storeAllTriplets(data, path);
    }
    /**
     * @brief Read training set from file that has previously been extracted
     *
     * @param the path to the file
     *
     * @return  the training set
     */
    std::vector<GPCTriplet_t> loadTrainingData(std::string path) {
        struct stat buffer;
        if (stat(path.c_str(), &buffer) != 0) {
            std::vector<GPCTriplet_t> emptyset;
            cout << "ERR: No extracted training set found at given path" << endl;
            return emptyset;
        } else {
            return Feature.loadAllTriplets(path);
        }
    }

private:
    std::string dispDir, cleanLeftDir, cleanRightDir, oclDir, oofDir;
    std::string selectedScene = "alley_1";  // This is the default scene
    std::vector<std::string> sceneNames = {
        "alley_1",  "alley_2",    "ambush_2",   "ambush_4",  "ambush_5",   "ambush_6",
        "ambush_7", "bamboo_1",   "bamboo_2",   "bandage_1", "bandage_2",  "cave_2",
        "cave_4",   "market_2",   "market_5",   "market_6",  "mountain_1", "shaman_2",
        "shaman_3", "sleeping_1", "sleeping_2", "temple_2",  "temple_3"};
    // Instantiate a feature for the feature transform
    // some of them offer
    F Feature;
    // Image Caches
    ndb::Buffer<uint8_t> cacheL, cacheR;

    int numScenes = 23;
    int numFrames = 50;
    /**
     * @brief      Determines if the given coordinates are safe to extract a patch
     *             of size 7x7 from when the given coords are the center of the
     *             7x7 patch. Here it is unimportant whether this patch is also
     *             visible in the original pixel since this method is only being
     *             used to determine whether a (deliberate) negative patch is in
     *             the frame.
     *
     * @param[in]  x          x coord of patch center
     * @param[in]  y          y coord of patch center
     * @param[in]  patchSize  The patch size (side length, e.g. 7)
     * @param[in]  width      The width
     * @param[in]  height     The height
     *
     * @return     True if safe patch center, False otherwise.
     */
    inline bool isSafePatchCenter(int x, int y, int width, int height) {
        if (x > 20 && y > 20 && x < (width - 21) && y < (height - 21))
            return true;
        else
            return false;
    }
    /**
     * @brief      Counts the number of images in the current scene.
     *             It does this in a very crude way, simply couting all the f
     *
     * @return     Number of images.
     */
    int countImages(void) {
        std::string currFname;
        int cnt = 0;
        DIR* dir;
        struct dirent* ent;
        std::string selectedScenePath = cleanLeftDir + "/" + selectedScene;
        if ((dir = opendir(selectedScenePath.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                currFname = ent->d_name;
                if (currFname.length() >= 3 &&
                    currFname.substr(currFname.length() - 3) == "png")
                    cnt++;
            }
            closedir(dir);
            return cnt;
        } else {
            /* could not open directory */
            std::cout << "ERR:couldn't open directory" << std::endl;
            return 0;
        }
    }

    /**
     * @brief      select a scene from the Sintel set by name
     *
     * @param[in]  sceneName  The scene name
     *
     * @return     0 if success
     */
    int selectScene(std::string sceneName) {
        if (std::find(sceneNames.begin(), sceneNames.end(), sceneName) !=
            sceneNames.end()) {
            selectedScene = sceneName;
            return 0;
        } else {
            std::cout << "ERR:Scene with name (" << sceneName << ") was not found"
                      << std::endl;
            return 1;
        }
    }

    /**
     * @brief      select a scene from the Sintel set by index
     *
     * @param[in]  idx   The index
     *
     * @return     { 0 if success}
     */
    int selectScene(int idx) {
        if (idx > numScenes - 1) return 1;
        selectedScene = sceneNames[idx];
        numFrames = countImages();
        cout << "Scene name:" << selectedScene << " (" << numFrames << " imgs)"
             << std::endl;
        return 0;
    }

    /**
     * @brief      Gets the bw.
     *
     * @param[in]  id    The identifier
     * @param      L     Left image
     * @param      R     Right image
     *
     * @return     The bw.
     */
    int getBW(int id, ndb::Buffer<uint8_t>& L, ndb::Buffer<uint8_t>& R) {
        char buf[16];

        sprintf(buf, "%04d", id);
        int err1 =
            L.readPNG(cleanLeftDir + "/" + selectedScene + "/frame_" + buf + ".png");
        int err2 =
            R.readPNG(cleanRightDir + "/" + selectedScene + "/frame_" + buf + ".png");

        return err1 | err2;
    }

    /**
     * @brief      Gets the rgb.
     *
     * @param[in]  id    The identifier
     * @param      L     Left image
     * @param      R     Right image
     *
     * @return     The rgb.
     */
    int getRGB(int id, ndb::Buffer<uint8_t>& L, ndb::Buffer<uint8_t>& R) {
        char buf[16];
        sprintf(buf, "%04d", id);
        int err1 =
            L.readPNG(cleanLeftDir + "/" + selectedScene + "/frame_" + buf + ".png");
        int err2 =
            R.readPNG(cleanRightDir + "/" + selectedScene + "/frame_" + buf + ".png");

        return err1 | err2;
    }

    /**
     * @brief      Gets the occlusion map
     *
     * @param[in]  id    The image id
     * @param      O     occlusion image
     *
     * @return     0 if succecss.
     */
    int getOcclusion(int id, ndb::Buffer<uint8_t>& O) {
        char buf[16];
        sprintf(buf, "%04d", id);
        return O.readPNG(oclDir + "/" + selectedScene + "/frame_" + buf + ".png");
    }
    /**
     * @brief      Gets the disparity map
     *
     * @param[in]  id    The image id
     * @param      D     disparity image
     *
     * @return     0 if success.
     */
    int getDisparity(int id, ndb::RGBBuffer& D) {
        char buf[16];
        sprintf(buf, "%04d", id);
        return D.readPNGRGB(dispDir + "/" + selectedScene + "/frame_" + buf + ".png");
    }

    /**
     * @brief      Gets the invalid pixel map (out of frame)
     *
     * @param[in]  id    The image id
     * @param      I     Invalid pixel map
     *
     * @return     0 if success
     */
    int getInvalid(int id, ndb::Buffer<uint8_t>& I) {
        char buf[16];
        sprintf(buf, "%04d", id);
        return I.readPNG(oofDir + "/" + selectedScene + "/frame_" + buf + ".png");
    }

    /**
     * @brief      Gets the ground truth matches from a pair of images in the
     *             Sintel Set. These are generated from the given disparity images
     *             in the dataset Out of frame or occluded pixels are excluded
     *             from the set of keypoint pairs.
     *
     *             A vector of DMatch is returned s.t. the matches can be
     *             evaluated visually with DrawMatches.
     *
     * @param      disp         disparity image
     * @param      oof          out of frame pixels
     * @param      occ          occluded pixels
     * @param      numKpts      number of keypoints to get
     * @param      radiusLower  lower radius of annulus to generate negative patch
     * @param      radiusUpper  upper radius of annulus to generate negative patch
     * @param      kptsL        The keypoints in the left image (ref)
     * @param      kptsR        The keypoints in the right image (pos)
     * @param      kptsN        The keypoints in the right image (neg)
     *

     * @return     <>0 if failed
     */

    int getGroundTruthMatches(
        ndb::RGBBuffer& disp,
        ndb::Buffer<uint8_t>& oof,
        ndb::Buffer<uint8_t>& occ,
        int numKpts,
        int radiusLower,
        int radiusUpper,
        std::vector<ndb::Point>& kptsL,
        std::vector<ndb::Point>& kptsR,
        std::vector<ndb::Point>& kptsN) {
        // Image dimensions are constant
        int width = 1024;
        int height = 436;

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> randX(0, width - 1), randY(0, height - 1);

        // To sample negative patches
        std::uniform_int_distribution<int> randOffset(radiusLower, radiusUpper),
            signum(-1, 1);
        std::uniform_real_distribution<> rej(0, 1);

        // Completely randomize patch coordinates
        while (kptsL.size() < numKpts) {
            int xCoord = randX(rng);
            int yCoord = randY(rng);

            ndb::RGBColor encodedDisp = disp.getPixel(xCoord, yCoord);
            int disparityGroundTruth = (4 * encodedDisp.r + encodedDisp.g / 64);

            int rightX = xCoord - disparityGroundTruth;
            double alpha = 0.5;
            double rejectionProp =
                (15 - std::min(abs(disparityGroundTruth), 15)) / 15 * alpha;

            // Make sure this coordinate is not occluded, out of frame and that it is
            // far enough away from the border to crop a patch.
            if (isSafePatchCenter(xCoord, yCoord, width, height) &&
                isSafePatchCenter(rightX, yCoord, width, height) &&
                occ.getPixel(xCoord, yCoord) == 0x00 &&
                oof.getPixel(xCoord, yCoord) == 0x00) {
                if (rejectionProp < rej(rng)) {  // reject sample based on disparity
                    kptsL.push_back(ndb::Point(xCoord, yCoord));
                    kptsR.push_back(ndb::Point(rightX, yCoord));
                    // Find a negative patch
                    bool notInFrame = true;
                    int newX, newY;
                    while (notInFrame) {
                        // make a guess [1,radius] (this is why the signum is randomized)
                        auto sig = [&](void) {
                            int k = signum(rng);
                            while (k == 0) k = signum(rng);
                            return k;
                        };
                        newX = rightX + randOffset(rng) * signum(rng);
                        newY = yCoord + randOffset(rng) * signum(rng);

                        // Check if (newX, newY) is valid
                        if (isSafePatchCenter(newX, newY, width, height)) {
                            notInFrame = false;
                        }
                    }
                    // Add negative keypoint
                    kptsN.push_back(ndb::Point(newX, newY));
                }
            }  // boundary distance test
        }      // while
        return 0;
    }  // getGroundTruthMatches
};
}  // namespace datasource
}  // namespace gpc
#endif
