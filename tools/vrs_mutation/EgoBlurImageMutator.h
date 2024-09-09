/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/types.h>
#include <vrs/RecordFormat.h> // @manual

#include <projectaria_tools/tools/samples/vrs_mutation/ImageMutationFilterCopier.h> // @manual

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>       // For file I/O (reading the TXT file)
#include <sstream>       // For parsing strings
#include <vector>        // For storing AprilTag corners
#include <unordered_map> // For mapping timestamps to AprilTag data

#include <c10/cuda/CUDACachingAllocator.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace EgoBlur
{
  struct AprilTagInfo
  {
    double timestamp;
    std::vector<cv::Point2f> corners;
  };

  struct EgoBlurImageMutator : public vrs::utils::UserDefinedImageMutator
  {
    std::unordered_map<double, std::vector<AprilTagInfo>> rgbAprilTagData_;
    std::unordered_map<double, std::vector<AprilTagInfo>> leftAprilTagData_;
    std::unordered_map<double, std::vector<AprilTagInfo>> rightAprilTagData_;

    void loadAprilTagDataFromTXT(const std::string &txtFilePath, std::unordered_map<double, std::vector<AprilTagInfo>> &aprilTagData)
    {
      std::ifstream file(txtFilePath);
      std::string line;
      if (!file.is_open())
      {
        throw std::runtime_error("Could not open TXT file: " + txtFilePath);
      }

      while (std::getline(file, line))
      {
        std::stringstream ss(line);
        AprilTagInfo tagInfo;
        std::vector<cv::Point2f> corners;

        // Assuming TXT format: timestamp x1 y1 x2 y2 x3 y3 x4 y4
        ss >> tagInfo.timestamp;

        for (int i = 0; i < 4; ++i)
        {
          float x, y;
          ss >> x >> y;
          corners.push_back(cv::Point2f(x, y));
        }

        tagInfo.corners = corners;
        aprilTagData[tagInfo.timestamp].push_back(tagInfo);
      }

      file.close();
    }

    std::unordered_map<double, std::vector<AprilTagInfo>> &getAprilTagDataForStream(const vrs::StreamId &streamId)
    {
      if (streamId.getNumericName().find("214") != std::string::npos)
      {
        return rgbAprilTagData_;
      }
      else if (streamId.getNumericName().find("1201-1") != std::string::npos)
      {
        return leftAprilTagData_;
      }
      else if (streamId.getNumericName().find("1201-2") != std::string::npos)
      {
        return rightAprilTagData_;
      }
      else
      {
        throw std::runtime_error("Unknown stream ID: " + streamId.getNumericName());
      }
    }

    // Blurs the image based on the AprilTag corner data
    cv::Mat detectAndBlur(vrs::utils::PixelFrame *frame, const vrs::StreamId &streamId, double timestamp)
    {
      const int width = frame->getWidth();
      const int height = frame->getHeight();
      const int channels = frame->getPixelFormat() == vrs::PixelFormat::RGB8 ? 3 : 1;

      // Convert PixelFrame to cv::Mat
      cv::Mat img(height, width, CV_8UC(channels), static_cast<void *>(frame->getBuffer().data())).clone();

      auto &aprilTagData = getAprilTagDataForStream(streamId);

      // Check if we have AprilTag data for this timestamp
      if (aprilTagData.find(timestamp) == aprilTagData.end())
      {
        return img; // No tags for this timestamp
      }

      // Iterate through the AprilTags for this timestamp
      for (const auto &tagInfo : aprilTagData[timestamp])
      {
        std::vector<cv::Point2f> corners = tagInfo.corners;

        // Create a mask for the AprilTag polygon
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        std::vector<cv::Point> polygon;
        for (const auto &corner : corners)
        {
          polygon.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
        }

        // Fill the polygon in the mask
        const std::vector<std::vector<cv::Point>> polygons = {polygon};
        cv::fillPoly(mask, polygons, cv::Scalar(255));

        // Blur the entire image
        cv::Mat blurredImage;
        cv::blur(img, blurredImage, cv::Size(30, 30)); // Adjust blur kernel size as needed

        // Combine the blurred region with the original image using the mask
        blurredImage.copyTo(img, mask);
      }
      return img;
    }

    // Operator to apply the blur to each frame based on the AprilTag data
    bool operator()(double timestamp, const vrs::StreamId &streamId, vrs::utils::PixelFrame *frame) override
    {
      if (!frame)
      {
        return false;
      }

      cv::Mat blurredImage;
      try
      {
        blurredImage = detectAndBlur(frame, streamId, timestamp);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during blurring: " << e.what() << std::endl;
        return false;
      }

      if (!blurredImage.empty())
      {
        // RGB
        if (streamId.getNumericName().find("214") != std::string::npos)
        {
          std::memcpy(
              frame->wdata(),
              blurredImage.data,
              frame->getWidth() * frame->getStride());
        }
        // Gray
        else if (streamId.getNumericName().find("1201-1") != std::string::npos || streamId.getNumericName().find("1201-2") != std::string::npos)
        {
          std::memcpy(
              frame->wdata(),
              blurredImage.data,
              frame->getWidth() * frame->getHeight());
        }
      }

      return true;
    }

    std::string logStatistics() const
    {
      std::ostringstream summary;
      summary << "AprilTag Data Loaded and Processed" << std::endl;
      return summary.str();
    }
  };

} // namespace EgoBlur
