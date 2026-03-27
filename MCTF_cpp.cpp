#include "EncTemporalFilter_pybind.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int BASELINE_BIT_DEPTH = 10;
constexpr int MOTION_VECTOR_FACTOR = 16;
constexpr int NTAPS = 6;
constexpr int HALF_NTAPS = (NTAPS - 1) / 2;

static const int16_t INTERP[16][NTAPS] = {
    { 0,  0, 64,  0,  0,  0 },
    { 1, -3, 63,  4, -2,  1 },
    { 1, -5, 62,  8, -3,  1 },
    { 1, -6, 60, 13, -4,  0 },
    { 1, -7, 57, 19, -5, -1 },
    { 1, -8, 54, 24, -6, -1 },
    { 1, -9, 50, 29, -6, -1 },
    { 1, -9, 46, 35, -7, -2 },
    { 1, -10, 42, 42, -10, 1 },
    { -2, -7, 35, 46, -9, 1 },
    { -1, -6, 29, 50, -9, 1 },
    { -1, -6, 24, 54, -8, 1 },
    { -1, -5, 19, 57, -7, 1 },
    { 0, -4, 13, 60, -6, 1 },
    { 1, -3,  8, 62, -5, 1 },
    { 1, -2,  4, 63, -3, 1 },
};

inline int clamp_int(int v, int lo, int hi) {
    return std::min(std::max(v, lo), hi);
}

inline void validate_final_block_size(int finalBlockSize) {
    if (!(finalBlockSize == 8 || finalBlockSize == 16)) {
        throw std::runtime_error("final_block_size must be 8 or 16");
    }
}

template <typename T>
TFImage numpy_to_image(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    auto b = arr.template unchecked<2>();
    TFImage img(static_cast<int>(b.shape(1)), static_cast<int>(b.shape(0)));
    for (ssize_t y = 0; y < b.shape(0); ++y) {
        for (ssize_t x = 0; x < b.shape(1); ++x) {
            img.at(static_cast<int>(x), static_cast<int>(y)) = static_cast<int>(b(y, x));
        }
    }
    return img;
}

template <typename T>
py::array_t<T> image_to_numpy(const TFImage& img) {
    py::array_t<T> out({img.h, img.w});
    auto b = out.template mutable_unchecked<2>();
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            b(y, x) = static_cast<T>(img.at(x, y));
        }
    }
    return out;
}

std::pair<TFImage, int> parse_image_and_bitdepth(const py::array& arr, int bitdepth) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("input must be a 2D numpy array");
    }

    TFImage img;
    if (py::isinstance<py::array_t<uint8_t>>(arr)) {
        img = numpy_to_image<uint8_t>(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 8;
    } else if (py::isinstance<py::array_t<uint16_t>>(arr)) {
        img = numpy_to_image<uint16_t>(py::array_t<uint16_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else if (py::isinstance<py::array_t<int16_t>>(arr)) {
        img = numpy_to_image<int16_t>(py::array_t<int16_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else if (py::isinstance<py::array_t<int32_t>>(arr)) {
        img = numpy_to_image<int32_t>(py::array_t<int32_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else {
        throw std::runtime_error("supported dtypes: uint8, uint16, int16, int32");
    }
    return {img, bitdepth};
}

py::array_t<int16_t> motion_to_numpy(const TFArray2D<TFMotionVector>& mvs) {
    py::array_t<int16_t> out({mvs.h(), mvs.w(), 2});
    auto b = out.mutable_unchecked<3>();
    for (int y = 0; y < mvs.h(); ++y) {
        for (int x = 0; x < mvs.w(); ++x) {
            const auto& mv = mvs.get(x, y);
            b(y, x, 0) = static_cast<int16_t>(mv.x);
            b(y, x, 1) = static_cast<int16_t>(mv.y);
        }
    }
    return out;
}

TFArray2D<TFMotionVector> numpy_to_motion(const py::array_t<int16_t, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 3 || arr.shape(2) != 2) {
        throw std::runtime_error("motion must have shape (H/final_block_size, W/final_block_size, 2)");
    }
    auto b = arr.unchecked<3>();
    TFArray2D<TFMotionVector> mvs(static_cast<int>(b.shape(1)), static_cast<int>(b.shape(0)));
    for (ssize_t y = 0; y < b.shape(0); ++y) {
        for (ssize_t x = 0; x < b.shape(1); ++x) {
            mvs.get(static_cast<int>(x), static_cast<int>(y)) = {b(y, x, 0), b(y, x, 1), 0, 0.0};
        }
    }
    return mvs;
}

py::array estimate_motion_py(const py::array& target, const py::array& reference,
                             int bitdepth = -1, int final_block_size = 8) {
    auto [timg, bd0] = parse_image_and_bitdepth(target, bitdepth);
    auto [rimg, bd1] = parse_image_and_bitdepth(reference, bitdepth);
    const int bd = std::max(bd0, bd1);
    const auto mvs = tf_motion_estimation(timg, rimg, bd, final_block_size);
    return motion_to_numpy(mvs);
}

py::array apply_motion_py(const py::array_t<int16_t, py::array::c_style | py::array::forcecast>& motion,
                          const py::array& reference,
                          int bitdepth = -1,
                          int final_block_size = 8) {
    auto [rimg, bd] = parse_image_and_bitdepth(reference, bitdepth);
    const auto mvs = numpy_to_motion(motion);
    const TFImage out = tf_apply_motion(mvs, rimg, bd, final_block_size);

    if (py::isinstance<py::array_t<uint8_t>>(reference)) {
        return image_to_numpy<uint8_t>(out);
    }
    if (py::isinstance<py::array_t<uint16_t>>(reference)) {
        return image_to_numpy<uint16_t>(out);
    }
    if (py::isinstance<py::array_t<int16_t>>(reference)) {
        return image_to_numpy<int16_t>(out);
    }
    return image_to_numpy<int32_t>(out);
}

} // namespace

int TFImage::at_clamped(int x, int y) const {
    x = clamp_int(x, 0, w - 1);
    y = clamp_int(y, 0, h - 1);
    return pix[static_cast<size_t>(y) * w + x];
}

TFImage tf_subsample_luma(const TFImage& input, int factor) {
    if (input.w % factor != 0 || input.h % factor != 0) {
        throw std::runtime_error("Input size must be divisible by subsampling factor");
    }
    TFImage out(input.w / factor, input.h / factor);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            const int x0 = x * factor;
            const int y0 = y * factor;
            out.at(x, y) = (input.at(x0, y0) + input.at(x0 + 1, y0) +
                            input.at(x0, y0 + 1) + input.at(x0 + 1, y0 + 1) + 2) >> 2;
        }
    }
    return out;
}

int tf_filtered_sample_6tap(const TFImage& buf, int base_x, int base_y, int dx, int dy, int max_value) {
    const int dxFrac = dx & 15;
    const int dyFrac = dy & 15;
    const int dxInt = dx >> 4;
    const int dyInt = dy >> 4;
    const auto* xFilter = INTERP[dxFrac];
    const auto* yFilter = INTERP[dyFrac];

    const int ox = base_x + dxInt;
    const int oy = base_y + dyInt;

    if (dxFrac == 0 && dyFrac == 0) {
        return buf.at_clamped(ox, oy);
    }
    if (dxFrac == 0) {
        int sum = 1 << 5;
        for (int k = 0; k < NTAPS; ++k) {
            sum += buf.at_clamped(ox, oy + k - HALF_NTAPS) * yFilter[k];
        }
        return clamp_int(sum >> 6, 0, max_value);
    }
    if (dyFrac == 0) {
        int sum = 1 << 5;
        for (int k = 0; k < NTAPS; ++k) {
            sum += buf.at_clamped(ox + k - HALF_NTAPS, oy) * xFilter[k];
        }
        return clamp_int(sum >> 6, 0, max_value);
    }

    int temp[NTAPS];
    for (int ky = 0; ky < NTAPS; ++ky) {
        int sum = 0;
        const int yy = oy + ky - HALF_NTAPS;
        for (int kx = 0; kx < NTAPS; ++kx) {
            sum += buf.at_clamped(ox + kx - HALF_NTAPS, yy) * xFilter[kx];
        }
        temp[ky] = sum;
    }

    int sum = 1 << 11;
    for (int k = 0; k < NTAPS; ++k) {
        sum += temp[k] * yFilter[k];
    }
    return clamp_int(sum >> 12, 0, max_value);
}

int64_t tf_motion_error_luma(const TFImage& orig, const TFImage& buffer, int x, int y, int dx, int dy,
                             int bs, int64_t besterror, int max_value) {
    int64_t error = 0;
    const int bw = std::min(bs, orig.w - x);
    const int bh = std::min(bs, orig.h - y);

    for (int y1 = 0; y1 < bh; ++y1) {
        for (int x1 = 0; x1 < bw; ++x1) {
            const int pred = tf_filtered_sample_6tap(buffer, x + x1, y + y1, dx, dy, max_value);
            const int diff = orig.at(x + x1, y + y1) - pred;
            error += static_cast<int64_t>(diff) * diff;
        }
        if (error > besterror) {
            return error;
        }
    }
    return error;
}

void tf_motion_estimation_luma(TFArray2D<TFMotionVector>& mvs, const TFImage& orig, const TFImage& buffer,
                               int blockSize, int bitdepth,
                               const TFArray2D<TFMotionVector>* previous,
                               int factor, bool doubleRes) {
    int range = doubleRes ? 0 : 5;
    const int stepSize = blockSize;
    const int origWidth = orig.w;
    const int origHeight = orig.h;
    const int maxValue = (1 << bitdepth) - 1;

    const double offset = 5.0 / (1 << (2 * BASELINE_BIT_DEPTH - 16)) * (1 << (2 * bitdepth - 16));
    const double scale  = 50.0 / (1 << (2 * BASELINE_BIT_DEPTH - 16)) * (1 << (2 * bitdepth - 16));

    for (int blockY = 0; blockY < origHeight; blockY += stepSize) {
        for (int blockX = 0; blockX < origWidth; blockX += stepSize) {
            TFMotionVector best;

            if (previous == nullptr) {
                range = 8;
            } else {
                for (int py = -1; py <= 1; ++py) {
                    const int testy = blockY / (2 * blockSize) + py;
                    for (int px = -1; px <= 1; ++px) {
                        const int testx = blockX / (2 * blockSize) + px;
                        if (testx >= 0 && testx < previous->w() && testy >= 0 && testy < previous->h()) {
                            TFMotionVector old = previous->get(testx, testy);
                            const int64_t error = tf_motion_error_luma(
                                orig, buffer, blockX, blockY,
                                old.x * factor, old.y * factor,
                                blockSize, best.error, maxValue);
                            if (error < best.error) {
                                best = {old.x * factor, old.y * factor, error, 0.0};
                            }
                        }
                    }
                }
                const int64_t zero_error = tf_motion_error_luma(orig, buffer, blockX, blockY, 0, 0,
                                                                blockSize, best.error, maxValue);
                if (zero_error < best.error) {
                    best = {0, 0, zero_error, 0.0};
                }
            }

            TFMotionVector prevBest = best;
            for (int y2 = prevBest.y / MOTION_VECTOR_FACTOR - range; y2 <= prevBest.y / MOTION_VECTOR_FACTOR + range; ++y2) {
                for (int x2 = prevBest.x / MOTION_VECTOR_FACTOR - range; x2 <= prevBest.x / MOTION_VECTOR_FACTOR + range; ++x2) {
                    const int64_t error = tf_motion_error_luma(
                        orig, buffer, blockX, blockY,
                        x2 * MOTION_VECTOR_FACTOR, y2 * MOTION_VECTOR_FACTOR,
                        blockSize, best.error, maxValue);
                    if (error < best.error) {
                        best = {x2 * MOTION_VECTOR_FACTOR, y2 * MOTION_VECTOR_FACTOR, error, 0.0};
                    }
                }
            }

            if (doubleRes) {
                prevBest = best;
                int doubleRange = 12;
                for (int y2 = prevBest.y - doubleRange; y2 <= prevBest.y + doubleRange; y2 += 4) {
                    for (int x2 = prevBest.x - doubleRange; x2 <= prevBest.x + doubleRange; x2 += 4) {
                        const int64_t error = tf_motion_error_luma(
                            orig, buffer, blockX, blockY, x2, y2,
                            blockSize, best.error, maxValue);
                        if (error < best.error) {
                            best = {x2, y2, error, 0.0};
                        }
                    }
                }

                prevBest = best;
                doubleRange = 3;
                for (int y2 = prevBest.y - doubleRange; y2 <= prevBest.y + doubleRange; ++y2) {
                    for (int x2 = prevBest.x - doubleRange; x2 <= prevBest.x + doubleRange; ++x2) {
                        const int64_t error = tf_motion_error_luma(
                            orig, buffer, blockX, blockY, x2, y2,
                            blockSize, best.error, maxValue);
                        if (error < best.error) {
                            best = {x2, y2, error, 0.0};
                        }
                    }
                }
            }

            if (blockY > 0) {
                const TFMotionVector above = mvs.get(blockX / stepSize, (blockY - stepSize) / stepSize);
                const int64_t error = tf_motion_error_luma(
                    orig, buffer, blockX, blockY, above.x, above.y,
                    blockSize, best.error, maxValue);
                if (error < best.error) {
                    best = {above.x, above.y, error, 0.0};
                }
            }

            if (blockX > 0) {
                const TFMotionVector left = mvs.get((blockX - stepSize) / stepSize, blockY / stepSize);
                const int64_t error = tf_motion_error_luma(
                    orig, buffer, blockX, blockY, left.x, left.y,
                    blockSize, best.error, maxValue);
                if (error < best.error) {
                    best = {left.x, left.y, error, 0.0};
                }
            }

            const int bw = std::min(blockSize, orig.w - blockX);
            const int bh = std::min(blockSize, orig.h - blockY);
            double avg = 0.0;
            for (int y1 = 0; y1 < bh; ++y1) {
                for (int x1 = 0; x1 < bw; ++x1) {
                    avg += orig.at(blockX + x1, blockY + y1);
                }
            }
            avg /= static_cast<double>(bw * bh);

            double variance = 0.0;
            for (int y1 = 0; y1 < bh; ++y1) {
                for (int x1 = 0; x1 < bw; ++x1) {
                    const double d = orig.at(blockX + x1, blockY + y1) - avg;
                    variance += d * d;
                }
            }

            best.error = static_cast<int64_t>(
                20.0 * ((best.error + offset) / (variance + offset)) +
                (best.error / static_cast<double>(bw * bh)) / scale
            );
            best.overlap = (static_cast<double>(bw) * bh) / (blockSize * blockSize);
            mvs.get(blockX / stepSize, blockY / stepSize) = best;
        }
    }
}

TFArray2D<TFMotionVector> tf_motion_estimation(const TFImage& orgPic, const TFImage& buffer,
                                               int bitdepth, int finalBlockSize) {
    validate_final_block_size(finalBlockSize);

    const int coarseBlockSize = finalBlockSize * 2;
    const int requiredMultiple = finalBlockSize * 4;

    if (orgPic.w != buffer.w || orgPic.h != buffer.h) {
        throw std::runtime_error("target and reference must have the same shape");
    }
    if (orgPic.w % requiredMultiple != 0 || orgPic.h % requiredMultiple != 0) {
        throw std::runtime_error("input height and width must be multiples of 4 * final_block_size");
    }

    const TFImage origSub2 = tf_subsample_luma(orgPic, 2);
    const TFImage origSub4 = tf_subsample_luma(origSub2, 2);
    const TFImage bufSub2  = tf_subsample_luma(buffer, 2);
    const TFImage bufSub4  = tf_subsample_luma(bufSub2, 2);

    TFArray2D<TFMotionVector> mv0((origSub4.w + coarseBlockSize - 1) / coarseBlockSize,
                                  (origSub4.h + coarseBlockSize - 1) / coarseBlockSize);
    TFArray2D<TFMotionVector> mv1((origSub2.w + coarseBlockSize - 1) / coarseBlockSize,
                                  (origSub2.h + coarseBlockSize - 1) / coarseBlockSize);
    TFArray2D<TFMotionVector> mv2((orgPic.w + coarseBlockSize - 1) / coarseBlockSize,
                                  (orgPic.h + coarseBlockSize - 1) / coarseBlockSize);
    TFArray2D<TFMotionVector> mv((orgPic.w + finalBlockSize - 1) / finalBlockSize,
                                 (orgPic.h + finalBlockSize - 1) / finalBlockSize);

    tf_motion_estimation_luma(mv0, origSub4, bufSub4, coarseBlockSize, bitdepth);
    tf_motion_estimation_luma(mv1, origSub2, bufSub2, coarseBlockSize, bitdepth, &mv0, 2);
    tf_motion_estimation_luma(mv2, orgPic,   buffer,  coarseBlockSize, bitdepth, &mv1, 2);
    tf_motion_estimation_luma(mv,  orgPic,   buffer,  finalBlockSize,  bitdepth, &mv2, 1, true);

    return mv;
}

TFImage tf_apply_motion(const TFArray2D<TFMotionVector>& mvs, const TFImage& ref,
                        int bitdepth, int finalBlockSize) {
    validate_final_block_size(finalBlockSize);

    if (ref.w % finalBlockSize != 0 || ref.h % finalBlockSize != 0) {
        throw std::runtime_error("reference size must be multiples of final_block_size");
    }
    if (mvs.w() != (ref.w + finalBlockSize - 1) / finalBlockSize ||
        mvs.h() != (ref.h + finalBlockSize - 1) / finalBlockSize) {
        throw std::runtime_error("motion shape must be (ceil(H/final_block_size), ceil(W/final_block_size), 2)");
    }

    TFImage out(ref.w, ref.h);
    const int maxValue = (1 << bitdepth) - 1;

    for (int by = 0; by < mvs.h(); ++by) {
        for (int bx = 0; bx < mvs.w(); ++bx) {
            const TFMotionVector mv = mvs.get(bx, by);
            const int x0 = bx * finalBlockSize;
            const int y0 = by * finalBlockSize;
            const int bw = std::min(finalBlockSize, ref.w - x0);
            const int bh = std::min(finalBlockSize, ref.h - y0);

            for (int iy = 0; iy < bh; ++iy) {
                for (int ix = 0; ix < bw; ++ix) {
                    out.at(x0 + ix, y0 + iy) =
                        tf_filtered_sample_6tap(ref, x0 + ix, y0 + iy, mv.x, mv.y, maxValue);
                }
            }
        }
    }
    return out;
}

PYBIND11_MODULE(tf_memc, m) {
    m.doc() = "Standalone pybind11 wrapper for EncTemporalFilter MEMC path (luma only, configurable final_block_size)";

    m.def("estimate_motion", &estimate_motion_py,
          py::arg("target"), py::arg("reference"),
          py::arg("bitdepth") = -1,
          py::arg("final_block_size") = 8,
          R"pbdoc(
Estimate block motion from reference -> target.

Input:
    2D numpy arrays with same shape.

Args:
    target: 2D numpy array
    reference: 2D numpy array
    bitdepth: optional explicit bitdepth
    final_block_size: 8 or 16

Returns:
    int16 numpy array of shape (ceil(H/final_block_size), ceil(W/final_block_size), 2)
    Motion vectors are in 1/16-pel units
)pbdoc");

    m.def("apply_motion", &apply_motion_py,
          py::arg("motion"), py::arg("reference"),
          py::arg("bitdepth") = -1,
          py::arg("final_block_size") = 8,
          R"pbdoc(
Apply motion vectors to a reference image and return compensated prediction.

Input:
    motion: int16 numpy array, shape (ceil(H/final_block_size), ceil(W/final_block_size), 2)
    reference: 2D numpy array, shape (H, W)
    bitdepth: optional explicit bitdepth
    final_block_size: 8 or 16

Returns:
    Compensated image with same shape/dtype as reference
)pbdoc");
}




















#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct TFMotionVector {
    int x = 0;
    int y = 0;
    int64_t error = std::numeric_limits<int64_t>::max();
    double overlap = 0.0;
};

template <typename T>
class TFArray2D {
public:
    TFArray2D() = default;
    TFArray2D(int w, int h, const T& value = T()) { allocate(w, h, value); }

    void allocate(int w, int h, const T& value = T()) {
        width_ = w;
        height_ = h;
        data_.assign(static_cast<size_t>(w) * static_cast<size_t>(h), value);
    }

    int w() const { return width_; }
    int h() const { return height_; }

    T& get(int x, int y) { return data_[static_cast<size_t>(y) * width_ + x]; }
    const T& get(int x, int y) const { return data_[static_cast<size_t>(y) * width_ + x]; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<T> data_;
};

struct TFImage {
    int w = 0;
    int h = 0;
    std::vector<int> pix;

    TFImage() = default;
    TFImage(int width, int height) : w(width), h(height), pix(static_cast<size_t>(width) * height, 0) {}

    int& at(int x, int y) { return pix[static_cast<size_t>(y) * w + x]; }
    int at(int x, int y) const { return pix[static_cast<size_t>(y) * w + x]; }
    int at_clamped(int x, int y) const;
};

TFImage tf_subsample_luma(const TFImage& input, int factor = 2);
int tf_filtered_sample_6tap(const TFImage& buf, int base_x, int base_y, int dx, int dy, int max_value);
int64_t tf_motion_error_luma(const TFImage& orig, const TFImage& buffer, int x, int y, int dx, int dy,
                             int bs, int64_t besterror, int max_value);
void tf_motion_estimation_luma(TFArray2D<TFMotionVector>& mvs, const TFImage& orig, const TFImage& buffer,
                               int blockSize, int bitdepth,
                               const TFArray2D<TFMotionVector>* previous = nullptr,
                               int factor = 1, bool doubleRes = false);
TFArray2D<TFMotionVector> tf_motion_estimation(const TFImage& orgPic, const TFImage& buffer,
                                               int bitdepth, int finalBlockSize);
TFImage tf_apply_motion(const TFArray2D<TFMotionVector>& mvs, const TFImage& ref,
                        int bitdepth, int finalBlockSize);


















#!/usr/bin/env bash
set -euo pipefail

MODULE_NAME=tf_memc
SRC=EncTemporalFilter_pybind.cpp

cxxflags="$(python3 -m pybind11 --includes) -O3 -Wall -shared -std=c++17 -fPIC"
ext_suffix="$(python3-config --extension-suffix)"

c++ ${cxxflags} ${SRC} -o ${MODULE_NAME}${ext_suffix}

echo "Built ${MODULE_NAME}${ext_suffix}"




