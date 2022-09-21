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
// Code Author: Niklaus Bamert (bamertn@ethz.ch)
#ifndef __NDB_BUFFER
#define __NDB_BUFFER
#include <vector>
#include <png.h>
#include <Eigen/Dense>
#include <type_traits>

using namespace std;

namespace ndb {
struct RGBColor {
    uint8_t b, g, r;
    RGBColor(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
    RGBColor(png_byte *ptr) {
        this->r = ptr[0];
        this->g = ptr[1];
        this->b = ptr[2];
    }
    RGBColor(){};
};

struct Point {
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
    Point(){};
};

struct Descriptor {
    Point point;
    uint64_t state;
    bool srcDescr = false;  // indicates if this is a descriptor from the source or from
                            // the target set
    Descriptor(ndb::Point point, uint64_t state) : point(point), state(state) {}
    Descriptor(){};
    // ops for sorting and comparing descriptors for matching
    bool operator==(const Descriptor &d) const {
        if (state == d.state) return true;
        return false;
    }
    // Checks if two descriptors are from different images
    bool diffImgs(const Descriptor &d) {
        if (srcDescr != d.srcDescr) return true;
        return false;
    }
    bool operator!=(const Descriptor &d) const {
        if (state != d.state) return true;
        return false;
    }
    bool operator<(const Descriptor &d) const { return state < d.state; }
    bool operator<=(const Descriptor &d) const { return state <= d.state; }
    int operator%(const int &d) const { return state % d; }
};

// Keeps support points with associated disparity
// Support points are only used in the left image
struct Support {
    int x, y;
    float d;
    Support(int x, int y, float d) : x(x), y(y), d(d) {}
    Support(int x, int y) : x(x), y(y), d(0.) {}
    Support(){};
};
// Keeps correspondences in case of non-epipolar matching scenario
struct Correspondence {
    Point srcPt, tarPt;
    Correspondence(Point srcPt, Point tarPt) : srcPt(srcPt), tarPt(tarPt) {}
};
// The Cg matrix elements used in Disparity Refinement
struct ConfidentSupport {
    int x, y, cost;
    float d;
    ConfidentSupport(){};
    ConfidentSupport(int x, int y, float d, int cost) : x(x), y(y), d(d), cost(cost) {}
};
struct InvalidMatch {
    int x, y, cost;
    InvalidMatch() { cost = 0; };
    InvalidMatch(int x, int y, int cost) : x(x), y(y), cost(cost) {}
};

struct Triangle {
    int v1, v2, v3;
    Triangle(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}
};
struct Edge {
    Support a, b;
    Edge(Support &a, Support &b) {
        if (a.y < b.y) {
            this->a = a;
            this->b = b;
        } else {
            this->a = b;
            this->b = a;
        }
    }
};
struct Span {
    int x1, x2;
    Span(int x1, int x2) : x1(x1), x2(x2) {}
};
struct Dimension {
    int w, h;
    Dimension(int w, int h) : w(w), h(h) {}
};

#define ALIGN16(X) (X % 16) == 0 ? X : ((X / 16) + 1) * 16
template <class T>
class __attribute__((aligned(32), packed)) Buffer
    : public Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
    // The actual height and width of the image.
public:
    int width;
    int height;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Base;
    Buffer(const int r, const int c) : Base(r, ALIGN16(c)) {
        this->width = c;
        this->height = r;
    }
    Buffer(const int r, const int c, T color) : Base(r, ALIGN16(c)) {
        this->width = c;
        this->height = r;
        T *ptr = reinterpret_cast<T *>(Base::data());
        for (int i = 0; i < Base::cols() * Base::rows(); i++) {
            *ptr = color;
            ptr++;
        }
    }
    // Create new buffer from two side by side
    Buffer(Buffer &i1, Buffer &i2) : Base(i1.rows(), i1.cols() + i2.cols()) {
        for (int x = 0; x < i1.cols(); x++) {
            for (int y = 0; y < i1.rows(); y++) {
                setPixel(x, y, i1.getPixel(x, y));
                setPixel(x + i1.cols(), y, i2.getPixel(x, y));
            }
        }
        this->height = Base::rows();
        this->width = Base::cols();
    }
    Buffer(const Eigen::Vector2i &size = Eigen::Vector2i(0, 0))
        : Base(size.y(), ALIGN16(size.x())) {
        this->width = size.x();
        this->height = size.y();
    }
    Buffer(const Eigen::Vector2i &size, T color) : Base(size.y(), ALIGN16(size.x())) {
        this->width = size.x();
        this->height = size.y();
        T *ptr = reinterpret_cast<T *>(Base::data());
        for (int i = 0; i < Base::cols() * Base::rows(); i++) {
            *ptr = color;
            ptr++;
        }
    }
    // original sources provided by Guillaume Cottenceau under the X11 license
    // from http://zarb.org/~gc/html/libpng.html

    int readPNG(std::string filename) {
        unsigned char header[8];  // 8 is the maximum size that can be checked
        png_byte colorType;
        png_byte bitDepth;

        png_structp pngPtr;
        png_infop infoPtr;
        png_bytep *rowPointers;

        // open file and test for it being a png
        FILE *fp = fopen(filename.c_str(), "rb");
        if (!fp) {
            cout << "ERR: File" << filename << " could not be opened for reading" << endl;
            return 1;
        }
        size_t res = fread(header, 1, 8, fp);
        if (png_sig_cmp(header, 0, 8)) {
            cout << "ERR: File" << filename << " is not recognized as a PNG file" << endl;
            return 1;
        }
        // initialize stuff
        pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!pngPtr) {
            cout << "ERR: png_create_read_struct failed" << endl;
            return 1;
        }

        infoPtr = png_create_info_struct(pngPtr);
        if (!infoPtr) {
            cout << "ERR: png_create_info_struct failed" << endl;
            return 1;
        }

        if (setjmp(png_jmpbuf(pngPtr))) {
            cout << "ERR: Error during init_io" << endl;
            return 1;
        }

        png_init_io(pngPtr, fp);
        png_set_sig_bytes(pngPtr, 8);

        png_read_info(pngPtr, infoPtr);

        this->width = png_get_image_width(pngPtr, infoPtr);
        this->height = png_get_image_height(pngPtr, infoPtr);
        colorType = png_get_color_type(pngPtr, infoPtr);
        bitDepth = png_get_bit_depth(pngPtr, infoPtr);

        // We will do a conservative resize after reading in the data,
        // such that we don't have to translate addresses ourselves
        Base::resize(this->height, this->width);

        int numberOfPasses = png_set_interlace_handling(pngPtr);
        png_read_update_info(pngPtr, infoPtr);

        // read file
        if (setjmp(png_jmpbuf(pngPtr))) {
            cout << "ERR: Error during read_image" << endl;
            return 1;
        }

        rowPointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
        for (int y = 0; y < height; y++)
            rowPointers[y] = (png_byte *)malloc(png_get_rowbytes(pngPtr, infoPtr));

        png_read_image(pngPtr, rowPointers);

        fclose(fp);
        // Read image into buffer (row-major)
        int nChannels;

        switch (png_get_color_type(pngPtr, infoPtr)) {
            case PNG_COLOR_TYPE_GRAY:
                nChannels = 1;
                break;
            case PNG_COLOR_TYPE_RGB:
                nChannels = 3;
                break;
            case PNG_COLOR_TYPE_RGBA:
                nChannels = 4;
                break;
            default:
                nChannels = 0;
                break;
        }
        T *ptr = reinterpret_cast<T *>(Base::data());

        if (bitDepth == 16) {
            for (int y = 0; y < this->height; y++) {
                png_byte *row = rowPointers[y];
                for (int x = 0; x < this->width; x++) {
                    int val = ((int)row[x * 2] << 8) + row[x * 2 + 1];
                    *ptr = val;
                    ptr++;
                }
            }
        } else {
            for (int y = 0; y < this->height; y++) {
                png_byte *row = rowPointers[y];
                for (int x = 0; x < this->width; x++) {
                    int offset = y * this->width + x;
                    if (nChannels == 1) {
                        *ptr = row[x];
                        ptr++;
                    } else if (nChannels == 3) {  // convert to grayscale if color
                        *ptr = (row[3 * x] + row[3 * x + 1] + row[3 * x + 2]) / 3;
                        ptr++;
                    }
                }
            }
        }
        // Conservative resize of underlying matrix to align with 16 byte boundary
        Base::conservativeResize(this->height, ALIGN16(this->width));
        if (nChannels == 0 || nChannels == 4) {
            cout << "ERR: found something other than gray or 3 channel color image("
                 << int(png_get_color_type(pngPtr, infoPtr)) << ") aborting!" << endl;

            return 1;
        }
        for (int y = 0; y < this->height; y++) free(rowPointers[y]);
        free(rowPointers);
        return 0;
    }
    void writePNG(std::string filename) {
        png_byte colorType = PNG_COLOR_TYPE_GRAY;
        int nChannels = 1;
        png_byte bitDepth = 8;

        png_structp pngPtr;
        png_infop infoPtr;
        png_bytep *rowPointers;

        // Copy Image matrix such that we can do a conservative resize to its correct size
        Base img = *this;
        img.conservativeResize(this->height, this->width);

        FILE *fp = fopen(filename.c_str(), "wb");
        if (!fp)
            cout << "ERR: File" << filename << " could not be opened for writing" << endl;

        // init
        pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!pngPtr) cout << "ERR: png_create_write_struct failed" << endl;

        infoPtr = png_create_info_struct(pngPtr);
        if (!infoPtr) cout << "ERR: png_create_info_struct failed" << endl;

        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during init_io" << endl;

        png_init_io(pngPtr, fp);

        // write header
        if (setjmp(png_jmpbuf(pngPtr)))
            cout << "ERR: Error during writing header" << endl;

        png_set_IHDR(
            pngPtr, infoPtr, img.cols(), img.rows(), bitDepth, colorType,
            PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(pngPtr, infoPtr);

        // write bytes
        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during writing bytes" << endl;

        // Allocate row pointers
        rowPointers = (png_bytep *)malloc(sizeof(png_bytep) * img.rows());
        for (int y = 0; y < img.rows(); y++)
            rowPointers[y] = (png_byte *)malloc(png_get_rowbytes(pngPtr, infoPtr));

        T *ptr = reinterpret_cast<T *>(img.data());
        // Copy our data from std::vector into allocated memory region
        for (int y = 0; y < img.rows(); y++) {
            png_byte *row = rowPointers[y];
            for (int x = 0; x < img.cols(); x++) {
                int offset;
                offset = y * img.cols() + x;
                row[x * nChannels] = ptr[offset];
            }
        }

        png_write_image(pngPtr, rowPointers);

        // end write
        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during end of write" << endl;

        png_write_end(pngPtr, NULL);

        for (int y = 0; y < img.rows(); y++) free(rowPointers[y]);
        free(rowPointers);

        fclose(fp);
    }
    void writePNGRGB(std::string filename) {
        png_byte colorType = PNG_COLOR_TYPE_RGB;
        int nChannels = 3;
        png_byte bitDepth = 8;

        png_structp pngPtr;
        png_infop infoPtr;
        png_bytep *rowPointers;
        // Make copy of self that we can resize to actual size
        Base img = *this;
        img.conservativeResize(this->height, this->width);
        // create file
        FILE *fp = fopen(filename.c_str(), "wb");
        if (!fp)
            cout << "ERR: File" << filename << " could not be opened for writing" << endl;

        // initialize stuff
        pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!pngPtr) cout << "ERR: png_create_write_struct failed" << endl;

        infoPtr = png_create_info_struct(pngPtr);
        if (!infoPtr) cout << "ERR: png_create_info_struct failed" << endl;

        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during init_io" << endl;

        png_init_io(pngPtr, fp);

        // write header
        if (setjmp(png_jmpbuf(pngPtr)))
            cout << "ERR: Error during writing header" << endl;

        png_set_IHDR(
            pngPtr, infoPtr, img.cols(), img.rows(), bitDepth, colorType,
            PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(pngPtr, infoPtr);

        // write bytes
        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during writing bytes" << endl;

        // Allocate row pointers
        rowPointers = (png_bytep *)malloc(sizeof(png_bytep) * img.rows());
        for (int y = 0; y < img.rows(); y++)
            rowPointers[y] = (png_byte *)malloc(png_get_rowbytes(pngPtr, infoPtr));
        T *ptr = reinterpret_cast<T *>(img.data());

        // Copy our data from std::vector into allocated memory region
        for (int y = 0; y < img.rows(); y++) {
            png_byte *row = rowPointers[y];
            for (int x = 0; x < img.cols(); x++) {
                int offset;
                offset = y * img.cols() + x;
                row[x * nChannels] = ptr[offset].r;
                row[x * nChannels + 1] = ptr[offset].g;
                row[x * nChannels + 2] = ptr[offset].b;
            }
        }

        png_write_image(pngPtr, rowPointers);

        // end write
        if (setjmp(png_jmpbuf(pngPtr))) cout << "ERR: Error during end of write" << endl;

        png_write_end(pngPtr, NULL);

        // cleanup heap allocation
        for (int y = 0; y < img.rows(); y++) free(rowPointers[y]);
        free(rowPointers);

        fclose(fp);
    }
    /**
     * @brief      "Unsafe" but fast pixel set method (no dimension check. If you write
     *             outside the bounds, it'll segfault!)
     *
     * @param[in]  x     { parameter_description }
     * @param[in]  y     { parameter_description }
     */
    void setPixel(int x, int y, T color) {
        T *ptr = reinterpret_cast<T *>(Base::data());
        *(ptr + Base::cols() * y + x) = color;
    }

    /**
     * @brief      Set all values in matrix to same value
     *
     * @param[in]  color  The color
     */
    void set(T color) {
        T *ptr = reinterpret_cast<T *>(Base::data());
        int size = Base::cols() * Base::rows();
        for (int i = 0; i < size; i++) {
            *ptr = color;
            ptr++;
        }
    }
    /**
     * @brief      Gets the pixel.
     *
     * @param[in]  x     { parameter_description }
     * @param[in]  y     { parameter_description }
     *
     * @return     The pixel.
     */
    T getPixel(int x, int y) {
        T *ptr = reinterpret_cast<T *>(Base::data());
        return *(ptr + Base::cols() * y + x);
    }

    /**
     * @brief      Gets the pixel. Const method overload for calls from GPC.
     *
     * @param[in]  x     { parameter_description }
     * @param[in]  y     { parameter_description }
     *
     * @return     The pixel.
     */
    T getPixel(int x, int y) const {
        T *ptr = const_cast<T *>(Base::data());
        return *(ptr + Base::cols() * y + x);
    }
    /**
     * @brief      Gets a patch from the buffer
     *
     * @param[in]  x     { parameter_description }
     * @param[in]  y     { parameter_description }
     * @param[in]  size  The size
     *
     * @return     The patch.
     */
    void getPatch(ndb::Buffer<uint8_t> &patch, int x, int y, int size) {
        patch.resize(size, size);

        // extract patch
        for (int ix = 0; ix < size; ix++) {
            for (int iy = 0; iy < size; iy++) {
                patch(ix, iy) = getPixel(x + ix - (size / 2), y + iy - (size / 2));
            }
        }
    }
    Dimension getDimension() { return Dimension(Base::cols(), Base::rows()); }
    /**
     * @brief      Draws a line.
     *
     * @param      a      { parameter_description }
     * @param      b      { parameter_description }
     * @param[in]  color  The color
     */
    void drawLine(Support &a, Support &b, T color) {
        float xdiff = (b.x - a.x);
        float ydiff = (b.y - a.y);

        if (xdiff == 0.0f && ydiff == 0.0f) {
            setPixel(a.x, a.y, color);
            return;
        }

        if (fabs(xdiff) > fabs(ydiff)) {
            float xmin, xmax;

            // set xmin to the lower x value given
            // and xmax to the higher value
            if (a.x < b.x) {
                xmin = a.x;
                xmax = b.x;
            } else {
                xmin = b.x;
                xmax = a.x;
            }

            // draw line in terms of y slope
            float slope = ydiff / xdiff;
            for (float x = xmin; x <= xmax; x += 1.0f) {
                float y = a.y + ((x - a.x) * slope);
                setPixel(x, y, color);
            }
        } else {
            float ymin, ymax;

            // set ymin to the lower y value given
            // and ymax to the higher value
            if (a.y < b.y) {
                ymin = a.y;
                ymax = b.y;
            } else {
                ymin = b.y;
                ymax = a.y;
            }

            // draw line in terms of x slope
            float slope = xdiff / ydiff;
            for (float y = ymin; y <= ymax; y += 1.0f) {
                float x = a.x + ((y - a.y) * slope);
                setPixel(x, y, color);
            }
        }
    }
    void drawLine(Point &a, Point &b, T color) {
        Support aa(a.x, a.y, 0), bb(b.x, b.y, 0);
        drawLine(aa, bb, color);
    }
    void drawLine(Point a, Point b, T color) {
        Support aa(a.x, a.y, 0), bb(b.x, b.y, 0);
        drawLine(aa, bb, color);
    }
    /**
     * @brief      Draws a span.
     *
     * @param[in]  span   The span
     * @param[in]  y
     * @param      color  The color
     */
    void drawSpan(const Span &span, int y, T &color) {
        int xdiff = span.x2 - span.x1;
        if (xdiff == 0) {
            return;
        }

        for (int x = span.x1; x < span.x2; x++) setPixel(x, y, color);
    }
    void clearBoundary() {
        T *ptr = reinterpret_cast<T *>(Base::data());
        // Dimension of the (visible) image
        int h = this->height;
        int w = this->width;
        // Width of 16-aligned container
        int waligned = this->cols();
        // first 2 columns contains invalid data -> set them zero.
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < h; y++) {
                ptr[y * waligned + x] = 0x00;
            }
        }
        // first row
        for (int x = 0; x < w; x++) ptr[x] = 0x00;
        // last two rowsrow
        for (int x = 0; x < w; x++)
            for (int y = h - 2; y < h; y++) ptr[y * waligned + x] = 0x00;
        // last column
        for (int y = 0; y < h; y++) ptr[y * waligned + (waligned - 1)] = 0x00;
    }
    /**
     * @brief      Draws spans between edges.
     *             modified from
     *             https://github.com/joshb/triangleraster
     * @param[in]  e1    The e 1 (long edge)
     * @param[in]  e2    The e 2 (short edge)
     */
    void drawSpansBetweenEdges(const Edge &e1, const Edge &e2, T &color) {
        // calculate difference between the y coordinates
        // of the first edge and return if 0
        float e1ydiff = (float)(e1.b.y - e1.a.y);
        if (e1ydiff == 0.0f) {
            return;
        }

        // calculate difference between the y coordinates
        // of the second edge and return if 0
        float e2ydiff = (float)(e2.b.y - e2.a.y);
        if (e2ydiff == 0.0f) {
            return;
        }

        // calculate differences between the x coordinates
        float e1xdiff = (float)(e1.b.x - e1.a.x);
        float e2xdiff = (float)(e2.b.x - e2.a.x);

        // calculate factors to use for interpolation
        // with the edges and the step values to increase
        // them by after drawing each span
        float factor1 = (float)(e2.a.y - e1.a.y) / e1ydiff;
        float factorStep1 = 1.0f / e1ydiff;
        float factor2 = 0.0f;
        float factorStep2 = 1.0f / e2ydiff;

        // loop through the lines between the edges and draw spans
        for (int y = e2.a.y; y < e2.b.y; y++) {
            // create and draw span
            Span span(
                e1.a.x + (int)(e1xdiff * factor1), e2.a.x + (int)(e2xdiff * factor2));
            if (span.x1 > span.x2) std::swap(span.x1, span.x2);
            drawSpan(span, y, color);

            // increase factors
            factor1 += factorStep1;
            factor2 += factorStep2;
        }
    }
    /**
     * @brief      Draw a triangle from three vertices and fill it. modified from
     *             https://github.com/joshb/triangleraster
     *             released under BSD licence
     *
     * @param      a      { parameter_description }
     * @param      b      { parameter_description }
     * @param      c      { parameter_description }
     * @param[in]  color  The color
     */
    void fillTriangle(Support a, Support b, Support c, T color) {
        // create edges for the triangle
        Edge edges[3] = {Edge(a, b), Edge(b, c), Edge(c, a)};

        int maxLength = 0;
        int longEdge = 0;

        // find edge with the greatest length in the y axis
        for (int i = 0; i < 3; i++) {
            int length = edges[i].b.y - edges[i].a.y;
            if (length > maxLength) {
                maxLength = length;
                longEdge = i;
            }
        }

        int shortEdge1 = (longEdge + 1) % 3;
        int shortEdge2 = (longEdge + 2) % 3;

        // draw spans between edges; the long edge can be drawn
        // with the shorter edges to draw the full triangle
        drawSpansBetweenEdges(edges[longEdge], edges[shortEdge1], color);
        drawSpansBetweenEdges(edges[longEdge], edges[shortEdge2], color);
    }
    /**
     * @brief      Draws a triangle.
     *
     * @param      a      { parameter_description }
     * @param      b      { parameter_description }
     * @param      c      { parameter_description }
     * @param[in]  color  The color
     */
    void drawTriangle(Support &a, Support &b, Support &c, T color) {
        drawLine(a, b, color);
        drawLine(b, c, color);
        drawLine(c, a, color);
    }
    Buffer<RGBColor> convertToRGB() {
        Buffer<RGBColor> out(Eigen::Vector2i(Base::cols(), Base::rows()));

        T *ptr = reinterpret_cast<T *>(Base::data());
        int width = Base::cols();
        int height = Base::rows();
        out.width = this->width;

        // should fail graciously if heights incompatible
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out(y, x) = ndb::RGBColor(*ptr, *ptr, *ptr);
                ptr++;
            }
        }
        return out;
    }
};
class RGBBuffer : public Buffer<RGBColor> {
public:
    RGBBuffer() {}
    int readPNGRGB(std::string filename) {
        unsigned char header[8];  // 8 is the maximum size that can be checked
        png_byte colorType;
        png_byte bitDepth;

        png_structp pngPtr;
        png_infop infoPtr;
        png_bytep *rowPointers;

        // open file and test for it being a png
        FILE *fp = fopen(filename.c_str(), "rb");
        if (!fp) {
            cout << "ERR: File" << filename << " could not be opened for reading" << endl;
            return 1;
        }
        size_t res = fread(header, 1, 8, fp);
        if (png_sig_cmp(header, 0, 8)) {
            cout << "ERR: File" << filename << " is not recognized as a PNG file" << endl;
            return 1;
        }
        pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!pngPtr) {
            cout << "ERR: png_create_read_struct failed" << endl;
            return 1;
        }

        infoPtr = png_create_info_struct(pngPtr);
        if (!infoPtr) {
            cout << "ERR: png_create_info_struct failed" << endl;
            return 1;
        }

        if (setjmp(png_jmpbuf(pngPtr))) {
            cout << "ERR: Error during init_io" << endl;
            return 1;
        }

        png_init_io(pngPtr, fp);
        png_set_sig_bytes(pngPtr, 8);
        png_read_info(pngPtr, infoPtr);

        this->width = png_get_image_width(pngPtr, infoPtr);
        this->height = png_get_image_height(pngPtr, infoPtr);
        colorType = png_get_color_type(pngPtr, infoPtr);
        bitDepth = png_get_bit_depth(pngPtr, infoPtr);

        // We will do a conservative resize after reading in the data,
        // such that we don't have to translate addresses ourselves
        Base::resize(this->height, this->width);

        png_read_update_info(pngPtr, infoPtr);

        if (setjmp(png_jmpbuf(pngPtr))) {
            cout << "ERR: Error during read_image" << endl;
            return 1;
        }

        rowPointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
        for (int y = 0; y < height; y++)
            rowPointers[y] = (png_byte *)malloc(png_get_rowbytes(pngPtr, infoPtr));

        png_read_image(pngPtr, rowPointers);

        fclose(fp);
        // Read image into buffer (row-major)
        int nChannels;

        switch (png_get_color_type(pngPtr, infoPtr)) {
            case PNG_COLOR_TYPE_GRAY:
                nChannels = 1;
                break;
            case PNG_COLOR_TYPE_RGB:
                nChannels = 3;
                break;
            case PNG_COLOR_TYPE_RGBA:
                nChannels = 4;
                break;
            default:
                nChannels = 0;
                break;
        }
        RGBColor *ptr = reinterpret_cast<RGBColor *>(Base::data());
        if (bitDepth == 8) {
            if (nChannels == 3)
                for (int y = 0; y < this->height; y++) {
                    png_byte *row = rowPointers[y];
                    for (int x = 0; x < this->width; x++) {
                        int offset = y * this->cols() + x;
                        ptr[offset] =
                            RGBColor(row[3 * x], row[3 * x + 1], row[3 * x + 2]);
                    }
                }
        }
        // Conservative resize of underlying matrix to align with 16 byte boundary
        Base::conservativeResize(this->height, ALIGN16(this->width));
        if (nChannels == 0 || nChannels == 4) {
            cout << "ERR: found something other than gray or 3 channel color image("
                 << int(png_get_color_type(pngPtr, infoPtr)) << ") aborting!" << endl;

            return 1;
        }
        for (int y = 0; y < this->height; y++) free(rowPointers[y]);
        free(rowPointers);
        return 0;
    }
};
Buffer<RGBColor> getDisparityVisualization(
    ndb::Buffer<uint8_t> &srcImg,
    std::vector<int> &validEstimateIndices,
    ndb::Buffer<float> &disparity) {
    float min_disparity = 0;
    float max_disparity = 128;
    Buffer<RGBColor> dispVis(Eigen::Vector2i(srcImg.width, srcImg.rows()));
    for (int x = 0; x < srcImg.width; x++) {
        for (int y = 0; y < srcImg.height; y++) {
            uint8_t c = srcImg.getPixel(x, y);
            dispVis.setPixel(x, y, RGBColor(c, c, c));
        }
    }

    // Create color-coded reconstruction disparity visualization.
    // This uses Andreas Geiger's color map from the Kitti benchmark:
    float map[8][4] = {{0, 0, 0, 114}, {0, 0, 1, 185}, {1, 0, 0, 114}, {1, 0, 1, 174},
                       {0, 1, 0, 114}, {0, 1, 1, 185}, {1, 1, 0, 114}, {1, 1, 1, 0}};
    float sum = 0;
    for (int32_t i = 0; i < 8; ++i) {
        sum += map[i][3];
    }

    float weights[8];  // relative weights
    float cumsum[8];   // cumulative weights
    cumsum[0] = 0;
    for (int32_t i = 0; i < 7; ++i) {
        weights[i] = sum / map[i][3];
        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
    }

    // Copy image into a three channel color image first:
    for (auto &idx : validEstimateIndices) {
        // Pixel coords of disparity value
        int x = idx % srcImg.cols();
        int y = idx / srcImg.cols();

        uint8_t p = srcImg.getPixel(x, y);
        dispVis.setPixel(x, y, RGBColor(p, p, p));
        // Overwrite pixel in red if we have significant error.
        float reconstruction_disp = disparity.getPixel(x, y);
        float value = std::max(
            0.f, std::min(
                     0.8f, (reconstruction_disp - min_disparity) /
                               (max_disparity - min_disparity)));

        int32_t bin;
        for (bin = 0; bin < 7; ++bin) {
            if (value < cumsum[bin + 1]) {
                break;
            }
        }
        uint8_t colR, colG, colB;

        // Compute red/green/blue values.
        float w = 1.0f - (value - cumsum[bin]) * weights[bin];
        colR = static_cast<uint8_t>(
            (w * map[bin][0] + (1.0f - w) * map[bin + 1][0]) * 255.0f);
        colG = static_cast<uint8_t>(
            (w * map[bin][1] + (1.0f - w) * map[bin + 1][1]) * 255.0f);
        colB = static_cast<uint8_t>(
            (w * map[bin][2] + (1.0f - w) * map[bin + 1][2]) * 255.0f);

        dispVis.setPixel(x, y, RGBColor(colR, colG, colB));
    }
    return dispVis;
}
Buffer<RGBColor> getDisparityVisualization(
    ndb::Buffer<uint8_t> &srcImg, std::vector<Support> &support) {
    float min_disparity = 0;
    float max_disparity = 128;
    Buffer<RGBColor> dispVis(Eigen::Vector2i(srcImg.width, srcImg.rows()));
    dispVis = srcImg.convertToRGB();
    ;
    for (auto &s : support) dispVis.setPixel(s.x, s.y, RGBColor(s.d, s.d, s.d));

    // Create color-coded reconstruction disparity visualization.
    // This uses Andreas Geiger's color map from the Kitti benchmark:
    float map[8][4] = {
        {0, 0, 1, 185}, {1, 0, 0, 114}, {1, 0, 1, 174}, {0, 1, 0, 114},
        {0, 1, 1, 185}, {1, 1, 0, 114}, {1, 1, 1, 0},   {0, 0, 0, 114},
    };
    float sum = 0;
    for (int32_t i = 0; i < 8; ++i) {
        sum += map[i][3];
    }

    float weights[8];  // relative weights
    float cumsum[8];   // cumulative weights
    cumsum[0] = 0;
    for (int32_t i = 0; i < 7; ++i) {
        weights[i] = sum / map[i][3];
        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
    }

    // Copy image into a three channel color image first:
    for (auto &s : support) {
        // Pixel coords of disparity value
        int x = s.x;
        int y = s.y;

        // Overwrite pixel in red if we have significant error.
        float reconstruction_disp = s.d;
        float value = std::max(
            0.f, std::min(
                     0.8f, (reconstruction_disp - min_disparity) /
                               (max_disparity - min_disparity)));

        int32_t bin;
        for (bin = 0; bin < 7; ++bin) {
            if (value < cumsum[bin + 1]) {
                break;
            }
        }
        uint8_t colR, colG, colB;

        // Compute red/green/blue values.
        float w = 1.0f - (value - cumsum[bin]) * weights[bin];
        colR = static_cast<uint8_t>(
            (w * map[bin][0] + (1.0f - w) * map[bin + 1][0]) * 255.0f);
        colG = static_cast<uint8_t>(
            (w * map[bin][1] + (1.0f - w) * map[bin + 1][1]) * 255.0f);
        colB = static_cast<uint8_t>(
            (w * map[bin][2] + (1.0f - w) * map[bin + 1][2]) * 255.0f);

        dispVis.setPixel(x, y, RGBColor(colR, colG, colB));
    }
    return dispVis;
}
}  // namespace ndb
#endif
