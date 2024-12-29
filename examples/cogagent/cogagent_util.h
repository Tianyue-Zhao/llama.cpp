#pragma once

#include "ggml.h"
#include <string>

void print_dims(struct ggml_tensor * input_tensor, const char * name) {
    printf("Tensor %s has shape %ld x %ld x %ld x %ld and data type %d\n", name,
        input_tensor->ne[0], input_tensor->ne[1], input_tensor->ne[2],
        input_tensor->ne[3], input_tensor->type);
}

struct ggml_tensor * get_tensor(struct ggml_context * dst_ctx, struct ggml_context * src_ctx, std::string tensor_name, int &count_failed) {
    struct ggml_tensor * cur_tensor = ggml_get_tensor(src_ctx, tensor_name.c_str());
    if (!cur_tensor) {
        printf("Retrieval of tensor %s from model context failed\n", tensor_name.c_str());
        count_failed++;
        return nullptr;
    }
    struct ggml_tensor * new_tensor = ggml_dup_tensor(dst_ctx, cur_tensor);
    ggml_set_name(new_tensor, cur_tensor->name);
    return new_tensor;
}

void save_tensor_filename(struct ggml_tensor * input_tensor, std::string filename) {
    std::string prefix = "/home/tianyue/myworkspace/";
    filename = prefix + filename;
    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "model.architecture", "cogagent");
    gguf_set_val_u32(gguf_ctx, "general.file_type", GGML_TYPE_F32);

    struct ggml_init_params params = {
        ggml_nbytes(input_tensor) + 1000000,  // Memory to allocate
        nullptr,  // Buffer location
        false,  // Allocate tensor data
    };
    struct ggml_context * tensor_ctx = ggml_init(params);
    struct ggml_tensor * tensor_with_data = ggml_dup(tensor_ctx, input_tensor);
    ggml_backend_tensor_get(input_tensor, tensor_with_data->data,
        0, ggml_nbytes(input_tensor));

    ggml_set_name(tensor_with_data, "output_tensor");
    gguf_add_tensor(gguf_ctx, tensor_with_data);
    gguf_write_to_file(gguf_ctx, filename.c_str(), false);
    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
}

// Function that loads data from the GGUF file to a temporary buffer
// and then from the temporary buffer to the GGML backend buffer
// Copied from the GGML MNIST example
bool load_from_gguf(const char * fname, struct ggml_context * ctx_ggml, struct gguf_context * ctx_gguf) {
    FILE * f = ggml_fopen(fname, "rb");
    if (!f) {
        return false;
    }

    const size_t buf_size = 4*1024*1024;
    void * buf = malloc(buf_size);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);

        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);
        if (!tensor) {
            printf("Tensor found in GGUF file but not in GGML context: %s\n", name);
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

        if (fseek(f, offs, SEEK_SET) != 0) {
            fclose(f);
            free(buf);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = buf_size < nbytes - pos ? buf_size : nbytes - pos;

            if (fread(buf, 1, nbytes_cpy, f) != nbytes_cpy) {
                fclose(f);
                free(buf);
                return false;
            }

            ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
        }
    }

    fclose(f);
    free(buf);
    return true;
}

int get_input(
    std::vector<float> &input_data,
    const char * filename
) {
    struct ggml_context * meta_info;

    struct gguf_init_params gguf_params = {
        true, &meta_info,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(filename, gguf_params);
    if (!gguf_ctx) {
        printf("Failed to initialize GGUF context. Check filename.\n");
        return false;
    }

    // I don't know how to set tensor name when writing a GGUF file
    // in cpp, so the cross vision output tensor doesn't appear
    // to have a name when reading the GGUF file
    struct ggml_tensor * meta_tensor = ggml_get_first_tensor(meta_info);
    if (meta_tensor->type != GGML_TYPE_F32) {
        printf("Expected the input image datatype to be float 32.\n");
        printf("Image loading failed because the datatype is actually %d\n",
            meta_tensor->type);
        return -1;
    }

    size_t tensor_size = ggml_nbytes(meta_tensor);
    printf("Input tensor size is %ld bytes\n", tensor_size);

    input_data.resize(meta_tensor->ne[0] * meta_tensor->ne[1] *
        meta_tensor->ne[2]);
    int num_tokens = meta_tensor->ne[1];

    std::ifstream input_file = std::ifstream(filename, std::ios::binary);
    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, 0);
    input_file.seekg(offset, std::ios::beg);
    printf("Seeked to input tensor GGUF position %ld\n", offset);
    input_file.read(reinterpret_cast<char *>(input_data.data()), tensor_size);
    input_file.close();
    ggml_free(meta_info);
    return num_tokens;
}

// Copied from the LLAVA example

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}
void clip_image_f32_batch_free(struct clip_image_f32_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}

static void build_clip_img_from_data(const stbi_uc * data, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), data, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// Linear interpolation between two points
inline float clip_lerp(float s, float e, float t) {
    return s + (e - s) * t;
}
// Bilinear resize function
static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float x_ratio = static_cast<float>(src.nx - 1) / target_width;
    float y_ratio = static_cast<float>(src.ny - 1) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = clip_lerp(
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                float bottom = clip_lerp(
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(clip_lerp(top, bottom, y_lerp));
            }
        }
    }
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

static bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8& image, clip_image_u8 &image_output, const std::pair<int, int>& target_resolution) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}