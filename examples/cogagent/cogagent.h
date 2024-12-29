#pragma once

#include "vision_encoder.h"
#include "cross_vision.h"

struct cogagent_ctx {
    // Vision encoder and cross vision encoder models
    vision_encoder_ctx vision_encoder;
    cross_vision_ctx cross_vision;

    struct llama_context * ctx_llama;
    struct llama_model * cogvlm_model;

    // I think this will only be for the
    // vision encoder and cross vision encoder
    ggml_backend_t backend;

    std::string user_prompt;
    std::vector<uint8_t> image;
    std::vector<uint8_t> small_image;  // Small image size for vision encoder
    std::vector<uint8_t> large_image;  // Large image size for cross vision encoder
};

extern cogagent_ctx cogagent_global;