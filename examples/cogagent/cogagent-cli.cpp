#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cogagent.h"

cogagent_ctx cogagent_global;

// This function is mostly copied from cogagent cli
static bool eval_string_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();

    set_processing_text(ctx_llama, true);

    //// Processing the input tokens in batches
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool eval_image_tokens(llama_context * ctx_llama, std::vector<float> &img_data,
        int n_batch, int * n_past) {
    int n_embd = 4096;
    int num_tokens = 258;
    int positions[258];

    set_processing_text(ctx_llama, false);

    positions[0] = *n_past;
    for (int i=0; i<num_tokens-2; i++) {
        positions[i + 1] = *n_past + 1;
    }
    positions[num_tokens - 1] = *n_past + 2;

    float * data_ptr = img_data.data();

    for (int i = 0; i < num_tokens; i += n_batch) {
        int n_eval = num_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, data_ptr, positions, nullptr, nullptr, nullptr, 0, 0, 0, };
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        data_ptr += i * n_embd;
    }
    *n_past += 3;
    return true;
}

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <cogagent-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <cogagent-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static const char * sample(struct gpt_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = gpt_sampler_sample(smpl, ctx_llama, -1);
    gpt_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    // Give the new token to the model. I'm not sure how it is stored.
    // Perhaps it is stored in the KV cache.
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    eval_string_tokens(ctx_llama, tokens, 1, n_past);

    return ret.c_str();
}

int main(int argc, char ** argv) {
    ggml_time_init();
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_COGAGENT, print_usage)) {
        return 1;
    }
    gpt_init();

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == nullptr) {
        printf("Failed to load decoder model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    printf("Context size is %d tokens\n", ctx_params.n_ctx);
    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == nullptr) {
        printf("Failed to create the llama context\n");
        return 1;
    }

    cogagent_global.ctx_llama = ctx_llama;
    cogagent_global.cogvlm_model = model;

    // Load the image tensors
    std::vector<float> small_encoded_picture;
    const char * small_picture_file = "/home/tianyue/myworkspace/"
        "vlm_intermediate/reference_vision_encoder_real_image.gguf";
    get_input(small_encoded_picture, small_picture_file);

    std::vector<float> big_encoded_picture;
    const char * big_picture_file = "/home/tianyue/myworkspace/"
        "vlm_intermediate/reference_cross_vision_encoder_real_image.gguf";
    get_input(big_encoded_picture, big_picture_file);

    // Give the output from the cross vision encoder to the llama context
    set_cross_input(cogagent_global.ctx_llama, big_encoded_picture);

    // At the moment I can't figure out how the llama kv cache
    // keeps its information across runs.
    // It seems to me that the graph is allocated for each batch,
    // which would invalidate any tensors stored in the kv cache.
    // I don't spot logic for separately allocating the kv cache
    // tensors to avoid this, so it doesn't make sense.
    // Maybe the graph isn't actually allocated for each batch?
    // Perhaps that is why a worst case graph is allocated.

    // TODO: Check if system prompt is compatible
    std::vector<llama_token> begin_token;
    begin_token.push_back(llama_token_bos(cogagent_global.cogvlm_model));

    int n_past = 0;
    printf("Run model with bos token.\n");
    clear_cross_kv(cogagent_global.ctx_llama);
    eval_string_tokens(cogagent_global.ctx_llama,
        begin_token, params.n_batch, &n_past);
    printf("Run model with image tokens.\n");
    eval_image_tokens(cogagent_global.ctx_llama, small_encoded_picture,
        params.n_batch, &n_past);
    // Tokenize user prompt
    // Third option set to false to that the tokenizer doesn't add
    // beginning of sentence and end of sentence
    std::vector<llama_token> user_prompt_tokens = ::llama_tokenize(
        cogagent_global.ctx_llama, params.prompt, false, true
    );
    printf("Run model with user entered text tokens.\n");
    eval_string_tokens(cogagent_global.ctx_llama, user_prompt_tokens,
        params.n_batch, &n_past);

    printf("Parsed maximum sampling length %d.\n", params.n_predict);
    int max_len = params.n_predict < 0 ? 256 : params.n_predict;

    struct gpt_sampler * smpl = gpt_sampler_init(cogagent_global.cogvlm_model, params.sparams);
    if (!smpl) {
        printf("Failed to initialize sampler.\n");
        return 1;
    }
    printf("\nReprinting entered prompt.\n %s \n", params.prompt.c_str());
    printf("\n\n Beginning of response.\n");
    std::string response = "";
    for (int i=0; i<max_len; ++i) {
        const char * tmp = sample(smpl, cogagent_global.ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) {
            break;
        }
        printf("%s", tmp);
        fflush(stdout);
    }
    gpt_sampler_free(smpl);

    llama_free_model(model);
    return 0;
}