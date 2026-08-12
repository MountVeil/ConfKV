#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* lmcache_gpu_aesgcm_key_t;

enum {
    LMCACHE_GPU_AESGCM_OK = 0,
    LMCACHE_GPU_AESGCM_ERR_ARGUMENT = -1,
    LMCACHE_GPU_AESGCM_ERR_ALLOC = -2,
    LMCACHE_GPU_AESGCM_ERR_CUDA = -3,
    LMCACHE_GPU_AESGCM_ERR_INTERNAL = -4,
};

size_t lmcache_gpu_aesgcm_frame_size(
    size_t plaintext_len
);

int lmcache_gpu_aes128gcm_key_create(
    const uint8_t* key,
    size_t key_len,
    int device,
    lmcache_gpu_aesgcm_key_t* out
);

int lmcache_gpu_aes128gcm_key_destroy(
    lmcache_gpu_aesgcm_key_t key
);

/*
 * Preallocate GHASH workspace.
 *
 * Call outside the timed path. seal/open will grow the workspace
 * automatically if needed, but that path may invoke cudaMalloc.
 */
int lmcache_gpu_aes128gcm_reserve(
    lmcache_gpu_aesgcm_key_t key,
    size_t max_plaintext_len
);

/*
 * Asynchronously produce:
 *
 * [version=1][12-byte IV][ciphertext][16-byte tag]
 *
 * src and dst are DEVICE pointers.
 * iv is a HOST pointer containing exactly 12 bytes.
 * stream is cudaStream_t cast to void*.
 *
 * One key handle must not have multiple overlapping operations on
 * different CUDA streams because its GHASH workspace is reused.
 */
int lmcache_gpu_aes128gcm_seal_async(
    lmcache_gpu_aesgcm_key_t key,
    const void* src,
    size_t plaintext_len,
    void* dst,
    size_t dst_capacity,
    const uint8_t* iv,
    void* stream
);

/*
 * Authentication-gated asynchronous open.
 *
 * src/dst/auth_ok are DEVICE pointers.
 *
 * auth_ok receives:
 *   1 -> tag/version valid
 *   0 -> authentication failed
 *
 * On authentication failure, dst is zero-filled. No unauthenticated
 * plaintext is intentionally published.
 */
int lmcache_gpu_aes128gcm_open_async(
    lmcache_gpu_aesgcm_key_t key,
    const void* src,
    size_t frame_len,
    void* dst,
    size_t plaintext_len,
    int* auth_ok,
    void* stream
);

const char* lmcache_gpu_aesgcm_strerror(
    int rc
);

#ifdef __cplusplus
}
#endif
