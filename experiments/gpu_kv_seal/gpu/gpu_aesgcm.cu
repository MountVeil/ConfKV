#include "gpu_aesgcm_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>


namespace {

constexpr size_t VERSION_LEN = 1;
constexpr size_t IV_LEN = 12;
constexpr size_t TAG_LEN = 16;
constexpr size_t HEADER_LEN = VERSION_LEN + IV_LEN;
constexpr size_t FRAME_OVERHEAD = HEADER_LEN + TAG_LEN;

constexpr uint8_t VERSION = 1;

constexpr int AES_BLOCK = 16;
constexpr int AES_ROUNDS = 10;
constexpr int AES_ROUND_KEY_BYTES = 176;

constexpr uint64_t GHASH_SEGMENT_BLOCKS = 64;


/* ================================================================
 * AES S-box
 *
 * NOTE:
 * This first implementation uses the conventional table-based S-box.
 * It is intended to establish functional and performance feasibility.
 * Microarchitectural side-channel hardening is a separate security
 * question and must be handled explicitly before claiming such a
 * threat model.
 * ================================================================ */

static constexpr uint8_t HOST_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,
    0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
    0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,
    0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,
    0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
    0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,
    0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,
    0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
    0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,
    0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,
    0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
    0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,
    0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,
    0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
    0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,
    0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,
    0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__device__ __constant__ uint8_t DEVICE_SBOX[256];


struct Block128 {
    uint64_t hi;
    uint64_t lo;
};


struct GHashNode {
    Block128 y;
    uint64_t len_blocks;
};


struct Iv96 {
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
};


struct KeyHandle {
    int device = 0;

    uint8_t* d_round_keys = nullptr;

    /*
     * H^(2^i), i = 0..63.
     *
     * Allows tree reduction to calculate H^N using only popcount(N)
     * field multiplications instead of recomputing powers from scratch.
     */
    Block128* d_h_pow2 = nullptr;

    /*
     * GHASH reduction workspace.
     */
    GHashNode* d_nodes_a = nullptr;
    GHashNode* d_nodes_b = nullptr;

    size_t node_capacity = 0;
};


/* ================================================================
 * Utility
 * ================================================================ */

inline int cuda_rc(cudaError_t rc)
{
    return rc == cudaSuccess
        ? LMCACHE_GPU_AESGCM_OK
        : LMCACHE_GPU_AESGCM_ERR_CUDA;
}


__host__ __device__
inline Block128 bxor(Block128 a, Block128 b)
{
    return {
        a.hi ^ b.hi,
        a.lo ^ b.lo
    };
}


__device__
inline Block128 load_be128(const uint8_t* p)
{
    Block128 x {0, 0};

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x.hi = (x.hi << 8) | p[i];
    }

    #pragma unroll
    for (int i = 8; i < 16; ++i) {
        x.lo = (x.lo << 8) | p[i];
    }

    return x;
}


__device__
inline void store_be128(
    uint8_t* p,
    Block128 x)
{
    #pragma unroll
    for (int i = 7; i >= 0; --i) {
        p[i] = static_cast<uint8_t>(x.hi);
        x.hi >>= 8;
    }

    #pragma unroll
    for (int i = 15; i >= 8; --i) {
        p[i] = static_cast<uint8_t>(x.lo);
        x.lo >>= 8;
    }
}


/* ================================================================
 * GF(2^128)
 *
 * NIST GHASH representation.
 *
 * Irreducible polynomial:
 *
 *   x^128 + x^7 + x^2 + x + 1
 *
 * This generic implementation intentionally avoids relying on clmad
 * so it can compile on CUDA toolkits older than 13.3.
 *
 * CUDA >= 13.3 fast path can replace only this primitive later.
 * ================================================================ */

__device__
inline Block128 gf_mul(
    Block128 x,
    Block128 y)
{
    Block128 z {0, 0};
    Block128 v = y;

    #pragma unroll 1
    for (int i = 0; i < 128; ++i) {

        uint64_t bit;

        if (i < 64) {
            bit = (
                x.hi
                >> (63 - i)
            ) & 1ULL;
        } else {
            bit = (
                x.lo
                >> (127 - i)
            ) & 1ULL;
        }

        const uint64_t xmask =
            0ULL - bit;

        z.hi ^= v.hi & xmask;
        z.lo ^= v.lo & xmask;

        const uint64_t lsb =
            v.lo & 1ULL;

        const uint64_t new_lo =
            (v.lo >> 1)
            | (v.hi << 63);

        uint64_t new_hi =
            v.hi >> 1;

        const uint64_t rmask =
            0ULL - lsb;

        new_hi ^=
            0xe100000000000000ULL
            & rmask;

        v.hi = new_hi;
        v.lo = new_lo;
    }

    return z;
}


/*
 * Multiplicative identity under the GHASH bit representation.
 */
__device__
inline Block128 gf_one()
{
    return {
        0x8000000000000000ULL,
        0ULL
    };
}


__device__
inline Block128 gf_pow_from_table(
    const Block128* h_pow2,
    uint64_t exponent)
{
    Block128 result = gf_one();

    int bit = 0;

    while (exponent != 0) {

        if (exponent & 1ULL) {
            result = gf_mul(
                result,
                h_pow2[bit]
            );
        }

        exponent >>= 1;
        ++bit;
    }

    return result;
}


/* ================================================================
 * AES-128
 * ================================================================ */

__device__
inline uint8_t xtime(uint8_t x)
{
    return static_cast<uint8_t>(
        (x << 1)
        ^ ((x & 0x80) ? 0x1b : 0)
    );
}


/*
 * Expand the raw AES-128 K_store entirely on the GPU.
 *
 * ConfKV provisioning model:
 *
 *   TDX guest raw K_store (16 B)
 *       -> CUDA H2D
 *       -> GPU temporary raw key
 *       -> this kernel
 *       -> GPU round-key schedule (176 B)
 *
 * In H100 CC-On, the H2D transfer is protected by NVIDIA's
 * confidential-computing transport below the CUDA application API.
 *
 * Only one thread is needed because key setup happens once per key
 * lifecycle, not once per KV chunk.
 */
__global__
void expand_aes128_key_kernel(
    const uint8_t* key,
    uint8_t* round_keys)
{
    if (
        blockIdx.x != 0
        || threadIdx.x != 0
    ) {
        return;
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        round_keys[i] = key[i];
    }

    int generated = 16;
    uint8_t rcon = 1;
    uint8_t temp[4];

    while (
        generated
        < AES_ROUND_KEY_BYTES
    ) {

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            temp[i] =
                round_keys[
                    generated - 4 + i
                ];
        }

        if (
            generated % 16
            == 0
        ) {

            const uint8_t t =
                temp[0];

            temp[0] =
                DEVICE_SBOX[temp[1]];

            temp[1] =
                DEVICE_SBOX[temp[2]];

            temp[2] =
                DEVICE_SBOX[temp[3]];

            temp[3] =
                DEVICE_SBOX[t];

            temp[0] ^= rcon;

            rcon =
                xtime(rcon);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {

            round_keys[generated] =
                round_keys[
                    generated - 16
                ]
                ^ temp[i];

            ++generated;
        }
    }
}


/*
 * Explicitly erase temporary secret material in device memory.
 *
 * This is used for the raw 16-byte K_store after GPU-side
 * key expansion.
 */
__global__
void zeroize_bytes_kernel(
    uint8_t* ptr,
    size_t len)
{
    const size_t i =
        static_cast<size_t>(
            blockIdx.x
        )
        * blockDim.x
        + threadIdx.x;

    if (i < len) {
        ptr[i] = 0;
    }
}


__device__
inline void add_round_key(
    uint8_t s[16],
    const uint8_t* rk)
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[i] ^= rk[i];
    }
}


__device__
inline void sub_bytes(
    uint8_t s[16])
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[i] =
            DEVICE_SBOX[s[i]];
    }
}


__device__
inline void shift_rows(
    uint8_t s[16])
{
    uint8_t t;

    /*
     * Row 1: rotate left by 1.
     */
    t = s[1];
    s[1] = s[5];
    s[5] = s[9];
    s[9] = s[13];
    s[13] = t;

    /*
     * Row 2: rotate left by 2.
     */
    t = s[2];
    s[2] = s[10];
    s[10] = t;

    t = s[6];
    s[6] = s[14];
    s[14] = t;

    /*
     * Row 3: rotate left by 3.
     */
    t = s[15];
    s[15] = s[11];
    s[11] = s[7];
    s[7] = s[3];
    s[3] = t;
}


__device__
inline void mix_columns(
    uint8_t s[16])
{
    #pragma unroll
    for (int c = 0; c < 4; ++c) {

        const int i = c * 4;

        const uint8_t a0 = s[i + 0];
        const uint8_t a1 = s[i + 1];
        const uint8_t a2 = s[i + 2];
        const uint8_t a3 = s[i + 3];

        const uint8_t all =
            a0 ^ a1 ^ a2 ^ a3;

        s[i + 0] ^=
            all ^ xtime(a0 ^ a1);

        s[i + 1] ^=
            all ^ xtime(a1 ^ a2);

        s[i + 2] ^=
            all ^ xtime(a2 ^ a3);

        s[i + 3] ^=
            all ^ xtime(a3 ^ a0);
    }
}


__device__
inline void aes128_encrypt_block(
    const uint8_t in[16],
    uint8_t out[16],
    const uint8_t* round_keys)
{
    uint8_t s[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[i] = in[i];
    }

    add_round_key(
        s,
        round_keys
    );

    #pragma unroll
    for (
        int round = 1;
        round < AES_ROUNDS;
        ++round
    ) {
        sub_bytes(s);
        shift_rows(s);
        mix_columns(s);

        add_round_key(
            s,
            round_keys
                + round * 16
        );
    }

    sub_bytes(s);
    shift_rows(s);

    add_round_key(
        s,
        round_keys
            + AES_ROUNDS * 16
    );

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = s[i];
    }
}


/* ================================================================
 * AES-GCM kernels
 * ================================================================ */

__global__
void init_hash_key_kernel(
    const uint8_t* round_keys,
    Block128* h_pow2)
{
    if (
        blockIdx.x != 0
        || threadIdx.x != 0
    ) {
        return;
    }

    uint8_t zero[16] = {0};
    uint8_t out[16];

    aes128_encrypt_block(
        zero,
        out,
        round_keys
    );

    h_pow2[0] =
        load_be128(out);

    for (int i = 1; i < 64; ++i) {
        h_pow2[i] =
            gf_mul(
                h_pow2[i - 1],
                h_pow2[i - 1]
            );
    }
}


__global__
void write_frame_header_kernel(
    uint8_t* frame,
    Iv96 iv)
{
    if (
        blockIdx.x != 0
        || threadIdx.x != 0
    ) {
        return;
    }

    frame[0] = VERSION;

    const uint8_t* iv_bytes =
        reinterpret_cast<
            const uint8_t*
        >(&iv);

    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        frame[1 + i] =
            iv_bytes[i];
    }
}


__device__
inline void make_counter_block(
    const uint8_t* iv,
    uint32_t counter,
    uint8_t block[16])
{
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        block[i] = iv[i];
    }

    block[12] =
        static_cast<uint8_t>(
            counter >> 24
        );

    block[13] =
        static_cast<uint8_t>(
            counter >> 16
        );

    block[14] =
        static_cast<uint8_t>(
            counter >> 8
        );

    block[15] =
        static_cast<uint8_t>(
            counter
        );
}


__global__
void ctr_crypt_kernel(
    const uint8_t* src,
    uint8_t* dst,
    size_t len,
    const uint8_t* iv,
    const uint8_t* round_keys,
    const int* auth_ok)
{
    const uint64_t block_index =
        static_cast<uint64_t>(
            blockIdx.x
        )
        * blockDim.x
        + threadIdx.x;

    const uint64_t block_count =
        (
            static_cast<uint64_t>(len)
            + 15
        ) / 16;

    if (
        block_index
        >= block_count
    ) {
        return;
    }

    const size_t offset =
        static_cast<size_t>(
            block_index * 16
        );

    /*
     * Authentication-gated decrypt:
     *
     * failed authentication never publishes provisional plaintext.
     */
    if (
        auth_ok != nullptr
        && *auth_ok == 0
    ) {

        #pragma unroll
        for (int j = 0; j < 16; ++j) {

            const size_t pos =
                offset + j;

            if (pos < len) {
                dst[pos] = 0;
            }
        }

        return;
    }

    /*
     * For a 96-bit IV:
     *
     *   J0 = IV || 0x00000001
     *
     * First data counter is inc32(J0), i.e. 2.
     */
    const uint64_t ctr64 =
        block_index + 2ULL;

    if (
        ctr64
        > 0xffffffffULL
    ) {
        return;
    }

    uint8_t counter_block[16];
    uint8_t stream_block[16];

    make_counter_block(
        iv,
        static_cast<uint32_t>(ctr64),
        counter_block
    );

    aes128_encrypt_block(
        counter_block,
        stream_block,
        round_keys
    );

    #pragma unroll
    for (int j = 0; j < 16; ++j) {

        const size_t pos =
            offset + j;

        if (pos < len) {
            dst[pos] =
                src[pos]
                ^ stream_block[j];
        }
    }
}


__device__
inline Block128 load_ciphertext_term(
    const uint8_t* ciphertext,
    size_t ciphertext_len,
    uint64_t term_index,
    uint64_t cipher_blocks)
{
    if (
        term_index
        == cipher_blocks
    ) {
        /*
         * Final GCM length block:
         *
         *   [64-bit AAD length in bits = 0]
         *   [64-bit ciphertext length in bits]
         */
        return {
            0ULL,
            static_cast<uint64_t>(
                ciphertext_len
            ) * 8ULL
        };
    }

    const size_t offset =
        static_cast<size_t>(
            term_index * 16
        );

    uint8_t tmp[16] = {0};

    #pragma unroll
    for (int j = 0; j < 16; ++j) {

        const size_t pos =
            offset + j;

        if (pos < ciphertext_len) {
            tmp[j] =
                ciphertext[pos];
        }
    }

    return load_be128(tmp);
}


__global__
void ghash_segment_kernel(
    const uint8_t* ciphertext,
    size_t ciphertext_len,
    const Block128* h_pow2,
    GHashNode* nodes)
{
    const uint64_t segment_index =
        static_cast<uint64_t>(
            blockIdx.x
        )
        * blockDim.x
        + threadIdx.x;

    const uint64_t cipher_blocks =
        (
            static_cast<uint64_t>(
                ciphertext_len
            )
            + 15
        ) / 16;

    const uint64_t total_terms =
        cipher_blocks + 1;

    const uint64_t start =
        segment_index
        * GHASH_SEGMENT_BLOCKS;

    if (
        start
        >= total_terms
    ) {
        return;
    }

    const uint64_t end =
        min(
            start
                + GHASH_SEGMENT_BLOCKS,
            total_terms
        );

    const Block128 h =
        h_pow2[0];

    Block128 y {0, 0};

    for (
        uint64_t i = start;
        i < end;
        ++i
    ) {

        const Block128 x =
            load_ciphertext_term(
                ciphertext,
                ciphertext_len,
                i,
                cipher_blocks
            );

        y = gf_mul(
            bxor(y, x),
            h
        );
    }

    nodes[segment_index].y =
        y;

    nodes[segment_index].len_blocks =
        end - start;
}


__global__
void ghash_combine_kernel(
    const GHashNode* in,
    GHashNode* out,
    size_t count,
    const Block128* h_pow2)
{
    const size_t out_index =
        static_cast<size_t>(
            blockIdx.x
        )
        * blockDim.x
        + threadIdx.x;

    const size_t left =
        out_index * 2;

    if (left >= count) {
        return;
    }

    const GHashNode a =
        in[left];

    if (
        left + 1
        >= count
    ) {
        out[out_index] = a;
        return;
    }

    const GHashNode b =
        in[left + 1];

    /*
     * If A hashes LA blocks and B hashes LB blocks:
     *
     *   GHASH(A || B)
     *     = GHASH(A) * H^LB
     *       XOR GHASH(B)
     */
    const Block128 h_to_lb =
        gf_pow_from_table(
            h_pow2,
            b.len_blocks
        );

    out[out_index].y =
        bxor(
            gf_mul(
                a.y,
                h_to_lb
            ),
            b.y
        );

    out[out_index].len_blocks =
        a.len_blocks
        + b.len_blocks;
}


__global__
void finalize_tag_kernel(
    uint8_t* frame,
    size_t plaintext_len,
    const uint8_t* round_keys,
    const GHashNode* root)
{
    if (
        blockIdx.x != 0
        || threadIdx.x != 0
    ) {
        return;
    }

    const uint8_t* iv =
        frame + 1;

    uint8_t j0[16];
    uint8_t e_j0[16];

    make_counter_block(
        iv,
        1,
        j0
    );

    aes128_encrypt_block(
        j0,
        e_j0,
        round_keys
    );

    uint8_t ghash[16];

    store_be128(
        ghash,
        root->y
    );

    uint8_t* tag =
        frame
        + HEADER_LEN
        + plaintext_len;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        tag[i] =
            e_j0[i]
            ^ ghash[i];
    }
}


__global__
void verify_tag_kernel(
    const uint8_t* frame,
    size_t plaintext_len,
    const uint8_t* round_keys,
    const GHashNode* root,
    int* auth_ok)
{
    if (
        blockIdx.x != 0
        || threadIdx.x != 0
    ) {
        return;
    }

    const uint8_t* iv =
        frame + 1;

    uint8_t j0[16];
    uint8_t e_j0[16];

    make_counter_block(
        iv,
        1,
        j0
    );

    aes128_encrypt_block(
        j0,
        e_j0,
        round_keys
    );

    uint8_t ghash[16];

    store_be128(
        ghash,
        root->y
    );

    const uint8_t* tag =
        frame
        + HEADER_LEN
        + plaintext_len;

    unsigned int diff = 0;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {

        const uint8_t expected =
            e_j0[i]
            ^ ghash[i];

        diff |= static_cast<unsigned int>(
            expected
            ^ tag[i]
        );
    }

    const int version_ok =
        frame[0] == VERSION;

    *auth_ok =
        (
            version_ok
            && diff == 0
        )
        ? 1
        : 0;
}


/* ================================================================
 * Workspace
 * ================================================================ */

size_t required_nodes(
    size_t plaintext_len)
{
    const uint64_t cipher_blocks =
        (
            static_cast<uint64_t>(
                plaintext_len
            )
            + 15
        ) / 16;

    const uint64_t terms =
        cipher_blocks + 1;

    return static_cast<size_t>(
        (
            terms
            + GHASH_SEGMENT_BLOCKS
            - 1
        )
        / GHASH_SEGMENT_BLOCKS
    );
}


int ensure_workspace(
    KeyHandle* h,
    size_t plaintext_len)
{
    const size_t required =
        required_nodes(
            plaintext_len
        );

    if (
        required
        <= h->node_capacity
    ) {
        return LMCACHE_GPU_AESGCM_OK;
    }

    GHashNode* a = nullptr;
    GHashNode* b = nullptr;

    cudaError_t rc =
        cudaMalloc(
            &a,
            required
                * sizeof(GHashNode)
        );

    if (rc != cudaSuccess) {
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    rc =
        cudaMalloc(
            &b,
            required
                * sizeof(GHashNode)
        );

    if (rc != cudaSuccess) {
        cudaFree(a);
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    if (h->d_nodes_a != nullptr) {
        cudaFree(h->d_nodes_a);
    }

    if (h->d_nodes_b != nullptr) {
        cudaFree(h->d_nodes_b);
    }

    h->d_nodes_a = a;
    h->d_nodes_b = b;
    h->node_capacity = required;

    return LMCACHE_GPU_AESGCM_OK;
}


/*
 * Enqueue complete GHASH and return the device pointer holding the root.
 */
int enqueue_ghash(
    KeyHandle* h,
    const uint8_t* ciphertext,
    size_t ciphertext_len,
    cudaStream_t stream,
    GHashNode** root_out)
{
    const int erc =
        ensure_workspace(
            h,
            ciphertext_len
        );

    if (
        erc
        != LMCACHE_GPU_AESGCM_OK
    ) {
        return erc;
    }

    size_t count =
        required_nodes(
            ciphertext_len
        );

    constexpr int threads = 128;

    const int blocks =
        static_cast<int>(
            (
                count
                + threads
                - 1
            )
            / threads
        );

    ghash_segment_kernel<<<
        blocks,
        threads,
        0,
        stream
    >>>(
        ciphertext,
        ciphertext_len,
        h->d_h_pow2,
        h->d_nodes_a
    );

    if (
        cudaPeekAtLastError()
        != cudaSuccess
    ) {
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    GHashNode* in =
        h->d_nodes_a;

    GHashNode* out =
        h->d_nodes_b;

    while (count > 1) {

        const size_t next =
            (
                count + 1
            ) / 2;

        const int reduce_blocks =
            static_cast<int>(
                (
                    next
                    + threads
                    - 1
                )
                / threads
            );

        ghash_combine_kernel<<<
            reduce_blocks,
            threads,
            0,
            stream
        >>>(
            in,
            out,
            count,
            h->d_h_pow2
        );

        if (
            cudaPeekAtLastError()
            != cudaSuccess
        ) {
            return LMCACHE_GPU_AESGCM_ERR_CUDA;
        }

        std::swap(
            in,
            out
        );

        count = next;
    }

    *root_out = in;

    return LMCACHE_GPU_AESGCM_OK;
}


class DeviceGuard {
public:
    explicit DeviceGuard(int target)
    {
        cudaGetDevice(&old_);
        changed_ = (
            old_ != target
        );

        if (changed_) {
            cudaSetDevice(target);
        }
    }

    ~DeviceGuard()
    {
        if (changed_) {
            cudaSetDevice(old_);
        }
    }

private:
    int old_ = 0;
    bool changed_ = false;
};


}  // namespace


extern "C"
size_t lmcache_gpu_aesgcm_frame_size(
    size_t plaintext_len)
{
    return (
        plaintext_len
        + FRAME_OVERHEAD
    );
}


extern "C"
int lmcache_gpu_aes128gcm_key_create(
    const uint8_t* key,
    size_t key_len,
    int device,
    lmcache_gpu_aesgcm_key_t* out)
{
    if (
        key == nullptr
        || out == nullptr
        || key_len != 16
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    *out = nullptr;

    DeviceGuard guard(device);

    auto* h =
        new (
            std::nothrow
        ) KeyHandle();

    if (h == nullptr) {
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    h->device = device;

    uint8_t* d_raw_key = nullptr;

    /*
     * Best-effort cleanup for every failed provisioning path.
     *
     * Both the raw GPU key and derived GPU round keys are secret
     * material and are erased before their allocations are released.
     */
    auto cleanup_failure = [&]() {

        if (d_raw_key != nullptr) {
            cudaMemset(
                d_raw_key,
                0,
                16
            );

            cudaFree(
                d_raw_key
            );

            d_raw_key = nullptr;
        }

        if (h->d_h_pow2 != nullptr) {
            cudaMemset(
                h->d_h_pow2,
                0,
                sizeof(Block128) * 64
            );

            cudaFree(
                h->d_h_pow2
            );

            h->d_h_pow2 = nullptr;
        }

        if (h->d_round_keys != nullptr) {
            cudaMemset(
                h->d_round_keys,
                0,
                AES_ROUND_KEY_BYTES
            );

            cudaFree(
                h->d_round_keys
            );

            h->d_round_keys = nullptr;
        }

        delete h;
    };

    /*
     * DEVICE_SBOX is constant GPU state used by both AES encryption
     * and the GPU-side key-expansion kernel.
     */
    cudaError_t rc =
        cudaMemcpyToSymbol(
            DEVICE_SBOX,
            HOST_SBOX,
            sizeof(HOST_SBOX)
        );

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    rc =
        cudaMalloc(
            &h->d_round_keys,
            AES_ROUND_KEY_BYTES
        );

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    rc =
        cudaMalloc(
            &h->d_h_pow2,
            sizeof(Block128) * 64
        );

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    /*
     * This allocation exists only during provisioning.
     *
     * In the target deployment the source pointer is inside the
     * TDX guest.  With H100 CC-On, NVIDIA transparently protects
     * this H2D transfer.
     */
    rc =
        cudaMalloc(
            &d_raw_key,
            16
        );

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_ALLOC;
    }

    rc =
        cudaMemcpy(
            d_raw_key,
            key,
            16,
            cudaMemcpyHostToDevice
        );

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    /*
     * Raw K_store has now crossed the TDX -> H100 provisioning
     * boundary.  Expand it inside H100 protected GPU memory.
     */
    expand_aes128_key_kernel<<<
        1,
        1
    >>>(
        d_raw_key,
        h->d_round_keys
    );

    if (
        cudaPeekAtLastError()
        != cudaSuccess
    ) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    /*
     * Erase the temporary raw K_store immediately after the key
     * expansion kernel in the same default-stream order.
     */
    zeroize_bytes_kernel<<<
        1,
        32
    >>>(
        d_raw_key,
        16
    );

    if (
        cudaPeekAtLastError()
        != cudaSuccess
    ) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    /*
     * GHASH's H and its power table are also derived inside the GPU.
     */
    init_hash_key_kernel<<<
        1,
        1
    >>>(
        h->d_round_keys,
        h->d_h_pow2
    );

    if (
        cudaPeekAtLastError()
        != cudaSuccess
    ) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    /*
     * key_create() is intentionally synchronous.
     *
     * When it returns successfully:
     *
     *   - GPU round keys are ready;
     *   - GHASH key state is ready;
     *   - the temporary GPU raw key has been overwritten.
     *
     * This lets the caller safely erase its TDX-side mutable
     * K_store buffer immediately after key_create returns.
     */
    rc =
        cudaDeviceSynchronize();

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    rc =
        cudaFree(
            d_raw_key
        );

    d_raw_key = nullptr;

    if (rc != cudaSuccess) {
        cleanup_failure();
        return LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    *out =
        reinterpret_cast<
            lmcache_gpu_aesgcm_key_t
        >(h);

    return LMCACHE_GPU_AESGCM_OK;
}


extern "C"
int lmcache_gpu_aes128gcm_key_destroy(
    lmcache_gpu_aesgcm_key_t key)
{
    if (key == nullptr) {
        return LMCACHE_GPU_AESGCM_OK;
    }

    auto* h =
        reinterpret_cast<
            KeyHandle*
        >(key);

    DeviceGuard guard(
        h->device
    );

    int status =
        LMCACHE_GPU_AESGCM_OK;

    /*
     * Wipe all key-derived GPU state before releasing allocations.
     */
    if (
        h->d_nodes_a != nullptr
        && h->node_capacity != 0
    ) {
        if (
            cudaMemset(
                h->d_nodes_a,
                0,
                h->node_capacity
                    * sizeof(GHashNode)
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (
        h->d_nodes_b != nullptr
        && h->node_capacity != 0
    ) {
        if (
            cudaMemset(
                h->d_nodes_b,
                0,
                h->node_capacity
                    * sizeof(GHashNode)
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (h->d_h_pow2 != nullptr) {
        if (
            cudaMemset(
                h->d_h_pow2,
                0,
                sizeof(Block128) * 64
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (h->d_round_keys != nullptr) {
        if (
            cudaMemset(
                h->d_round_keys,
                0,
                AES_ROUND_KEY_BYTES
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (
        cudaDeviceSynchronize()
        != cudaSuccess
    ) {
        status =
            LMCACHE_GPU_AESGCM_ERR_CUDA;
    }

    if (h->d_nodes_a != nullptr) {
        if (
            cudaFree(
                h->d_nodes_a
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (h->d_nodes_b != nullptr) {
        if (
            cudaFree(
                h->d_nodes_b
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (h->d_h_pow2 != nullptr) {
        if (
            cudaFree(
                h->d_h_pow2
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    if (h->d_round_keys != nullptr) {
        if (
            cudaFree(
                h->d_round_keys
            )
            != cudaSuccess
        ) {
            status =
                LMCACHE_GPU_AESGCM_ERR_CUDA;
        }
    }

    delete h;

    return status;
}


extern "C"
int lmcache_gpu_aes128gcm_reserve(
    lmcache_gpu_aesgcm_key_t key,
    size_t max_plaintext_len)
{
    if (key == nullptr) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    auto* h =
        reinterpret_cast<
            KeyHandle*
        >(key);

    DeviceGuard guard(
        h->device
    );

    return ensure_workspace(
        h,
        max_plaintext_len
    );
}


extern "C"
int lmcache_gpu_aes128gcm_seal_async(
    lmcache_gpu_aesgcm_key_t key,
    const void* src,
    size_t plaintext_len,
    void* dst,
    size_t dst_capacity,
    const uint8_t* iv,
    void* stream_ptr)
{
    if (
        key == nullptr
        || src == nullptr
        || dst == nullptr
        || iv == nullptr
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    const size_t frame_len =
        lmcache_gpu_aesgcm_frame_size(
            plaintext_len
        );

    if (
        dst_capacity
        < frame_len
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    const uint64_t blocks =
        (
            static_cast<uint64_t>(
                plaintext_len
            )
            + 15
        ) / 16;

    if (
        blocks
        > 0xfffffffeULL
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    auto* h =
        reinterpret_cast<
            KeyHandle*
        >(key);

    DeviceGuard guard(
        h->device
    );

    cudaStream_t stream =
        reinterpret_cast<
            cudaStream_t
        >(stream_ptr);

    const int erc =
        ensure_workspace(
            h,
            plaintext_len
        );

    if (
        erc
        != LMCACHE_GPU_AESGCM_OK
    ) {
        return erc;
    }

    Iv96 iv_arg;

    static_assert(
        sizeof(Iv96) == 12,
        "Iv96 must be exactly 12 bytes"
    );

    std::memcpy(
        &iv_arg,
        iv,
        12
    );

    auto* out =
        reinterpret_cast<
            uint8_t*
        >(dst);

    const auto* in =
        reinterpret_cast<
            const uint8_t*
        >(src);

    write_frame_header_kernel<<<
        1,
        1,
        0,
        stream
    >>>(
        out,
        iv_arg
    );

    if (blocks > 0) {

        constexpr int threads = 256;

        const int grid =
            static_cast<int>(
                (
                    blocks
                    + threads
                    - 1
                )
                / threads
            );

        ctr_crypt_kernel<<<
            grid,
            threads,
            0,
            stream
        >>>(
            in,
            out + HEADER_LEN,
            plaintext_len,
            out + 1,
            h->d_round_keys,
            nullptr
        );
    }

    GHashNode* root = nullptr;

    const int grc =
        enqueue_ghash(
            h,
            out + HEADER_LEN,
            plaintext_len,
            stream,
            &root
        );

    if (
        grc
        != LMCACHE_GPU_AESGCM_OK
    ) {
        return grc;
    }

    finalize_tag_kernel<<<
        1,
        1,
        0,
        stream
    >>>(
        out,
        plaintext_len,
        h->d_round_keys,
        root
    );

    return cuda_rc(
        cudaPeekAtLastError()
    );
}


extern "C"
int lmcache_gpu_aes128gcm_open_async(
    lmcache_gpu_aesgcm_key_t key,
    const void* src,
    size_t frame_len,
    void* dst,
    size_t plaintext_len,
    int* auth_ok,
    void* stream_ptr)
{
    if (
        key == nullptr
        || src == nullptr
        || dst == nullptr
        || auth_ok == nullptr
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    const size_t expected =
        lmcache_gpu_aesgcm_frame_size(
            plaintext_len
        );

    /*
     * Match stock LMCache behavior:
     * padded source buffers are allowed as long as the complete frame
     * is present.
     */
    if (
        frame_len
        < expected
    ) {
        return LMCACHE_GPU_AESGCM_ERR_ARGUMENT;
    }

    auto* h =
        reinterpret_cast<
            KeyHandle*
        >(key);

    DeviceGuard guard(
        h->device
    );

    cudaStream_t stream =
        reinterpret_cast<
            cudaStream_t
        >(stream_ptr);

    const int erc =
        ensure_workspace(
            h,
            plaintext_len
        );

    if (
        erc
        != LMCACHE_GPU_AESGCM_OK
    ) {
        return erc;
    }

    const auto* frame =
        reinterpret_cast<
            const uint8_t*
        >(src);

    auto* plain =
        reinterpret_cast<
            uint8_t*
        >(dst);

    GHashNode* root = nullptr;

    const int grc =
        enqueue_ghash(
            h,
            frame + HEADER_LEN,
            plaintext_len,
            stream,
            &root
        );

    if (
        grc
        != LMCACHE_GPU_AESGCM_OK
    ) {
        return grc;
    }

    verify_tag_kernel<<<
        1,
        1,
        0,
        stream
    >>>(
        frame,
        plaintext_len,
        h->d_round_keys,
        root,
        auth_ok
    );

    const uint64_t blocks =
        (
            static_cast<uint64_t>(
                plaintext_len
            )
            + 15
        ) / 16;

    if (blocks > 0) {

        constexpr int threads = 256;

        const int grid =
            static_cast<int>(
                (
                    blocks
                    + threads
                    - 1
                )
                / threads
            );

        /*
         * Same stream:
         *
         * verify_tag_kernel writes auth_ok first.
         * This kernel consumes auth_ok and either decrypts or zeroizes.
         */
        ctr_crypt_kernel<<<
            grid,
            threads,
            0,
            stream
        >>>(
            frame + HEADER_LEN,
            plain,
            plaintext_len,
            frame + 1,
            h->d_round_keys,
            auth_ok
        );
    }

    return cuda_rc(
        cudaPeekAtLastError()
    );
}


extern "C"
const char* lmcache_gpu_aesgcm_strerror(
    int rc)
{
    switch (rc) {

    case LMCACHE_GPU_AESGCM_OK:
        return "success";

    case LMCACHE_GPU_AESGCM_ERR_ARGUMENT:
        return "invalid argument";

    case LMCACHE_GPU_AESGCM_ERR_ALLOC:
        return "GPU allocation failure";

    case LMCACHE_GPU_AESGCM_ERR_CUDA:
        return "CUDA failure";

    case LMCACHE_GPU_AESGCM_ERR_INTERNAL:
        return "internal error";

    default:
        return "unknown error";
    }
}
