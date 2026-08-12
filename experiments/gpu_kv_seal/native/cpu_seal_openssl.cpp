#include <openssl/evp.h>
#include <openssl/crypto.h>
#include <openssl/rand.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t VERSION_LEN = 1;
constexpr std::size_t IV_LEN = 12;
constexpr std::size_t TAG_LEN = 16;
constexpr std::size_t HEADER_LEN = VERSION_LEN + IV_LEN;
constexpr std::size_t FRAME_OVERHEAD = VERSION_LEN + IV_LEN + TAG_LEN;

struct ThreadCtx {
    EVP_CIPHER_CTX* ctx;

    ThreadCtx() : ctx(EVP_CIPHER_CTX_new()) {}

    ~ThreadCtx() {
        if (ctx != nullptr) {
            EVP_CIPHER_CTX_free(ctx);
        }
    }
};

thread_local ThreadCtx tls_ctx;

int encrypt_update_all(
    EVP_CIPHER_CTX* ctx,
    const std::uint8_t* src,
    std::size_t len,
    std::uint8_t* dst)
{
    std::size_t done = 0;

    while (done < len) {
        const std::size_t remain = len - done;
        const int chunk = static_cast<int>(
            std::min<std::size_t>(
                remain,
                static_cast<std::size_t>(INT_MAX)));

        int produced = 0;

        if (EVP_EncryptUpdate(
                ctx,
                dst + done,
                &produced,
                src + done,
                chunk) != 1) {
            return -20;
        }

        if (produced != chunk) {
            return -21;
        }

        done += static_cast<std::size_t>(chunk);
    }

    return 0;
}

int decrypt_update_all(
    EVP_CIPHER_CTX* ctx,
    const std::uint8_t* src,
    std::size_t len,
    std::uint8_t* dst)
{
    std::size_t done = 0;

    while (done < len) {
        const std::size_t remain = len - done;
        const int chunk = static_cast<int>(
            std::min<std::size_t>(
                remain,
                static_cast<std::size_t>(INT_MAX)));

        int produced = 0;

        if (EVP_DecryptUpdate(
                ctx,
                dst + done,
                &produced,
                src + done,
                chunk) != 1) {
            return -30;
        }

        if (produced != chunk) {
            return -31;
        }

        done += static_cast<std::size_t>(chunk);
    }

    return 0;
}

}  // namespace


extern "C"
int lmcache_aes128gcm_seal(
    const std::uint8_t* key,
    const std::uint8_t* src,
    std::size_t src_len,
    std::uint8_t* dst,
    std::size_t dst_len)
{
    if (key == nullptr || src == nullptr || dst == nullptr) {
        return -1;
    }

    if (dst_len < src_len + FRAME_OVERHEAD) {
        return -2;
    }

    EVP_CIPHER_CTX* ctx = tls_ctx.ctx;

    if (ctx == nullptr) {
        return -3;
    }

    if (EVP_CIPHER_CTX_reset(ctx) != 1) {
        return -4;
    }

    // Same LMCache frame:
    //
    // [1-byte version]
    // [12-byte IV]
    // [ciphertext]
    // [16-byte GCM tag]

    dst[0] = 1;

    std::uint8_t* iv = dst + VERSION_LEN;
    std::uint8_t* ciphertext = dst + HEADER_LEN;
    std::uint8_t* tag = ciphertext + src_len;

    if (RAND_bytes(iv, IV_LEN) != 1) {
        return -5;
    }

    if (EVP_EncryptInit_ex(
            ctx,
            EVP_aes_128_gcm(),
            nullptr,
            nullptr,
            nullptr) != 1) {
        return -6;
    }

    if (EVP_CIPHER_CTX_ctrl(
            ctx,
            EVP_CTRL_GCM_SET_IVLEN,
            IV_LEN,
            nullptr) != 1) {
        return -7;
    }

    if (EVP_EncryptInit_ex(
            ctx,
            nullptr,
            nullptr,
            key,
            iv) != 1) {
        return -8;
    }

    const int rc =
        encrypt_update_all(
            ctx,
            src,
            src_len,
            ciphertext);

    if (rc != 0) {
        return rc;
    }

    int final_len = 0;

    if (EVP_EncryptFinal_ex(
            ctx,
            ciphertext + src_len,
            &final_len) != 1) {
        return -9;
    }

    if (final_len != 0) {
        return -10;
    }

    if (EVP_CIPHER_CTX_ctrl(
            ctx,
            EVP_CTRL_GCM_GET_TAG,
            TAG_LEN,
            tag) != 1) {
        return -11;
    }

    return 0;
}


extern "C"
int lmcache_aes128gcm_open(
    const std::uint8_t* key,
    const std::uint8_t* frame,
    std::size_t frame_len,
    std::uint8_t* dst,
    std::size_t dst_len)
{
    if (key == nullptr || frame == nullptr || dst == nullptr) {
        return -1;
    }

    if (frame_len < FRAME_OVERHEAD) {
        return -2;
    }

    if (frame[0] != 1) {
        return -3;
    }

    const std::size_t plaintext_len =
        frame_len - FRAME_OVERHEAD;

    if (dst_len < plaintext_len) {
        return -4;
    }

    const std::uint8_t* iv =
        frame + VERSION_LEN;

    const std::uint8_t* ciphertext =
        frame + HEADER_LEN;

    const std::uint8_t* tag =
        ciphertext + plaintext_len;

    EVP_CIPHER_CTX* ctx = tls_ctx.ctx;

    if (ctx == nullptr) {
        return -5;
    }

    if (EVP_CIPHER_CTX_reset(ctx) != 1) {
        return -6;
    }

    if (EVP_DecryptInit_ex(
            ctx,
            EVP_aes_128_gcm(),
            nullptr,
            nullptr,
            nullptr) != 1) {
        return -7;
    }

    if (EVP_CIPHER_CTX_ctrl(
            ctx,
            EVP_CTRL_GCM_SET_IVLEN,
            IV_LEN,
            nullptr) != 1) {
        return -8;
    }

    if (EVP_DecryptInit_ex(
            ctx,
            nullptr,
            nullptr,
            key,
            iv) != 1) {
        return -9;
    }

    // Install the expected authentication tag before exposing any
    // provisional plaintext through EVP_DecryptUpdate.
    if (EVP_CIPHER_CTX_ctrl(
            ctx,
            EVP_CTRL_GCM_SET_TAG,
            TAG_LEN,
            const_cast<std::uint8_t*>(tag)) != 1) {
        return -10;
    }

    const int rc =
        decrypt_update_all(
            ctx,
            ciphertext,
            plaintext_len,
            dst);

    if (rc != 0) {
        // A failed update may have produced partial provisional plaintext.
        OPENSSL_cleanse(dst, plaintext_len);
        return rc;
    }

    int final_len = 0;

    // Authentication succeeds only if this returns 1.
    if (EVP_DecryptFinal_ex(
            ctx,
            dst + plaintext_len,
            &final_len) != 1) {

        // EVP_DecryptUpdate may already have written provisional
        // plaintext into dst. Authentication has failed, so that
        // plaintext must never be consumed or published.
        OPENSSL_cleanse(dst, plaintext_len);

        return -11;
    }

    return 0;
}
