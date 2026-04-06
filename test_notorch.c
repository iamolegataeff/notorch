// test_notorch.c — tests for notorch
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        printf("  FAIL: %s — got %.6f, expected %.6f (line %d)\n", msg, _a, _b, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define PASS(name) do { printf("  PASS: %s\n", name); tests_passed++; } while(0)

// ── Tensor tests ─────────────────────────────────────────────────────────────

static void test_tensor_create(void) {
    nt_tensor* t = nt_tensor_new(10);
    ASSERT(t != NULL, "tensor alloc");
    ASSERT(t->len == 10, "tensor len");
    ASSERT(t->ndim == 1, "tensor ndim");
    ASSERT(t->shape[0] == 10, "tensor shape");
    ASSERT(t->data[0] == 0.0f, "tensor zeroed");
    nt_tensor_free(t);
    PASS("tensor_create");
}

static void test_tensor_2d(void) {
    nt_tensor* t = nt_tensor_new2d(3, 4);
    ASSERT(t != NULL, "2d alloc");
    ASSERT(t->len == 12, "2d len");
    ASSERT(t->ndim == 2, "2d ndim");
    ASSERT(t->shape[0] == 3 && t->shape[1] == 4, "2d shape");
    ASSERT(t->stride[0] == 4 && t->stride[1] == 1, "2d stride");
    nt_tensor_free(t);
    PASS("tensor_2d");
}

static void test_tensor_clone(void) {
    nt_tensor* a = nt_tensor_new(5);
    for (int i = 0; i < 5; i++) a->data[i] = (float)i;
    nt_tensor* b = nt_tensor_clone(a);
    ASSERT(b != NULL, "clone alloc");
    ASSERT(b->data[3] == 3.0f, "clone data");
    a->data[3] = 99.0f;
    ASSERT(b->data[3] == 3.0f, "clone independence");
    nt_tensor_free(a);
    nt_tensor_free(b);
    PASS("tensor_clone");
}

static void test_tensor_reshape(void) {
    nt_tensor* t = nt_tensor_new(12);
    int shape[] = {3, 4};
    ASSERT(nt_tensor_reshape(t, shape, 2) == 0, "reshape ok");
    ASSERT(t->ndim == 2, "reshape ndim");
    ASSERT(t->shape[0] == 3 && t->shape[1] == 4, "reshape shape");
    int bad[] = {5, 5};
    ASSERT(nt_tensor_reshape(t, bad, 2) != 0, "reshape mismatch");
    nt_tensor_free(t);
    PASS("tensor_reshape");
}

static void test_tensor_xavier(void) {
    nt_tensor* t = nt_tensor_new(1000);
    nt_seed(42);
    nt_tensor_xavier(t, 100, 100);
    float sum = 0;
    for (int i = 0; i < t->len; i++) sum += t->data[i] * t->data[i];
    float var = sum / t->len;
    // Xavier for fan_in=100, fan_out=100: scale = sqrt(6/200) ≈ 0.173
    // Uniform[-s,s] variance = s²/3 ≈ 0.01
    ASSERT(var > 0.005f && var < 0.05f, "xavier variance");
    nt_tensor_free(t);
    PASS("tensor_xavier");
}

static void test_tensor_refcount(void) {
    nt_tensor* t = nt_tensor_new(5);
    ASSERT(t->refcount == 1, "initial refcount");
    nt_tensor_ref(t);
    ASSERT(t->refcount == 2, "ref refcount");
    nt_tensor_free(t);
    ASSERT(t->refcount == 1, "free decrements");
    nt_tensor_free(t);
    PASS("tensor_refcount");
}

// ── Tape + forward op tests ─────────────────────────────────────────────────

static void test_tape_basic(void) {
    nt_tape_start();
    ASSERT(nt_tape_is_active(), "tape active");

    nt_tensor* w = nt_tensor_new2d(4, 3);
    nt_tensor_xavier(w, 3, 4);
    int w_idx = nt_tape_param(w);
    ASSERT(w_idx >= 0, "param registered");

    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_linear(w_idx, x_idx, -1);
    ASSERT(y_idx >= 0, "linear ok");

    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT(ey->output->len == 4, "linear output dim");

    nt_tape_clear();
    nt_tensor_free(w);
    nt_tensor_free(x);
    PASS("tape_basic");
}

static void test_forward_backward_linear(void) {
    nt_seed(123);
    nt_tape_start();

    // W: 2x3, x: 3
    nt_tensor* W = nt_tensor_new2d(2, 3);
    W->data[0] = 1; W->data[1] = 0; W->data[2] = 0;
    W->data[3] = 0; W->data[4] = 1; W->data[5] = 0;
    int w_idx = nt_tape_param(W);

    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_linear(w_idx, x_idx, -1);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT_CLOSE(ey->output->data[0], 1.0f, 1e-5f, "linear W@x [0]");
    ASSERT_CLOSE(ey->output->data[1], 2.0f, 1e-5f, "linear W@x [1]");

    // CE loss against target=0
    int loss_idx = nt_cross_entropy(y_idx, 0);
    ASSERT(loss_idx >= 0, "cross_entropy ok");

    nt_tape_backward(loss_idx);

    // Check W has gradient
    nt_tape_entry* ew = &nt_tape_get()->entries[w_idx];
    ASSERT(ew->grad != NULL, "W grad exists");
    ASSERT(ew->grad->len == 6, "W grad len");

    // Gradient should be non-zero
    float gnorm = 0;
    for (int i = 0; i < 6; i++) gnorm += ew->grad->data[i] * ew->grad->data[i];
    ASSERT(gnorm > 1e-10f, "W grad non-zero");

    nt_tape_clear();
    nt_tensor_free(W);
    nt_tensor_free(x);
    PASS("forward_backward_linear");
}

static void test_adam_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    float orig0 = W->data[0];
    int w_idx = nt_tape_param(W);

    nt_tensor* x = nt_tensor_new(4);
    x->data[0] = 1; x->data[1] = 0; x->data[2] = 0; x->data[3] = 0;
    nt_tape_record(x, NT_OP_NONE, -1, -1, 0);

    int loss_idx = nt_cross_entropy(w_idx, 2);  // treat W as logits, target=2
    nt_tape_backward(loss_idx);

    nt_tape_adam_step(0.01f);

    // W should have changed
    ASSERT(fabsf(W->data[0] - orig0) > 1e-6f, "adam changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    nt_tensor_free(x);
    PASS("adam_step");
}

static void test_adamw_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 2);
    nt_tape_backward(loss_idx);

    float before = W->data[0];
    nt_tape_adamw_step(0.01f, 0.1f, 0.9f, 0.999f);
    ASSERT(fabsf(W->data[0] - before) > 1e-6f, "adamw changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("adamw_step");
}

static void test_chuck_step(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    W->data[0] = 1; W->data[1] = 2; W->data[2] = 3; W->data[3] = 4;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 2);
    nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
    float loss_val = el->output->data[0];

    nt_tape_backward(loss_idx);

    float before = W->data[0];
    nt_tape_chuck_step(0.01f, loss_val);
    ASSERT(fabsf(W->data[0] - before) > 1e-6f, "chuck changed W");

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("chuck_step");
}

static void test_grad_clip(void) {
    nt_tape_start();

    nt_tensor* W = nt_tensor_new(4);
    for (int i = 0; i < 4; i++) W->data[i] = (float)(i + 1) * 10.0f;
    int w_idx = nt_tape_param(W);

    int loss_idx = nt_cross_entropy(w_idx, 0);
    nt_tape_backward(loss_idx);

    float norm_before = nt_tape_clip_grads(1000.0f); // large max, no clipping
    ASSERT(norm_before > 0, "grad norm > 0");

    // Now actually clip
    // Re-do backward
    nt_tape_clear();
    nt_tape_start();
    w_idx = nt_tape_param(W);
    loss_idx = nt_cross_entropy(w_idx, 0);
    nt_tape_backward(loss_idx);

    float norm = nt_tape_clip_grads(0.1f);
    // After clipping, norm should be ~0.1
    float norm_after = 0;
    nt_tape_entry* ew = &nt_tape_get()->entries[w_idx];
    for (int i = 0; i < ew->grad->len; i++)
        norm_after += ew->grad->data[i] * ew->grad->data[i];
    norm_after = sqrtf(norm_after);
    if (norm > 0.1f) {
        ASSERT_CLOSE(norm_after, 0.1f, 0.01f, "clipped norm");
    }

    nt_tape_clear();
    nt_tensor_free(W);
    PASS("grad_clip");
}

static void test_silu(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = -1; x->data[1] = 0; x->data[2] = 1;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_silu(x_idx);

    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    // silu(0) = 0
    ASSERT_CLOSE(ey->output->data[1], 0.0f, 1e-5f, "silu(0)=0");
    // silu(1) = 1 * sigmoid(1) ≈ 0.731
    ASSERT_CLOSE(ey->output->data[2], 0.7311f, 0.01f, "silu(1)≈0.73");
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.269
    ASSERT_CLOSE(ey->output->data[0], -0.2689f, 0.01f, "silu(-1)≈-0.27");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("silu");
}

static void test_softmax(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_softmax(x_idx);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];

    float sum = 0;
    for (int i = 0; i < 3; i++) sum += ey->output->data[i];
    ASSERT_CLOSE(sum, 1.0f, 1e-5f, "softmax sums to 1");
    ASSERT(ey->output->data[2] > ey->output->data[1], "softmax ordering");
    ASSERT(ey->output->data[1] > ey->output->data[0], "softmax ordering 2");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("softmax");
}

static void test_rmsnorm(void) {
    nt_tape_start();
    nt_tensor* x = nt_tensor_new(4);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3; x->data[3] = 4;
    int x_idx = nt_tape_record(x, NT_OP_NONE, -1, -1, 0);
    int y_idx = nt_rmsnorm(x_idx, -1);
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];

    // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    float rms = sqrtf(7.5f + 1e-6f);
    ASSERT_CLOSE(ey->output->data[0], 1.0f / rms, 1e-4f, "rmsnorm [0]");
    ASSERT_CLOSE(ey->output->data[3], 4.0f / rms, 1e-4f, "rmsnorm [3]");

    nt_tape_clear();
    nt_tensor_free(x);
    PASS("rmsnorm");
}

static void test_causal_attention(void) {
    nt_tape_start();
    int T = 3, D = 4;

    nt_tensor* q = nt_tensor_new(T * D);
    nt_tensor* k = nt_tensor_new(T * D);
    nt_tensor* v = nt_tensor_new(T * D);
    nt_seed(42);
    nt_tensor_rand(q, 0.5f);
    nt_tensor_rand(k, 0.5f);
    nt_tensor_rand(v, 0.5f);

    int q_idx = nt_tape_record(q, NT_OP_NONE, -1, -1, 0);
    int k_idx = nt_tape_record(k, NT_OP_NONE, -1, -1, 0);
    int v_idx = nt_tape_record(v, NT_OP_NONE, -1, -1, 0);

    int out_idx = nt_causal_attention(q_idx, k_idx, v_idx, T, D);
    ASSERT(out_idx >= 0, "attention ok");
    nt_tape_entry* eo = &nt_tape_get()->entries[out_idx];
    ASSERT(eo->output->len == T * D, "attention output size");

    // First position should be exactly V[0] (only attends to self)
    for (int d = 0; d < D; d++)
        ASSERT_CLOSE(eo->output->data[d], v->data[d], 1e-4f, "attn pos0 = V[0]");

    nt_tape_clear();
    nt_tensor_free(q); nt_tensor_free(k); nt_tensor_free(v);
    PASS("causal_attention");
}

static void test_mh_causal_attention(void) {
    nt_tape_start();
    int T = 4, D = 8, head_dim = 4; // 2 heads
    nt_tensor* q = nt_tensor_new(T * D);
    nt_tensor* k = nt_tensor_new(T * D);
    nt_tensor* v = nt_tensor_new(T * D);
    nt_seed(7);
    nt_tensor_rand(q, 0.3f);
    nt_tensor_rand(k, 0.3f);
    nt_tensor_rand(v, 0.3f);

    int q_idx = nt_tape_record(q, NT_OP_NONE, -1, -1, 0);
    int k_idx = nt_tape_record(k, NT_OP_NONE, -1, -1, 0);
    int v_idx = nt_tape_record(v, NT_OP_NONE, -1, -1, 0);

    int out_idx = nt_mh_causal_attention(q_idx, k_idx, v_idx, T, head_dim);
    ASSERT(out_idx >= 0, "mh_attention ok");
    nt_tape_entry* eo = &nt_tape_get()->entries[out_idx];
    ASSERT(eo->output->len == T * D, "mh_attention output size");

    nt_tape_clear();
    nt_tensor_free(q); nt_tensor_free(k); nt_tensor_free(v);
    PASS("mh_causal_attention");
}

static void test_seq_cross_entropy(void) {
    nt_tape_start();
    int T = 3, V = 5;

    nt_tensor* logits = nt_tensor_new(T * V);
    nt_seed(99);
    nt_tensor_rand(logits, 1.0f);

    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 0; targets->data[1] = 2; targets->data[2] = 4;

    int l_idx = nt_tape_record(logits, NT_OP_NONE, -1, -1, 0);
    int t_idx = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

    int loss_idx = nt_seq_cross_entropy(l_idx, t_idx, T, V);
    ASSERT(loss_idx >= 0, "seq_ce ok");

    nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
    float loss = el->output->data[0];
    // Loss should be positive and reasonable (random logits → ~log(V) ≈ 1.6)
    ASSERT(loss > 0.5f && loss < 5.0f, "seq_ce loss range");

    // Test backward
    nt_tape_backward(loss_idx);
    nt_tape_entry* elg = &nt_tape_get()->entries[l_idx];
    ASSERT(elg->grad != NULL, "seq_ce grad exists");

    nt_tape_clear();
    nt_tensor_free(logits); nt_tensor_free(targets);
    PASS("seq_cross_entropy");
}

static void test_seq_linear(void) {
    nt_tape_start();
    int T = 3, in_d = 4, out_d = 2;

    nt_tensor* W = nt_tensor_new2d(out_d, in_d);
    // Identity-ish: first 2 dims
    W->data[0] = 1; W->data[1] = 0; W->data[2] = 0; W->data[3] = 0;
    W->data[4] = 0; W->data[5] = 1; W->data[6] = 0; W->data[7] = 0;
    int w_idx = nt_tape_param(W);

    nt_tensor* X = nt_tensor_new(T * in_d);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < in_d; d++)
            X->data[t * in_d + d] = (float)(t * in_d + d);
    int x_idx = nt_tape_record(X, NT_OP_NONE, -1, -1, 0);

    int y_idx = nt_seq_linear(w_idx, x_idx, T);
    ASSERT(y_idx >= 0, "seq_linear ok");
    nt_tape_entry* ey = &nt_tape_get()->entries[y_idx];
    ASSERT(ey->output->len == T * out_d, "seq_linear output size");

    // Y[0] = W @ X[0] = [X[0,0], X[0,1]] = [0, 1]
    ASSERT_CLOSE(ey->output->data[0], 0.0f, 1e-4f, "seq_linear [0,0]");
    ASSERT_CLOSE(ey->output->data[1], 1.0f, 1e-4f, "seq_linear [0,1]");

    nt_tape_clear();
    nt_tensor_free(W); nt_tensor_free(X);
    PASS("seq_linear");
}

static void test_save_load(void) {
    nt_tensor* t1 = nt_tensor_new2d(3, 4);
    nt_tensor* t2 = nt_tensor_new(5);
    for (int i = 0; i < 12; i++) t1->data[i] = (float)i;
    for (int i = 0; i < 5; i++) t2->data[i] = (float)(i * 10);

    nt_tensor* params[] = {t1, t2};
    int rc = nt_save("/tmp/notorch_test.bin", params, 2);
    ASSERT(rc == 0, "save ok");

    int n_loaded = 0;
    nt_tensor** loaded = nt_load("/tmp/notorch_test.bin", &n_loaded);
    ASSERT(loaded != NULL, "load ok");
    ASSERT(n_loaded == 2, "loaded count");
    ASSERT(loaded[0]->ndim == 2, "loaded[0] ndim");
    ASSERT(loaded[0]->shape[0] == 3 && loaded[0]->shape[1] == 4, "loaded[0] shape");
    ASSERT_CLOSE(loaded[0]->data[5], 5.0f, 1e-5f, "loaded[0] data");
    ASSERT(loaded[1]->len == 5, "loaded[1] len");
    ASSERT_CLOSE(loaded[1]->data[3], 30.0f, 1e-5f, "loaded[1] data");

    for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
    free(loaded);
    nt_tensor_free(t1); nt_tensor_free(t2);
    PASS("save_load");
}

static void test_hebbian(void) {
    int in_d = 4, out_d = 3, rank = 2;
    float A[4 * 2] = {0};
    float B[2 * 3] = {0};
    float x[4] = {1, 0, 0, 0};
    float dy[3] = {1, 0, 0};

    nt_hebbian_step(A, B, out_d, in_d, rank, x, dy, 1.0f, 0.01f, 0.999f);

    // A should have been updated: A[0,0] and A[0,1] should be non-zero... actually
    // with B=0 initially, proj = B^T @ dy = 0, so A stays 0. But proj2 = A^T @ x = 0 too.
    // So first step with zero matrices = no update. Need non-zero init.
    B[0] = 1.0f; // B[0,0] = 1
    nt_hebbian_step(A, B, out_d, in_d, rank, x, dy, 1.0f, 0.01f, 0.999f);
    // Now proj = B^T @ dy = [1, 0], so A[0,0] += 0.01 * 1 * 1 * 1 = 0.01
    ASSERT(fabsf(A[0]) > 1e-5f, "hebbian updated A");

    PASS("hebbian");
}

static void test_training_loop(void) {
    // Mini training loop: learn to predict target from embedding
    nt_seed(42);

    int vocab = 4, dim = 8;
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor* wout = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wout, dim, vocab);

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 50; step++) {
        nt_tape_start();
        int wte_idx = nt_tape_param(wte);
        nt_tape_no_decay(wte_idx);
        int wout_idx = nt_tape_param(wout);

        // Input token = 1, target = 2
        int h_idx = nt_embedding(wte_idx, 1);
        int logits_idx = nt_linear(wout_idx, h_idx, -1);
        int loss_idx = nt_cross_entropy(logits_idx, 2);

        nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
        float loss = el->output->data[0];
        if (step == 0) initial_loss = loss;
        if (step == 49) final_loss = loss;

        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.05f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "loss decreased");
    ASSERT(final_loss < 1.0f, "loss < 1.0");
    printf("    training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte);
    nt_tensor_free(wout);
    PASS("training_loop");
}

static void test_seq_training_loop(void) {
    // Sequence-level training: embed → seq_linear → seq_cross_entropy
    nt_seed(77);

    int vocab = 8, dim = 16, T = 4;
    nt_tensor* wte = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wte, vocab, dim);
    nt_tensor* wpe = nt_tensor_new2d(T, dim);
    nt_tensor_xavier(wpe, T, dim);
    nt_tensor* wout = nt_tensor_new2d(vocab, dim);
    nt_tensor_xavier(wout, dim, vocab);

    // Tokens: [1, 3, 5, 2], targets: [3, 5, 2, 7]
    nt_tensor* tokens = nt_tensor_new(T);
    tokens->data[0] = 1; tokens->data[1] = 3; tokens->data[2] = 5; tokens->data[3] = 2;
    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 3; targets->data[1] = 5; targets->data[2] = 2; targets->data[3] = 7;

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 100; step++) {
        nt_tape_start();
        int wte_idx = nt_tape_param(wte);
        nt_tape_no_decay(wte_idx);
        int wpe_idx = nt_tape_param(wpe);
        nt_tape_no_decay(wpe_idx);
        int wout_idx = nt_tape_param(wout);
        int tok_idx = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_idx = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        int h_idx = nt_seq_embedding(wte_idx, wpe_idx, tok_idx, T, dim);
        int logits_idx = nt_seq_linear(wout_idx, h_idx, T);
        int loss_idx = nt_seq_cross_entropy(logits_idx, tgt_idx, T, vocab);

        nt_tape_entry* el = &nt_tape_get()->entries[loss_idx];
        float loss = el->output->data[0];
        if (step == 0) initial_loss = loss;
        if (step == 99) final_loss = loss;

        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.01f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "seq loss decreased");
    printf("    seq training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte); nt_tensor_free(wpe); nt_tensor_free(wout);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    PASS("seq_training_loop");
}

static void test_attention_training(void) {
    // Train a tiny attention model
    nt_seed(42);
    int T = 3, D = 8, vocab = 6;

    nt_tensor* wte = nt_tensor_new2d(vocab, D);
    nt_tensor_xavier(wte, vocab, D);
    nt_tensor* wpe = nt_tensor_new2d(T, D);
    nt_tensor_xavier(wpe, T, D);
    nt_tensor* Wq = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wq, D, D);
    nt_tensor* Wk = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wk, D, D);
    nt_tensor* Wv = nt_tensor_new2d(D, D);
    nt_tensor_xavier(Wv, D, D);
    nt_tensor* Wout = nt_tensor_new2d(vocab, D);
    nt_tensor_xavier(Wout, D, vocab);

    nt_tensor* tokens = nt_tensor_new(T);
    tokens->data[0] = 1; tokens->data[1] = 3; tokens->data[2] = 5;
    nt_tensor* targets = nt_tensor_new(T);
    targets->data[0] = 3; targets->data[1] = 5; targets->data[2] = 0;

    float initial_loss = 0, final_loss = 0;

    for (int step = 0; step < 80; step++) {
        nt_tape_start();
        int wte_i = nt_tape_param(wte); nt_tape_no_decay(wte_i);
        int wpe_i = nt_tape_param(wpe); nt_tape_no_decay(wpe_i);
        int wq_i = nt_tape_param(Wq);
        int wk_i = nt_tape_param(Wk);
        int wv_i = nt_tape_param(Wv);
        int wo_i = nt_tape_param(Wout);
        int tok_i = nt_tape_record(tokens, NT_OP_NONE, -1, -1, 0);
        int tgt_i = nt_tape_record(targets, NT_OP_NONE, -1, -1, 0);

        int h = nt_seq_embedding(wte_i, wpe_i, tok_i, T, D);
        int q = nt_seq_linear(wq_i, h, T);
        int k = nt_seq_linear(wk_i, h, T);
        int v = nt_seq_linear(wv_i, h, T);
        int attn = nt_causal_attention(q, k, v, T, D);
        int logits = nt_seq_linear(wo_i, attn, T);
        int loss = nt_seq_cross_entropy(logits, tgt_i, T, vocab);

        float lv = nt_tape_get()->entries[loss].output->data[0];
        if (step == 0) initial_loss = lv;
        if (step == 79) final_loss = lv;

        nt_tape_backward(loss);
        nt_tape_clip_grads(1.0f);
        nt_tape_adam_step(0.005f);
        nt_tape_clear();
    }

    ASSERT(final_loss < initial_loss, "attn loss decreased");
    printf("    attn training: loss %.4f → %.4f\n", initial_loss, final_loss);

    nt_tensor_free(wte); nt_tensor_free(wpe);
    nt_tensor_free(Wq); nt_tensor_free(Wk); nt_tensor_free(Wv); nt_tensor_free(Wout);
    nt_tensor_free(tokens); nt_tensor_free(targets);
    PASS("attention_training");
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    printf("notorch tests\n");
    printf("═══════════════════════════════════════════\n");

    printf("\n[Tensor]\n");
    test_tensor_create();
    test_tensor_2d();
    test_tensor_clone();
    test_tensor_reshape();
    test_tensor_xavier();
    test_tensor_refcount();

    printf("\n[Ops]\n");
    test_silu();
    test_softmax();
    test_rmsnorm();

    printf("\n[Tape + Forward/Backward]\n");
    test_tape_basic();
    test_forward_backward_linear();
    test_causal_attention();
    test_mh_causal_attention();
    test_seq_cross_entropy();
    test_seq_linear();

    printf("\n[Optimizers]\n");
    test_adam_step();
    test_adamw_step();
    test_chuck_step();
    test_grad_clip();

    printf("\n[Hebbian]\n");
    test_hebbian();

    printf("\n[Save/Load]\n");
    test_save_load();

    printf("\n[Training]\n");
    test_training_loop();
    test_seq_training_loop();
    test_attention_training();

    printf("\n═══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
