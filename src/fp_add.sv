`default_nettype none

// ---------------------------------------------------------------------------
// fp_add -- y = a + b for floats with an 8-bit exponent (bias 127) and M
// mantissa bits (M = 7: BF16, M = 23: FP32). Combinational, single rounding.
//
// Number handling (shared with bf16_mul, mirrored by test/fp_model.py):
//   * subnormals flush to zero, on input and output (FTZ/DAZ)
//   * round to nearest, ties to even (guard / round / sticky)
//   * exact cancellation -> +0; (+-0) + (+-0) -> -0 only if both are -0
//   * x + 0 = x; inf + finite = inf; inf + -inf = NaN;
//     any NaN in -> canonical quiet NaN out
//   * overflow -> signed inf; underflow -> signed 0
// ---------------------------------------------------------------------------
module fp_add #(
    parameter int M = 23
) (
    input  logic [M+8:0] a,
    input  logic [M+8:0] b,
    output logic [M+8:0] y
);

    localparam int F = M + 1;   // significand incl. hidden one
    localparam int W = F + 3;   // + guard, round, sticky

    logic         sa, sb;
    logic [7:0]   ea, eb;
    logic [M-1:0] ma, mb;

    assign {sa, ea, ma} = a;
    assign {sb, eb, mb} = b;

    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    assign a_zero = (ea == 8'h00);
    assign b_zero = (eb == 8'h00);
    assign a_inf  = (ea == 8'hFF) && (ma == '0);
    assign b_inf  = (eb == 8'hFF) && (mb == '0);
    assign a_nan  = (ea == 8'hFF) && (ma != '0);
    assign b_nan  = (eb == 8'hFF) && (mb != '0);

    // ---- order by magnitude: x = larger, z = smaller ----------------------
    logic         a_big;
    logic         sx, sz;
    logic [7:0]   ex, ez;
    logic [M-1:0] mx, mz;

    assign a_big = {ea, ma} >= {eb, mb};
    assign {sx, ex, mx} = a_big ? a : b;
    assign {sz, ez, mz} = a_big ? b : a;

    // ---- align z to x, collecting shifted-out bits into sticky ------------
    logic [7:0]   d;
    logic [W-1:0] sig_x, sig_z, sig_z_sh, sig_z_al, lost_mask;
    logic         sticky;

    assign d     = ex - ez;
    assign sig_x = {1'b1, mx, 3'b000};
    assign sig_z = {1'b1, mz, 3'b000};

    always_comb begin
        if (d >= 8'(W)) begin
            sig_z_sh  = '0;
            lost_mask = '1;
        end else begin
            sig_z_sh  = sig_z >> d;
            lost_mask = ~({W{1'b1}} << d);
        end
        sticky   = |(sig_z & lost_mask);
        sig_z_al = {sig_z_sh[W-1:1], sig_z_sh[0] | sticky};
    end

    // ---- add / subtract magnitudes ------------------------------------------
    logic         eff_sub;
    logic [W:0]   sum;

    assign eff_sub = sx ^ sz;
    assign sum = eff_sub ? ({1'b0, sig_x} - {1'b0, sig_z_al})
                         : ({1'b0, sig_x} + {1'b0, sig_z_al});

    // ---- normalise ------------------------------------------------------------
    logic [W-1:0]      norm;
    logic signed [9:0] exp_n;
    logic [4:0]        lz;

    // leading-zero count of sum[W-1:0] (W when zero); the last match wins,
    // and scanning upward makes that the most significant one
    always_comb begin
        lz = 5'(W);
        for (int i = 0; i < W; i++)
            if (sum[i]) lz = 5'(W-1-i);
    end

    always_comb begin
        if (!eff_sub && sum[W]) begin
            norm  = {sum[W:2], sum[1] | sum[0]};
            exp_n = $signed({2'b00, ex}) + 10'sd1;
        end else begin
            norm  = sum[W-1:0] << lz;
            exp_n = $signed({2'b00, ex}) - $signed({5'b0, lz});
        end
    end

    // ---- round to nearest even ------------------------------------------------
    logic [M-1:0]      mant_r;
    logic              g, rs, ru, carry;
    logic signed [9:0] exp_r;

    assign g  = norm[2];
    assign rs = norm[1] | norm[0];
    assign ru = g & (rs | norm[3]);
    assign {carry, mant_r} = {1'b0, norm[W-2:3]} + {{M{1'b0}}, ru};
    assign exp_r = exp_n + (carry ? 10'sd1 : 10'sd0);

    localparam logic [M+8:0] QNAN = {1'b0, 8'hFF, 1'b1, {(M-1){1'b0}}};

    wire _unused = &{norm[W-1], 1'b0};   // hidden one, implied by the format

    always_comb begin
        if (a_nan || b_nan || (a_inf && b_inf && (sa != sb)))
            y = QNAN;
        else if (a_inf)
            y = {sa, 8'hFF, {M{1'b0}}};
        else if (b_inf)
            y = {sb, 8'hFF, {M{1'b0}}};
        else if (a_zero && b_zero)
            y = {sa & sb, 8'h00, {M{1'b0}}};
        else if (a_zero)
            y = b;
        else if (b_zero)
            y = a;
        else if (sum == '0)
            y = '0;                                   // exact cancellation: +0
        else if (exp_r >= 10'sd255)
            y = {sx, 8'hFF, {M{1'b0}}};
        else if (exp_r <= 10'sd0)
            y = {sx, 8'h00, {M{1'b0}}};               // flush to zero
        else
            y = {sx, exp_r[7:0], mant_r};
    end

endmodule

`default_nettype wire
