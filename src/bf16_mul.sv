`default_nettype none

// ---------------------------------------------------------------------------
// bf16_mul -- BF16 x BF16 -> float with an 8-bit exponent and OUT_M mantissa
// bits (OUT_M = 7: BF16 result, OUT_M = 23: FP32 result). Combinational.
//
// Number handling (shared with fp_add, mirrored by test/fp_model.py):
//   * subnormals flush to zero, on input and output (FTZ/DAZ)
//   * round to nearest, ties to even (only needed when OUT_M < 15; a BF16
//     product has 16 significant bits, so the FP32 result is exact)
//   * x * 0 = signed 0; inf * finite = signed inf; inf * 0 = NaN;
//     any NaN in -> canonical quiet NaN out (sign 0, MSB of mantissa set)
//   * exponent overflow -> signed inf; underflow -> signed 0
// ---------------------------------------------------------------------------
module bf16_mul #(
    parameter int OUT_M = 23
) (
    input  logic [15:0]      a,
    input  logic [15:0]      b,
    output logic [OUT_M+8:0] y
);

    logic       sa, sb, sy;
    logic [7:0] ea, eb;
    logic [6:0] ma, mb;

    assign {sa, ea, ma} = a;
    assign {sb, eb, mb} = b;
    assign sy = sa ^ sb;

    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    assign a_zero = (ea == 8'h00);
    assign b_zero = (eb == 8'h00);
    assign a_inf  = (ea == 8'hFF) && (ma == '0);
    assign b_inf  = (eb == 8'hFF) && (mb == '0);
    assign a_nan  = (ea == 8'hFF) && (ma != '0);
    assign b_nan  = (eb == 8'hFF) && (mb != '0);

    // 1.7 x 1.7 -> 2.14 significand product
    logic [15:0] prod;
    assign prod = {1'b1, ma} * {1'b1, mb};

    // normalise to 1.15: frac = the 15 bits after the leading one
    logic        hi;
    logic [14:0] frac;
    assign hi   = prod[15];
    assign frac = hi ? prod[14:0] : {prod[13:0], 1'b0};

    logic signed [10:0] exp_pre;
    assign exp_pre = $signed({3'b000, ea}) + $signed({3'b000, eb}) - 11'sd127
                   + (hi ? 11'sd1 : 11'sd0);

    // ---- fit the fraction to OUT_M bits ----------------------------------
    logic [OUT_M-1:0]   mant;
    logic signed [10:0] exp_r;

    if (OUT_M >= 15) begin : g_exact
        if (OUT_M > 15) begin : g_pad
            assign mant = {frac, {(OUT_M-15){1'b0}}};
        end else begin : g_nopad
            assign mant = frac;
        end
        assign exp_r = exp_pre;
    end else begin : g_round
        logic [OUT_M-1:0] keep;
        logic             g, st, ru, carry;
        assign keep = frac[14 -: OUT_M];
        assign g    = frac[14-OUT_M];
        if (14-OUT_M >= 1) begin : g_st
            assign st = |frac[13-OUT_M:0];
        end else begin : g_nost
            assign st = 1'b0;
        end
        assign ru = g & (st | keep[0]);
        assign {carry, mant} = {1'b0, keep} + {{OUT_M{1'b0}}, ru};
        assign exp_r = exp_pre + (carry ? 11'sd1 : 11'sd0);
    end

    localparam logic [OUT_M+8:0] QNAN = {1'b0, 8'hFF, 1'b1, {(OUT_M-1){1'b0}}};

    always_comb begin
        if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero))
            y = QNAN;
        else if (a_inf || b_inf)
            y = {sy, 8'hFF, {OUT_M{1'b0}}};
        else if (a_zero || b_zero)
            y = {sy, 8'h00, {OUT_M{1'b0}}};
        else if (exp_r >= 11'sd255)
            y = {sy, 8'hFF, {OUT_M{1'b0}}};
        else if (exp_r <= 11'sd0)
            y = {sy, 8'h00, {OUT_M{1'b0}}};
        else
            y = {sy, exp_r[7:0], mant};
    end

endmodule

`default_nettype wire
