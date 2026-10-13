// ---------------------------------------------------------------------------
// mac_unit -- output-stationary BF16 multiply-accumulate cell.
//
// Drop-in replacement for the integer mac_unit (identical ports and control
// contract); this is the swap point the array was designed around.
//
//   a, b : BF16  {sign, exp[7:0], mant[6:0]}
//   acc  : ACC_W = 16 -> BF16 accumulator (project spec; used on the chip)
//          ACC_W = 32 -> FP32 accumulator {sign, exp[7:0], mant[22:0]}
//                        (~1.7x the MAC area, far more accurate sums)
//
//   acc_next = round(acc + round(a*b)) -- bf16_mul then fp_add, each rounding
//   to nearest-even in the accumulator format (the FP32 product is exact).
//   Subnormals flush to zero; see bf16_mul.sv / fp_add.sv for inf/NaN rules.
//
// Contract (same as the integer unit):
//   clr && en  : acc <= +0 + a*b
//   en         : acc <= acc + a*b
//   clr only   : acc <= +0
//   neither    : hold
// Single cycle (MAC_LATENCY = 0). The array injects +0 operands on skew
// ramp-up/drain cycles; x + (+0)*(+0) = x, so they are harmless.
// ---------------------------------------------------------------------------
module mac_unit #(
    parameter int DATA_W = 16,
    parameter int ACC_W  = 32
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clr,
    input  logic                en,
    input  logic [DATA_W-1:0]   a,
    input  logic [DATA_W-1:0]   b,
    output logic [ACC_W-1:0]    acc
);

    localparam int AM = ACC_W - 9;   // accumulator mantissa bits

    // Needs DATA_W = 16 and ACC_W = 16 or 32. (No elaboration-time $error:
    // yosys also elaborates this module at the array's int8 defaults.)

    logic [ACC_W-1:0] prod, sum, acc_q;

    bf16_mul #(.OUT_M(AM)) u_mul (
        .a (a),
        .b (b),
        .y (prod)
    );

    fp_add #(.M(AM)) u_add (
        .a (clr ? {ACC_W{1'b0}} : acc_q),
        .b (prod),
        .y (sum)
    );

    always_ff @(posedge clk) begin
        if (!rst_n)     acc_q <= '0;
        else if (en)    acc_q <= sum;
        else if (clr)   acc_q <= '0;
    end

    assign acc = acc_q;

endmodule
