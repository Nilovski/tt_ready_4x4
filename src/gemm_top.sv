// ---------------------------------------------------------------------------
// gemm_top -- C = A * B on an N x N output-stationary systolic array.
//
//   A is N x K   (row-major in a_flat: index i*K + k)
//   B is K x N   (row-major in b_flat: index k*N + j)
//   C is N x N   (row-major in c_flat: index i*N + j)
//
// Skew scheduling
// ---------------
// PE(i,j) must see A[i][k] and B[k][j] on the same cycle. A[i][k] injected at
// the west edge on cycle (k+i) arrives at column j after j hops -> cycle k+i+j.
// B[k][j] injected at the north edge on cycle (k+j) arrives at row i after i
// hops -> cycle k+i+j. They meet. So row i is delayed by i cycles and column j
// by j cycles; that is the whole scheduler.
//
// Run length T = K + 2*(N-1): K cycles of real data plus the time for the last
// operand to reach PE(N-1,N-1). Trailing cycles inject zeros, which are a
// no-op for the accumulator.
//
// Handshake: pulse start (busy must be low), wait for done, read c_flat.
//
// accumulate (sampled with start): 0 -> C = A*B (accumulators cleared on the
// first cycle); 1 -> C += A*B (accumulators keep their previous value). The
// accumulate form lets a host compute larger matrices by tiling: sum several
// K-blocks into the same C block before reading it out.
// ---------------------------------------------------------------------------
module gemm_top #(
    parameter int N      = 2,
    parameter int K      = 2,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    // Extra cycles the array is held enabled after the last real operand, so a
    // pipelined MAC (e.g. a multi-cycle floating-point unit) can drain before
    // done is asserted. 0 for the combinational integer MAC.
    parameter int MAC_LATENCY = 0,
    // 1: snapshot a_flat/b_flat into local registers on start (2x2 behaviour).
    // 0: feed the array straight from a_flat/b_flat; the caller must hold them
    //    stable while busy. Saves 2*N*K*DATA_W flops when the caller already
    //    keeps the operands in registers (cmd_decoder does).
    parameter bit LATCH_OPERANDS = 1
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic                    accumulate,
    input  logic [N*K*DATA_W-1:0]   a_flat,
    input  logic [K*N*DATA_W-1:0]   b_flat,
    output logic                    busy,
    output logic                    done,
    output logic [N*N*ACC_W-1:0]    c_flat
);

    localparam int T  = K + 2*(N-1) + MAC_LATENCY;
    localparam int CW = (T > 1) ? $clog2(T) : 1;

    // ---- operand registers ------------------------------------------------
    logic [DATA_W-1:0] a_reg [N][K];
    logic [DATA_W-1:0] b_reg [K][N];

    if (LATCH_OPERANDS) begin : g_latch
        always_ff @(posedge clk) begin
            if (!busy && start) begin
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < K; k++)
                        a_reg[i][k] <= a_flat[(i*K + k)*DATA_W +: DATA_W];
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++)
                        b_reg[k][j] <= b_flat[(k*N + j)*DATA_W +: DATA_W];
            end
        end
    end else begin : g_direct
        always_comb begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++)
                    a_reg[i][k] = a_flat[(i*K + k)*DATA_W +: DATA_W];
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++)
                    b_reg[k][j] = b_flat[(k*N + j)*DATA_W +: DATA_W];
        end
    end

    // ---- run counter ------------------------------------------------------
    logic [CW-1:0] t;
    logic          acc_run;   // current run accumulates onto the previous C

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            t       <= '0;
            acc_run <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    busy    <= 1'b1;
                    t       <= '0;
                    acc_run <= accumulate;
                end
            end else begin
                if (t == CW'(T-1)) begin
                    busy <= 1'b0;
                    done <= 1'b1;      // c_flat valid from this cycle onward
                end else begin
                    t <= t + CW'(1);
                end
            end
        end
    end

    // ---- skewed edge feed -------------------------------------------------
    logic [N*DATA_W-1:0] a_west;
    logic [N*DATA_W-1:0] b_north;

    always_comb begin
        a_west  = '0;
        b_north = '0;
        if (busy) begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++)
                    if (t == CW'(k + i))
                        a_west[i*DATA_W +: DATA_W] = a_reg[i][k];
            for (int j = 0; j < N; j++)
                for (int k = 0; k < K; k++)
                    if (t == CW'(k + j))
                        b_north[j*DATA_W +: DATA_W] = b_reg[k][j];
        end
    end

`ifdef PE_TRACE
    always_ff @(posedge clk)
        if (rst_n && busy)
            $display("cycle t=%0d  a_west=%h  b_north=%h  clr=%b",
                     t, a_west, b_north, busy && (t == '0) && !acc_run);
`endif

    // ---- the array --------------------------------------------------------
    systolic_array #(
        .N      (N),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_array (
        .clk     (clk),
        .rst_n   (rst_n),
        // first cycle of a clearing run overwrites; an accumulating run never
        // clears (edge inputs are zero until real data arrives, so the extra
        // cycles add nothing)
        .clr     (busy && (t == '0) && !acc_run),
        .en      (busy),
        .a_west  (a_west),
        .b_north (b_north),
        .c_flat  (c_flat)
    );

endmodule
