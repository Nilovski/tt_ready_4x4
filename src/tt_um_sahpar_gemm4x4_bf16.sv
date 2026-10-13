`default_nettype none

// ---------------------------------------------------------------------------
// tt_um_sahpar_gemm4x4_bf16 -- Tiny Tapeout top level for the 4x4 BF16
// output-stationary systolic GEMM. Same array, scheduler, decoder and host
// protocol as the int8 tt_um_sahpar_gemm4x4; only the MAC (mac_unit.sv ->
// bf16_mul + fp_add) and the operand/result widths differ:
//   operands: BF16, 2 bytes each, low byte first (LOAD_A/B = 32 data bytes)
//   results : FP32 (4 LE bytes per element, same READ_C addressing)
//
// Pin map (same host byte protocol as the 2x2):
//   ui_in[7:0]  = host_data in   (RX byte)
//   uo_out[7:0] = host_data out  (TX byte)
//   uio[0] = RX_VALID   in   (host asserts when ui_in holds a new byte)
//   uio[1] = RX_READY   out  (always 1 -- see rx_byte_interface)
//   uio[2] = TX_READY   in   (host asserts to accept the byte on uo_out)
//   uio[3] = TX_VALID   out  (chip asserts while uo_out holds an unread byte)
//   uio[4] = BUSY       out  (GEMM running; STATUS bit 0)
//   uio[5] = DONE       out  (sticky: a run finished and C is ready; cleared by
//                             the next GEMM/CLEAR_GEMM; STATUS bit 1). Usable
//                             as an interrupt line on the host.
//   uio[7:6] reserved (inputs)
// ---------------------------------------------------------------------------
module tt_um_sahpar_gemm4x4_bf16 (
    input  logic [7:0] ui_in,
    output logic [7:0] uo_out,
    input  logic [7:0] uio_in,
    output logic [7:0] uio_out,
    output logic [7:0] uio_oe,
    input  logic       ena,
    input  logic       clk,
    input  logic       rst_n
);

    localparam int N      = 4;
    localparam int K      = 4;
    localparam int DATA_W = 16;  // BF16 operands
    // 16: BF16 accumulator (project spec: "BF16 ADD <- ACC_REG"); the full
    //     design synthesizes to ~142k um^2 -> fits 6x2 at ~61%.
    // 32: FP32 accumulator -- far more accurate running sums, but ~221k um^2
    //     -> needs 8x2 (~71%, tight). Change here + tiles in info.yaml.
    localparam int ACC_W  = 16;

    // ---- RX ----------------------------------------------------------
    logic       rx_byte_valid;
    logic       rx_host_ready;
    logic [7:0] rx_byte;

    rx_byte_interface u_rx (
        .clk           (clk),
        .rst_n         (rst_n),
        .host_data     (ui_in),
        .host_valid    (uio_in[0]),
        .host_ready    (rx_host_ready),
        .rx_byte       (rx_byte),
        .rx_byte_valid (rx_byte_valid)
    );

    // ---- TX ------------------------------------------------------------
    logic [7:0] tx_byte;
    logic       tx_byte_valid;
    logic       tx_ready;
    logic       tx_host_valid;

    tx_byte_interface u_tx (
        .clk           (clk),
        .rst_n         (rst_n),
        .tx_byte       (tx_byte),
        .tx_byte_valid (tx_byte_valid),
        .tx_ready_o    (tx_ready),
        .host_ready    (uio_in[2]),
        .host_data     (uo_out),
        .host_valid    (tx_host_valid)
    );

    // ---- decoder + compute core ----------------------------------------
    logic                  gemm_start, gemm_accumulate, gemm_busy, gemm_done;
    logic                  result_done;
    logic [N*K*DATA_W-1:0] a_flat;
    logic [K*N*DATA_W-1:0] b_flat;
    logic [N*N*ACC_W-1:0]  c_flat;

    cmd_decoder #(
        .N(N), .K(K), .DATA_W(DATA_W), .ACC_W(ACC_W), .CHIP_ID(8'hB4),
        .C_FLOAT(1'b1)
    ) u_decoder (
        .clk           (clk),
        .rst_n         (rst_n),
        .rx_byte       (rx_byte),
        .rx_byte_valid (rx_byte_valid & ena),
        .tx_byte       (tx_byte),
        .tx_byte_valid (tx_byte_valid),
        .tx_ready      (tx_ready),
        .gemm_start      (gemm_start),
        .gemm_accumulate (gemm_accumulate),
        .a_flat          (a_flat),
        .b_flat          (b_flat),
        .gemm_busy       (gemm_busy),
        .gemm_done       (gemm_done),
        .c_flat          (c_flat),
        .result_done     (result_done)
    );

    gemm_top #(
        .N(N), .K(K), .DATA_W(DATA_W), .ACC_W(ACC_W),
        .MAC_LATENCY(0), .LATCH_OPERANDS(1'b0)
    ) u_gemm (
        .clk    (clk),
        .rst_n  (rst_n),
        .start      (gemm_start),
        .accumulate (gemm_accumulate),
        .a_flat (a_flat),
        .b_flat (b_flat),
        .busy   (gemm_busy),
        .done   (gemm_done),
        .c_flat (c_flat)
    );

    assign uio_out = {2'b0, result_done, gemm_busy, tx_host_valid, 1'b0, rx_host_ready, 1'b0};
    assign uio_oe  = 8'b0011_1010;   // [5]=DONE, [4]=BUSY, [3]=TX_VALID, [1]=RX_READY

    wire _unused = &{uio_in[7:3], uio_in[1], 1'b0};

endmodule

`default_nettype wire
