`default_nettype none

// ---------------------------------------------------------------------------
// cmd_decoder -- byte-protocol front end for the N x N integer GEMM core
// (gemm_top). Same opcode set as the verified 2x2 bring-up, sized by N/K:
//
//   0x00  NOP
//   0x10  LOAD_A          -- followed by N*K data bytes (row-major)
//   0x11  LOAD_B          -- followed by K*N data bytes (row-major)
//   0x20  GEMM            -- pulses start (ignored while busy); C += A x B
//   0x21  CLEAR_GEMM      -- pulses start (ignored while busy); C  = A x B
//   0x31  READ_C_ELEMENT  -- followed by 1 address byte (0..N*N*4-1),
//                            replies with that byte of C
//   0x40  STATUS          -- replies {6'b0, done, busy}; done is sticky:
//                            set when a run finishes, cleared by the next
//                            GEMM/CLEAR_GEMM (also driven on a pin)
//   0x41  ID              -- replies CHIP_ID
//
// Anything else is ignored.
//
// Area-driven differences vs. cmd_decoder_2x2:
//   * The A/B byte registers here are the ONLY operand copy -- gemm_top is
//     built with LATCH_OPERANDS=0 and reads a_flat/b_flat directly. To keep
//     them stable during a run, data bytes of a LOAD_A/LOAD_B that arrives
//     while busy are consumed (so the byte stream stays in sync) but
//     discarded.
//   * C is read straight out of the PE accumulators (they hold after done,
//     since the array is only enabled while busy) instead of being copied
//     into a 64-byte shadow buffer. Reading during a run returns a partial
//     sum -- poll STATUS first.
//   * Accumulators are ACC_W bits; each element is sign-extended to 32 bits
//     on readout so the host still sees 4 little-endian bytes. With GEMM
//     (accumulate) the host must keep the total number of summed products
//     within ACC_W (see top level).
// ---------------------------------------------------------------------------
module cmd_decoder #(
    parameter int N        = 4,
    parameter int K        = 4,
    parameter int DATA_W   = 8,
    parameter int ACC_W    = 19,
    parameter logic [7:0] CHIP_ID = 8'hA4
) (
    input  logic clk,
    input  logic rst_n,

    // from rx_byte_interface
    input  logic [7:0] rx_byte,
    input  logic       rx_byte_valid,

    // to tx_byte_interface
    output logic [7:0] tx_byte,
    output logic       tx_byte_valid,
    input  logic       tx_ready,

    // gemm_top
    output logic                  gemm_start,
    output logic                  gemm_accumulate,
    output logic [N*K*DATA_W-1:0] a_flat,
    output logic [K*N*DATA_W-1:0] b_flat,
    input  logic                  gemm_busy,
    input  logic                  gemm_done,
    input  logic [N*N*ACC_W-1:0]  c_flat,

    output logic                  result_done   // sticky DONE flag
);

    localparam int NUM_A_BYTES = N*K*DATA_W/8;
    localparam int NUM_B_BYTES = K*N*DATA_W/8;
    localparam int NUM_C_ELEMS = N*N;
    localparam int AW = (NUM_A_BYTES > 1) ? $clog2(NUM_A_BYTES) : 1;
    localparam int BW = (NUM_B_BYTES > 1) ? $clog2(NUM_B_BYTES) : 1;
    localparam int EW = (NUM_C_ELEMS > 1) ? $clog2(NUM_C_ELEMS) : 1;

    localparam logic [7:0] OP_NOP            = 8'h00;
    localparam logic [7:0] OP_LOAD_A         = 8'h10;
    localparam logic [7:0] OP_LOAD_B         = 8'h11;
    localparam logic [7:0] OP_GEMM           = 8'h20;
    localparam logic [7:0] OP_CLEAR_GEMM     = 8'h21;
    localparam logic [7:0] OP_READ_C_ELEMENT = 8'h31;
    localparam logic [7:0] OP_STATUS         = 8'h40;
    localparam logic [7:0] OP_ID             = 8'h41;

    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD_A,
        S_LOAD_B,
        S_READ_ADDR
    } state_t;

    state_t state;

    logic [7:0]    a_bytes [NUM_A_BYTES];
    logic [7:0]    b_bytes [NUM_B_BYTES];
    logic [AW-1:0] a_idx;
    logic [BW-1:0] b_idx;
    logic          load_drop;   // LOAD started while busy: swallow its data

    // ---- C readout: element select, sign-extend to 32b, byte select -------
    logic [EW-1:0]           rd_elem;
    logic [1:0]              rd_byte;
    logic signed [ACC_W-1:0] rd_acc;
    logic [31:0]             rd_word;
    logic [7:0]              rd_data;

    assign rd_elem = rx_byte[EW+1:2];
    assign rd_byte = rx_byte[1:0];
    assign rd_acc  = c_flat[rd_elem*ACC_W +: ACC_W];
    assign rd_word = 32'(rd_acc);                 // sign-extending cast
    assign rd_data = rd_word[rd_byte*8 +: 8];

    // Note: replies are only issued one per command and the host drains each
    // before sending the next command, so tx_ready back-pressure is not
    // modelled (same contract as the 2x2 bring-up).

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            a_idx         <= '0;
            b_idx         <= '0;
            load_drop     <= 1'b0;
            gemm_start      <= 1'b0;
            gemm_accumulate <= 1'b0;
            result_done     <= 1'b0;
            tx_byte       <= '0;
            tx_byte_valid <= 1'b0;
            for (int i = 0; i < NUM_A_BYTES; i++) a_bytes[i] <= '0;
            for (int i = 0; i < NUM_B_BYTES; i++) b_bytes[i] <= '0;
        end else begin
            gemm_start    <= 1'b0;
            tx_byte_valid <= 1'b0;

            // sticky DONE: set on completion, cleared when a new run is issued
            if (gemm_done)       result_done <= 1'b1;
            else if (gemm_start) result_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (rx_byte_valid) begin
                        case (rx_byte)
                            OP_NOP: ;
                            OP_ID: begin
                                tx_byte       <= CHIP_ID;
                                tx_byte_valid <= 1'b1;
                            end
                            OP_STATUS: begin
                                tx_byte       <= {6'b0, result_done, gemm_busy};
                                tx_byte_valid <= 1'b1;
                            end
                            OP_LOAD_A: begin
                                state     <= S_LOAD_A;
                                a_idx     <= '0;
                                load_drop <= gemm_busy;
                            end
                            OP_LOAD_B: begin
                                state     <= S_LOAD_B;
                                b_idx     <= '0;
                                load_drop <= gemm_busy;
                            end
                            OP_GEMM, OP_CLEAR_GEMM: begin
                                if (!gemm_busy) begin
                                    gemm_start      <= 1'b1;
                                    gemm_accumulate <= (rx_byte == OP_GEMM);
                                end
                            end
                            OP_READ_C_ELEMENT: state <= S_READ_ADDR;
                            default: ;
                        endcase
                    end
                end

                S_LOAD_A: begin
                    if (rx_byte_valid) begin
                        if (!load_drop) a_bytes[a_idx] <= rx_byte;
                        if (a_idx == AW'(NUM_A_BYTES - 1)) state <= S_IDLE;
                        else                                a_idx <= a_idx + AW'(1);
                    end
                end

                S_LOAD_B: begin
                    if (rx_byte_valid) begin
                        if (!load_drop) b_bytes[b_idx] <= rx_byte;
                        if (b_idx == BW'(NUM_B_BYTES - 1)) state <= S_IDLE;
                        else                                b_idx <= b_idx + BW'(1);
                    end
                end

                S_READ_ADDR: begin
                    if (rx_byte_valid) begin
                        tx_byte       <= rd_data;
                        tx_byte_valid <= 1'b1;
                        state         <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // pack operand bytes into gemm_top's flat buses
    always_comb begin
        a_flat = '0;
        for (int i = 0; i < NUM_A_BYTES; i++)
            a_flat[i*DATA_W +: DATA_W] = a_bytes[i];
        b_flat = '0;
        for (int i = 0; i < NUM_B_BYTES; i++)
            b_flat[i*DATA_W +: DATA_W] = b_bytes[i];
    end

    // address bits above the valid range (addr >= NUM_C_BYTES) alias/wrap
    wire _unused = &{tx_ready, rx_byte[7:EW+2], 1'b0};

endmodule

`default_nettype wire
