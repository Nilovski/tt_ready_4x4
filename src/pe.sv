// ---------------------------------------------------------------------------
// pe -- one processing element of the output-stationary array.
//
//              b_in (north)
//                  |
//        a_in --> [PE] --> a_out (east)
//       (west)      |
//                b_out (south)
//
// The MAC consumes a_in/b_in in the same cycle they are registered out, so
// operands march one hop per cycle across the array.
// ---------------------------------------------------------------------------
module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    // Position in the mesh. Used only by the PE_TRACE debug hook.
    parameter int PE_I   = 0,
    parameter int PE_J   = 0
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clr,
    input  logic                en,
    input  logic [DATA_W-1:0]   a_in,
    input  logic [DATA_W-1:0]   b_in,
    output logic [DATA_W-1:0]   a_out,
    output logic [DATA_W-1:0]   b_out,
    output logic [ACC_W-1:0]    acc
);

    mac_unit #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_mac (
        .clk   (clk),
        .rst_n (rst_n),
        .clr   (clr),
        .en    (en),
        .a     (a_in),
        .b     (b_in),
        .acc   (acc)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end

`ifdef PE_TRACE
    // Debug hook: `make trace` shows every PE's inputs and accumulator each
    // cycle. Invaluable when a schedule bug makes the array produce plausible
    // but wrong numbers.
    always_ff @(posedge clk)
        if (rst_n && en)
            $display("      PE(%0d,%0d)  a_in=%4d  b_in=%4d   acc_before=%0d",
                     PE_I, PE_J, $signed(a_in), $signed(b_in), $signed(acc));
`endif

endmodule
