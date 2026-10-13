`timescale 1ns/1ps
// Vector-driven testbench for bf16_mul / fp_add. OP selects the unit:
//   0: bf16_mul OUT_M=7   1: bf16_mul OUT_M=23   2: fp_add M=7   3: fp_add M=23
// Reads "a b" hex pairs from +in=..., writes results as hex to +out=...
module tb_fp;
    logic [15:0] ma, mb;
    logic [31:0] aa, ab;
    logic [15:0] y_m7;  logic [31:0] y_m23;
    logic [15:0] y_a7;  logic [31:0] y_a23;

    bf16_mul #(.OUT_M(7))  u_m7  (.a(ma), .b(mb), .y(y_m7));
    bf16_mul #(.OUT_M(23)) u_m23 (.a(ma), .b(mb), .y(y_m23));
    fp_add   #(.M(7))      u_a7  (.a(aa[15:0]), .b(ab[15:0]), .y(y_a7));
    fp_add   #(.M(23))     u_a23 (.a(aa), .b(ab), .y(y_a23));

    integer fin, fout, op, n, cnt;
    reg [1023:0] fn_in, fn_out;
    reg [31:0] x, z;

    initial begin
        if (!$value$plusargs("op=%d", op)) op = 0;
        if (!$value$plusargs("in=%s", fn_in)) $fatal(1, "need +in");
        if (!$value$plusargs("out=%s", fn_out)) $fatal(1, "need +out");
        fin  = $fopen(fn_in, "r");
        fout = $fopen(fn_out, "w");
        cnt = 0;
        while (!$feof(fin)) begin
            n = $fscanf(fin, "%h %h\n", x, z);
            if (n == 2) begin
                ma = x[15:0]; mb = z[15:0]; aa = x; ab = z;
                #1;
                case (op)
                    0: $fwrite(fout, "%h\n", y_m7);
                    1: $fwrite(fout, "%h\n", y_m23);
                    2: $fwrite(fout, "%h\n", y_a7);
                    default: $fwrite(fout, "%h\n", y_a23);
                endcase
                cnt = cnt + 1;
            end else if (!$feof(fin)) begin
                $fatal(1, "tb_fp: malformed vector line after %0d vectors", cnt);
            end
        end
        $fclose(fin); $fclose(fout);
        $display("tb_fp: op=%0d vectors=%0d", op, cnt);
        $finish;
    end
endmodule
