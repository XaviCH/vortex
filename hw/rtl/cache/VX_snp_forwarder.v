`include "VX_cache_config.vh"

module VX_snp_forwarder #(
    parameter BANK_LINE_SIZE    = 0, 
    parameter NUM_REQUESTS      = 0, 
    parameter SNRQ_SIZE         = 0,
    parameter SNP_REQ_TAG_WIDTH = 0,
    parameter SNP_FWD_TAG_WIDTH = 0
) (
    input wire clk,
    input wire reset,

    // Snoop request
    input wire                          snp_req_valid,
    input wire [`DRAM_ADDR_WIDTH-1:0]   snp_req_addr,
    input wire [SNP_REQ_TAG_WIDTH-1:0]  snp_req_tag,
    output wire                         snp_req_ready,

    // Snoop response
    output wire                         snp_rsp_valid,    
    output wire [`DRAM_ADDR_WIDTH-1:0]  snp_rsp_addr,
    output wire [SNP_REQ_TAG_WIDTH-1:0] snp_rsp_tag,
    input  wire                         snp_rsp_ready,

    // Snoop Forwarding out
    output wire [NUM_REQUESTS-1:0]      snp_fwdout_valid,
    output wire [NUM_REQUESTS-1:0][`DRAM_ADDR_WIDTH-1:0] snp_fwdout_addr,
    output wire [NUM_REQUESTS-1:0][SNP_FWD_TAG_WIDTH-1:0] snp_fwdout_tag,
    input wire [NUM_REQUESTS-1:0]       snp_fwdout_ready,

    // Snoop forwarding in
    input wire [NUM_REQUESTS-1:0]       snp_fwdin_valid,    
    input wire [NUM_REQUESTS-1:0][SNP_FWD_TAG_WIDTH-1:0] snp_fwdin_tag,
    output wire [NUM_REQUESTS-1:0]      snp_fwdin_ready
);
    reg [`DRAM_ADDR_WIDTH+SNP_REQ_TAG_WIDTH-1:0] pending_reqs [SNRQ_SIZE-1:0];
    reg [`REQS_BITS:0] pending_cntrs [SNRQ_SIZE-1:0];
    reg [`LOG2UP(SNRQ_SIZE):0] rd_ptr, wr_ptr;
    reg [`REQS_BITS-1:0] fwdin_sel;

    wire [`LOG2UP(SNRQ_SIZE)-1:0] rd_a, wr_a;

    wire enqueue, dequeue, empty, full;

    wire fwdout_ready;

    wire fwdin_valid;
    wire [SNP_FWD_TAG_WIDTH-1:0] fwdin_tag;
    wire fwdin_ready;
    wire fwdin_taken;
    
    assign fwdout_ready = (& snp_fwdout_ready);

    assign snp_req_ready = !full && fwdout_ready;

    assign rd_a = rd_ptr[`LOG2UP(SNRQ_SIZE)-1:0];
    assign wr_a = wr_ptr[`LOG2UP(SNRQ_SIZE)-1:0];

    genvar i;

    for (i = 0; i < NUM_REQUESTS; i++) begin
        assign snp_fwdout_valid[i] = enqueue && fwdout_ready;
        assign snp_fwdout_addr[i]  = snp_req_addr;
        assign snp_fwdout_tag[i]   = wr_a;
    end

    assign fwdin_ready = snp_rsp_ready;
    assign fwdin_taken = fwdin_valid && fwdin_ready;

    assign snp_rsp_valid = fwdin_taken && (1 == pending_cntrs[fwdin_tag]); // send response
    assign {snp_rsp_addr, snp_rsp_tag} = pending_reqs[fwdin_tag];

    assign empty   = (wr_ptr == rd_ptr);
    assign full    = (wr_a == rd_a) && (wr_ptr[`LOG2UP(SNRQ_SIZE)] != rd_ptr[`LOG2UP(SNRQ_SIZE)]);
    
    assign enqueue = snp_req_valid && snp_req_ready;       
    assign dequeue = !empty && (0 == pending_cntrs[rd_a]);

    always @(posedge clk) begin
        if (reset) begin
            rd_ptr <= 0;
            wr_ptr <= 0;
        end else begin
            if (enqueue)  begin
                pending_reqs[wr_a]  <= {snp_req_addr, snp_req_tag};
                pending_cntrs[wr_a] <= NUM_REQUESTS;
                wr_ptr              <= wr_ptr + 1;
            end            
            if (dequeue) begin
                rd_ptr <= rd_ptr + 1;
            end
            if (fwdin_taken) begin
                pending_cntrs[fwdin_tag] <= pending_cntrs[fwdin_tag] - 1;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            fwdin_sel <= 0;
        end else begin
            fwdin_sel <= fwdin_sel + 1;
        end
    end

    assign fwdin_valid = snp_fwdin_valid[fwdin_sel];
    assign fwdin_tag = snp_fwdin_tag[fwdin_sel];

    for (i = 0; i < NUM_REQUESTS; i++) begin
        assign snp_fwdin_ready[i] = fwdin_ready && (fwdin_sel == `REQS_BITS'(i));
    end

`ifdef DBG_PRINT_CACHE_SNP
     always_ff @(posedge clk) begin
        if (snp_req_valid && snp_req_ready) begin
            $display("%t: snp req: addr=%0h, tag=%0h", $time, snp_req_addr, snp_req_tag);
        end
        if (snp_fwdout_valid[0] && snp_fwdout_ready[0]) begin
            $display("%t: snp fwd_out: addr=%0h, tag=%0h", $time, snp_fwdout_addr[0], snp_fwdout_tag[0]);
        end
        if (fwdin_valid && fwdin_ready) begin
            $display("%t: snp fwd_in[%01d]: tag=%0h", $time, fwdin_sel, fwdin_tag);
        end
        if (snp_rsp_valid && snp_rsp_ready) begin
            $display("%t: snp rsp: addr=%0h, tag=%0h", $time, snp_rsp_addr, snp_rsp_tag);
        end
    end
`endif

endmodule