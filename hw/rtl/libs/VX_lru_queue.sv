// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_platform.vh"
`include "VX_cache_define.vh"

`TRACING_OFF
module VX_lru_queue #(
    parameter DATAW     = 1,
    parameter DEPTH     = 2,
    parameter ALM_FULL  = (DEPTH - 1),
    parameter ALM_EMPTY = 1,
    parameter OUT_REG   = 0,
    //parameter LUTRAM    = 1,
    parameter SIZEW     = `CLOG2(DEPTH+1)
) ( 
    input  wire             clk,
    input  wire             reset,    
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,      
    output wire             alm_empty,
    output wire             full,            
    output wire             alm_full,
    output wire [SIZEW-1:0] size
); 
    
    localparam ADDRW = `CLOG2(DEPTH);    

    `STATIC_ASSERT(ALM_FULL > 0, ("alm_full must be greater than 0!"))
    `STATIC_ASSERT(ALM_FULL < DEPTH, ("alm_full must be smaller than size!"))
    `STATIC_ASSERT(ALM_EMPTY > 0, ("alm_empty must be greater than 0!"))
    `STATIC_ASSERT(ALM_EMPTY < DEPTH, ("alm_empty must be smaller than size!"))
    `STATIC_ASSERT(`IS_POW2(DEPTH), ("size must be a power of 2!"))
    
    if (DEPTH == 1) begin

        reg [DATAW-1:0] head_r;
        reg size_r;

        always @(posedge clk) begin
            if (reset) begin
                head_r <= '0;
                size_r <= '0;                    
            end else begin
                `ASSERT(~push || ~full, ("runtime error: writing to a full queue"));
                `ASSERT(~pop || ~empty, ("runtime error: reading an empty queue"));
                if (push) begin
                    if (~pop) begin
                        size_r <= 1;
                    end
                end else if (pop) begin
                    size_r <= '0;
                end
                if (push) begin 
                    head_r <= data_in;
                end
            end
        end        

        assign data_out  = head_r;
        assign empty     = (size_r == 0);
        assign alm_empty = 1'b1;
        assign full      = (size_r != 0);
        assign alm_full  = 1'b1;
        assign size      = size_r;

    end else begin

        // FIFO
        reg empty_r, alm_empty_r;
        reg full_r, alm_full_r;
        reg [ADDRW-1:0] used_r;
        wire [ADDRW-1:0] used_n;

        always @(posedge clk) begin
            if (reset) begin
                empty_r     <= 1;
                alm_empty_r <= 1;    
                full_r      <= 0;        
                alm_full_r  <= 0;
                used_r      <= '0;
            end else begin
                `ASSERT(~(push && ~pop) || ~full, ("runtime error: incrementing full queue"));
                `ASSERT(~(pop && ~push) || ~empty, ("runtime error: decrementing empty queue"));
                if (push) begin
                    if (~pop) begin
                        empty_r <= 0;
                        if (used_r == ADDRW'(ALM_EMPTY))
                            alm_empty_r <= 0;
                        if (used_r == ADDRW'(DEPTH-1))
                            full_r <= 1;
                        if (used_r == ADDRW'(ALM_FULL-1))
                            alm_full_r <= 1;
                    end 
                end else if (pop) begin
                    full_r <= 0;
                    if (used_r == ADDRW'(ALM_FULL))
                        alm_full_r <= 0;            
                    if (used_r == ADDRW'(1))
                        empty_r <= 1;
                    if (used_r == ADDRW'(ALM_EMPTY+1))
                        alm_empty_r <= 1;
                end                
                used_r <= used_n;  
            end                   
        end

        if (DEPTH == 2) begin

            assign used_n = used_r ^ (push ^ pop);

            if (0 == OUT_REG) begin 

                reg [1:0][DATAW-1:0] shift_reg;

                always @(posedge clk) begin
                    if (push) begin
                        shift_reg[1] <= shift_reg[0];
                        shift_reg[0] <= data_in;
                    end
                end

                assign data_out = shift_reg[!used_r[0]];    
                
            end else begin

                reg [DATAW-1:0] data_out_r;
                reg [DATAW-1:0] buffer;

                always @(posedge clk) begin
                    if (push) begin
                        buffer <= data_in;
                    end
                    if (push && (empty_r || (used_r && pop))) begin
                        data_out_r <= data_in;
                    end else if (pop) begin
                        data_out_r <= buffer;
                    end
                end

                assign data_out = data_out_r;

            end
        
        end else begin
            
            assign used_n = $signed(used_r) + ADDRW'($signed(2'(push) - 2'(pop)));

            if (0 == OUT_REG) begin          

                reg [DATAW-1:0] data_reg [DEPTH-1:0]; // Memory were data is stored
                reg [ADDRW-1:0] queue_ptr_reg [DEPTH-1:0]; // Ptr to memory data, 0 is lru, DEPTH-1 is mru
                reg [DEPTH-1:0] used_reg; // Memory were data is stored
                
                always @(posedge clk) begin
                    if (reset) begin
                        used_reg <= '0;
                        for (integer i = 0; i < DEPTH; i = i + 1) begin // Set all the positions in the queue
                            queue_ptr_reg[i] <= ADDRW'(i);
                        end
                    end else begin
                        if (~push && ~pop ) begin // write or read on a load memory, PRE memory is on data_reg
                            /*
                            // Set to first position ADDR of the read
			                reg [ADDRW-1:0] ptr;
                            for (integer i = 0; i < DEPTH; i = i + 1) begin
                                if (data_reg[queue_ptr_reg[i]] == data_in) begin
                                    ptr = ADDRW'(i);
                                    break;
                                end
                            end
                            // Move the respective positions
                            for (integer i = integer'(ptr)+1; i < integer'(used_n)-1; i = i +1) begin
                                            queue_ptr_reg[i-1] <= queue_ptr_reg[i];
                            end
                            queue_ptr_reg[used_n-1] <= queue_ptr_reg[ptr];
			                */
                        end else begin 
                            if (push) begin // put new data in first empty position, last if not found
                                if (pop) begin // insert the data were is going to pop
                                    if (used_n == DEPTH) begin
                                        data_reg[queue_ptr_reg[0]] <= data_in; 
                                    end else begin
                                        data_reg[queue_ptr_reg[used_n-1]] <= data_in;
                                    end
                                end else begin // find a position were to store new data
                                    data_reg[queue_ptr_reg[used_n-1]] <= data_in;
                                    used_reg[queue_ptr_reg[used_n-1]] <= 1;
                                end
                            end
                            if (pop) begin
                                if (~push) begin // register gets empty
                                    used_reg[queue_ptr_reg[0]] <= 0;
                                end
                                // move all the queue one step
                                for (integer i = 0; i < DEPTH-1; i = i + 1) begin
                                    queue_ptr_reg[i] <= queue_ptr_reg[i+1];
                                end
                                queue_ptr_reg[DEPTH-1] <= queue_ptr_reg[0];
                            end
                        end
                    end               
                end

                assign data_out = data_reg[queue_ptr_reg[0]];

            end else begin

                reg [DATAW-1:0] dout_r;
                reg [DEPTH-1:0][DATAW-1:0] data_reg; // Memory were data is stored
                reg [DEPTH-1:0][ADDRW-1:0] queue_ptr_reg; // Ptr to memory data, 0 is lru, DEPTH-1 is mru
                reg [DEPTH-1:0] used_reg; // Memory were data is stored

                // reg [`CS_LINE_ADDR_WIDTH-1:0] addr_r [DEPTH-1:0]; // Address stored

                always @(posedge clk) begin
                    if (reset) begin  
                        used_reg <= '0; // Set all registers to unused
                        for (integer i = 0; i < DEPTH; i = i + 1) begin // Set all the positions in the queue
                            queue_ptr_reg[i] <= ADDRW'(i);
                        end
                    end else begin
                        if (~push && ~pop ) begin // write or read on a load memory, PRE memory is on data_reg
                            /*
                            // Set to first position ADDR of the read
			                reg [ADDRW-1:0] ptr;
                            for (integer i = 0; i < DEPTH; i = i + 1) begin
                                if (data_reg[queue_ptr_reg[i]] == data_in) begin
                                    ptr = ADDRW'(i);
                                    break;
                                end
                            end
                            // Move the respective positions
                            for (integer i = integer'(ptr)+1; i < integer'(used_n)-1; i = i +1) begin
                                queue_ptr_reg[i-1] <= queue_ptr_reg[i];
                            end
                            queue_ptr_reg[used_n-1] <= queue_ptr_reg[ptr];
			                */
                        end else begin 
                            if (push) begin // put new data in first empty position, last if not found
                                if (pop) begin // insert the data were is going to pop
                                    if (used_n == DEPTH) begin
                                        data_reg[queue_ptr_reg[0]] <= data_in; 
                                    end else begin
                                        data_reg[queue_ptr_reg[used_n-1]] <= data_in;
                                    end
                                end else begin // find a position were to store new data
                                    data_reg[queue_ptr_reg[used_n-1]] <= data_in;
                                    used_reg[queue_ptr_reg[used_n-1]] <= 1;
                                end
                            end
                            if (pop) begin
                                if (~push) begin // register gets empty
                                    used_reg[queue_ptr_reg[0]] <= 0;
                                end
                                // move all the queue one step
                                for (integer i = 0; i < DEPTH-1; i = i + 1) begin
                                    queue_ptr_reg[i] <= queue_ptr_reg[i+1];
                                end
                                queue_ptr_reg[DEPTH-1] <= queue_ptr_reg[0];
                            end
                        end
                    end
                end

                wire going_empty;
                if (ALM_EMPTY == 1) begin
                    assign going_empty = alm_empty_r;
                end else begin
                    assign going_empty = (used_r == ADDRW'(1));
                end

                always @(posedge clk) begin
                    if (push && (empty_r || (going_empty && pop))) begin
                        dout_r <= data_in;
                    end else if (pop) begin
                        dout_r <= data_reg[queue_ptr_reg[0]];
                    end
                end

                assign data_out = dout_r;
            end
        end
        
        assign empty     = empty_r;        
        assign alm_empty = alm_empty_r;
        assign full      = full_r;
        assign alm_full  = alm_full_r;
        assign size      = {full_r, used_r};        
    end

endmodule
`TRACING_ON
