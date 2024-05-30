set_environment_viewer -instance_names
set_messages -log tmax_unconstrained.log -replace
read_netlist ./syn/out/cve2_multdiv_fast.v
read_netlist ./syn/techlib/NangateOpenCellLibrary.v
run_build_model cve2_multdiv_fast_RV32M3
add_pi_constraints 0 op_b_i[31]
add_pi_constraints 0 op_b_i[30]
add_pi_constraints 0 op_b_i[29]
add_pi_constraints 0 op_b_i[28]
add_pi_constraints 0 op_b_i[27]
add_pi_constraints 0 op_b_i[26]
add_pi_constraints 0 op_b_i[25]
add_pi_constraints 0 op_b_i[24]
add_pi_constraints 0 op_b_i[23]
add_pi_constraints 0 op_b_i[22]
add_pi_constraints 0 op_b_i[21]
add_pi_constraints 0 op_b_i[20]
add_pi_constraints 0 op_b_i[19]
add_pi_constraints 0 op_b_i[18]
add_pi_constraints 0 op_b_i[17]
add_pi_constraints 0 op_b_i[16]
add_pi_constraints 0 op_b_i[15]
add_pi_constraints 0 op_b_i[14]
add_pi_constraints 0 op_b_i[13]
add_pi_constraints 0 op_b_i[12]
add_pi_constraints 0 op_b_i[11]
add_pi_constraints 0 op_b_i[10]
add_pi_constraints 0 op_b_i[9]
add_pi_constraints 0 op_b_i[8]
add_pi_constraints 0 op_b_i[7]
add_pi_constraints 1 op_a_i[31]
add_pi_constraints 1 op_a_i[30]
add_pi_constraints 1 op_a_i[29]
add_pi_constraints 1 op_a_i[28]
add_pi_constraints 1 op_a_i[27]
add_pi_constraints 1 op_a_i[26]
add_pi_constraints 1 op_a_i[25]
add_pi_constraints 1 op_a_i[24]
add_pi_constraints 1 op_a_i[23]
add_pi_constraints 1 op_a_i[22]
add_pi_constraints 1 op_a_i[21]
add_pi_constraints 1 op_a_i[20]
add_pi_constraints 1 op_a_i[19]
add_pi_constraints 1 op_a_i[18]
add_pi_constraints 1 op_a_i[17]
add_pi_constraints 1 op_a_i[16]
add_pi_constraints 1 op_a_i[15]
add_pi_constraints 1 op_a_i[14]
add_pi_constraints 1 op_a_i[13]
add_pi_constraints 1 op_a_i[12]
add_pi_constraints 1 op_a_i[11]
add_pi_constraints 1 op_a_i[10]
add_pi_constraints 1 op_a_i[9]
add_pi_constraints 1 op_a_i[8]
add_pi_constraints 1 op_a_i[7]
add_pi_constraints 1 op_a_i[6]
add_pi_constraints 1 op_a_i[5]
add_pi_constraints 1 op_a_i[4]
add_pi_constraints 0 op_a_i[3]
add_pi_constraints 0 op_a_i[2]
add_pi_constraints 0 op_a_i[1]
add_pi_constraints 1 op_a_i[0]
add_pi_constraints 1 { signed_mode_i[0] signed_mode_i[1] }
add_po_masks valid_o
run_drc
read_faults ./flist.txt -add -force_retain_code
set_atpg -merge high -abort_limit 100 -patterns 1
set_faults -model stuck
run_atpg -ndetects 1 -auto_compression
write_patterns mul_patterns.txt -format stil -internal -replace
write_faults ./flist.txt -replace -all
report_summaries > ./summaries.txt
quit
