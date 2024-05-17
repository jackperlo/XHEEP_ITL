#!/bin/bash

# THIS SCRIPT IS MEANT TO BE LAUNCHED FROM THE RISCV_MUL FOLDER

scripts_folder="../XHEEP_ITL/outputs/atpg_scripts"
pattern_file="../XHEEP_ITL/outputs/atpg_patterns_gathered/patterns.txt"
# clean up the pattern file 
> $pattern_file

# total iterations can be retreived by multiplying (n_channels_out*height*width*n_filters)
total_iterations=5*5*5*1

# variables used to calculate the atpg script file name
n_channels_out=0
height=0
width=0
n_filters=0

for ((i=0; i<$total_iterations; i++))
do
  # computing file name
  if [ $i -lt 5 ]; then
    width=$i
  else
    width=$(($i % 5))
    i=$(($i / 5))
    if [ $i -ge 5 ]; then
      height=$(($i % 5))
      n_channels_out=$(($i / 5))
    else
      height=$i
      n_channels_out=0
    fi
  fi

  echo "Execution of script: weight_${n_channels_out}_${height}_${width}_${n_filters}.tcl"

  script_file="$scripts_folder/weight_${n_channels_out}_${height}_${width}_${n_filters}.tcl"
  # check for script existance
  if [ -f "$script_file" ]; then
    # run the script
    max -shell "$script_file"

    # retrieve the _pi pattern (only the 32 bits of the input) from the mul_patterns and enqueue it in the patterns.txt file
    pattern=$(awk '/"_pi"=/ {if (match($0, /"_pi"=([01]{64})[01]*/)) {print substr($0, RSTART+45, 32); exit}}' mul_patterns.txt | tr -d ';')
    # remove the double carriage return which comes out from the awk command 
    pattern=${pattern#"${pattern%%[![:space:]]*}"}
    pattern=${pattern#"${pattern%%[![:space:]]*}"}
    # quit from atpg console
    echo -ne 'quit\n' 
    echo "pattern_${n_channels_out}_${height}_${width}_${n_filters} : ${pattern}" >> "${pattern_file}"

    echo "pattern_${n_channels_out}_${height}_${width}_${n_filters} saved correctly"
    
  else
    echo "ATPG script file not found: $script_file. Skipping this file."
  fi
done
