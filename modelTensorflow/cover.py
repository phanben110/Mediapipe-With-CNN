import os
import subprocess
import shutil

source_cmd= "/opt/intel/openvino_2021.2.185/bin/setupvars.sh"

# Setup model optimizer command ...
ir_name = "modelOpenvino"
model_fname ="modelOpenvino"
ir_data_type = "FP32"
ir_out_dir = f"{model_fname}/IR_models/{ir_data_type}"
ir_input_shape = "[1,26,26,1]"
mo_cmd = f"/home/pcwork/openvino/model-optimizer/mo_tf.py \
      --saved_model_dir {model_fname} \
      --input_shape {ir_input_shape} \
      --data_type {ir_data_type} \
      --output_dir {ir_out_dir}  \
      --model_name {ir_name}"
print ("Running model optimizer to convert model to OpenVINO IR format ....")
print("\n--".join(mo_cmd.split("--")))

output = subprocess.check_output(source_cmd+" && "+mo_cmd, shell=True)
print (output.decode('utf-8'))     
