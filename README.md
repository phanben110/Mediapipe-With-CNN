# Mediapipe-With-CNN

1, Cách chuyển đổi model từ Pytorch sang .onnx rồi từ .onnx chuyển qua model .xml rồi chuyển qua .blob chạy dưới thiết bị edge

B1 : chuyển file model .pt sang file .onnx 
B2 : activate env openvino 
B3 : chuyển file model .onnx sang .xml
  
cd /openvino/model-optimizer 
openvino ( activate openvino ) 
python3 mo_onnx.py --input_model /home/pcwork/ai/ftech/finger/CNN/myyeu.onnx --output_dir /home/pcwork/ai/ftech/finger/CNN/modelPytorch/openvino/ 
B4 : compile model .xml and .bin to file .blob 

 

compile ( export MYRIAD_COMLILE ) 
$MYRIAD_COMPILE -m ~/ai/ftech/finger/CNN/modelPytorch/openvino/myyeu.xml 

or 

$MYRIAD_COMPILE -m ~/ai/ftech/finger/CNN/modelPytorch/openvino/myyeu.xml -ip U8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

