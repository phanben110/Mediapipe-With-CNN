import torch  
from torch import nn 
from BEN_modelCNN import CNN 

class convertModel(): 
    def __init__(self,name , inputShape , batchSize, debug = True ): 
        self.name = name 
        self.debug = debug 
        self.inputShape = inputShape 
        self.batchSize = batchSize 

    def ptToOnnx(self, model):
        x = torch.randn(self.batchSize, self.inputShape[0],self.inputShape[1],self.inputShape[2], requires_grad=True)
        torch_out = model(x)

        # Export the model
        torch.onnx.export(model,               # model being run
                          x,                         # model input (or a tuple for multiple inputs)
                          self.name,   # where to save the model (can be a file or file-like object)
                          #export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=10,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the model's input names
                          output_names = ['output'], # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


if __name__ == "__main__": 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    path = '/home/pcwork/ai/ftech/finger/CNN/modelPytorch/model12ClassSize50.pt'
    pathOnnx = '/home/pcwork/ai/ftech/finger/CNN/modelPytorch/model12ClassSize50.onnx'

    numClasses = 12  
    print ("init model ...") 
    model = CNN(numClasses ) 
    model.load_state_dict( torch.load(path)) 
    model.eval() 
    print ( model ) 
    
    #this line to convert model from .pt to .onnx  
    print ("start conver model ....")  
    convert = convertModel( pathOnnx, [1,50,50] , 1 ) 
    convert.ptToOnnx(model)  
    print ("model Pytorch convert format onnx done.")  



