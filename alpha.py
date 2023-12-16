import tools
import my_models
import process_data_simple1


    
if __name__ == '__main__':
    lr,num_epochs,batch_size,weight_decay = 0.01,150,25,0
    step_size,gamma = 100,0.65
    net = my_models.multi_perceptrons(input=57,output=10)
    net.apply(tools.weight_init)

    dataset = process_data_simple1.GTZANDataset(r"..\data\music\features_3_sec.csv")
    train_dataset = dataset(train="True")
    test_dataset = dataset(train="False")

    train_iter,test_iter = tools.data_iter(train_dataset,test_dataset,batch_size=batch_size)
    tools.train(net,train_iter,test_iter,num_epochs,tools.loss(),tools.optimize(net,lr,weight_decay),'cuda')
    
    tools.save_model(net,r"..\models\alpha.pth")
