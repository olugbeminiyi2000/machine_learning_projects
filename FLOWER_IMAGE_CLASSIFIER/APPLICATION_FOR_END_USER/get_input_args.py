import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir1', type = str, default = 'flowers/train',
                        help = 'path to the folder that contains the train data folders of images of each flower categories')
    
    parser.add_argument('--dir2', type = str, default = 'flowers/valid',
                        help = 'path to the folder that contains the cross_validation data folders of images of each flower categories')
    
    parser.add_argument('--dir3', type = str, default = 'flowers/test',
                        help = 'path to the folder that contains the test data folders of images of each flower categories')
    
    parser.add_argument('--flower_input', type = str, default = 'flowers/test/100',
                        help = 'give me a folder leading to an image ')
    
    parser.add_argument('--epochs', type = int, default = 2,
                        help = 'put the number of epochs as an integer')
    
    parser.add_argument('--hu', type = int, default = 500,
                        help = 'put the number of hidden units as an integer')
    
    parser.add_argument('--lr1', type = float, default = 0.005 ,
                        help = 'put the first learning rate as an float')
    
    parser.add_argument('--lr2', type = float, default = 0.005 ,
                        help = 'put the second learning rate as an float')
    
    parser.add_argument('--lr3', type = float, default = 0.002 ,
                        help = 'put the second learning rate as an float')
    
    
    parser.add_argument('--gpu', type = str, default = 'cuda', choices = ['cpu'],
                        help = 'do you want to train with gpu called cuda or cpu')
    
    
    parser.add_argument('--arch', default = 'resnet152',choices = ['densenet121','densenet169'],
                        help = 'name of your choice of CNN model btwn densenet121|densenet169 or resnet152')
    
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth',
                        help = 'put your saved checkpoint file with saved model and state_dict')
    
    parser.add_argument('--topk', type = int, default = 5,
                        help = 'put the number of top probabilities as an integer')
    
    parser.add_argument('--category_name', type = str, default = 'cat_to_name.json',
                        help = 'put the label category name  as an file')
    
    
    in_args = parser.parse_args()
    
    return in_args
   