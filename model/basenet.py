from model.densenet import DenseNet
from model.resnet import ResNet_ft

def get_model(configs):

    if "resnet" in configs["model"]:
        from model.resnet import ResNet
        pretrained=False
        if 'pretrained' in configs['model']:
            configs['model']=configs['model'][11:]
            pretrained=True
            num_classes=configs['num_classes']
            configs['num_classes']=1000
        model=ResNet(configs)

    elif "densenet" in configs["model"]:
        from model.densenet import densenet
        model=densenet(configs)
    else:
        print("No model")
        raise NotImplementedError

    
    if 'pre' not in configs['model'] and 'resnet' in configs['model'] and pretrained:
        configs['num_classes']=num_classes
        from torch.hub import load_state_dict_from_url
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
        }
        state_dict = load_state_dict_from_url(model_urls[configs['model']])
        model.load_state_dict(state_dict)
        if configs['mode']!='eval':
            model.fc = model.fc.__class__(
                model.fc.weight.size(1), configs["num_classes"]
            )
            configs['model']='pretrained_'+configs['model']


    if configs['mode']=='train' and configs['train_mode'] in ['icarl','eeil']:
        if isinstance(model, ResNet):
            model=ResNet_ft(configs)
    return model
