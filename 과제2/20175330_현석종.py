import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
         ])

    test_set = torchvision.datasets.ImageNet(root="/home/ssac16/차량지능기초/imagenet", transform=transform, split='val')
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=4)

    alexnet_model = torchvision.models.alexnet(pretrained=True).to(device)
    vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
    resnet18_model = torchvision.models.resnet18(pretrained=True).to(device)
    googlenet_model = torchvision.models.googlenet(pretrained=True).to(device)
    
    model_name_list=["AlexNet","VGG16","ResNet18","GoogLeNet"]
    model_list=[]
    
    model_list.append(alexnet_model.eval())
    model_list.append(vgg16_model.eval())
    model_list.append(resnet18_model.eval())
    model_list.append(googlenet_model.eval())

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
   
    
    for i,j in enumerate(model_name_list):
        model=model_list[i]
        print("<<<",j,">>>")
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader):

                images = images.to(device)      # [100, 3, 224, 224]
                labels = labels.to(device)      # [100]
                outputs = model(images)

                # ------------------------------------------------------------------------------
                # rank 1
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct_top1 += (pred == labels).sum().item()

                # ------------------------------------------------------------------------------
                # rank 5
                _, rank5 = outputs.topk(5, 1, True, True)
                rank5 = rank5.t()
                correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

                # ------------------------------------------------------------------------------
                for k in range(6):
                    correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

                correct_top5 += correct_k.item()

    #             print("step : {} / {}".format(idx + 1, len(test_set)/int(labels.size(0))))
    #             print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    #             print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))

        print(" top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
        print(" top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))