
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, device):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16().to(device))
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self, device, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG16().to(device))
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    # @ staticmethod
    # def criterion(feature1, feature2):
    #     # cosine
    #     # feature1 N*C*W*H， feature2 N*C*W*H
    #     feature1 = feature1.view(feature1.shape[0], -1)
    #     feature2 = feature2.view(feature2.shape[0], -1)
    #     feature1 = F.normalize(feature1)  # L2
    #     feature2 = F.normalize(feature2)
    #     distance = 1.-torch.mean((torch.sum(feature1*feature2, dim=1) + 1.)*0.5)
    #     return distance

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return content_loss

class SSIM:
    def __init__(self, device, size=11, sigma=1.5):
        self.window = self._special_gauss(size, sigma).to(device)
        self.window.requires_grad = False

    @staticmethod
    def _special_gauss(size, sigma):
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)
        y = torch.tensor(y_data, dtype=torch.float)
        x = torch.tensor(x_data, dtype=torch.float)
        g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))).permute(2, 3, 0, 1)
        return g / torch.sum(g)

    def cal_one_channel(self, img1, img2, mask=None):
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image; L=1 -> image [0,1]
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = F.conv2d(img1, self.window, padding=0)
        mu2 = F.conv2d(img2, self.window, padding=0)
        # print(self.window)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=0) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=0) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))
        if mask is not None: # inside mask = 0
            mask = F.interpolate(mask, size=[ssim_map.shape[2], ssim_map.shape[3]])
            mean_metric = torch.sum(torch.abs(ssim_map*mask)) / (torch.sum(mask)+1e-3)
        else:
            mean_metric = torch.mean(ssim_map)
        return mean_metric

    def __call__(self, img1, img2, mask=None):
        # img1 [-1,1]
        # img2 [-1,1]
        # mask inside=0
        img1 = (img1 + 1) * 0.5
        img2 = (img2 + 1) * 0.5
        if img1.shape[1] == 3:
            loss1 = self.cal_one_channel(img1[:, 0:1, :, :], img2[:, 0:1, :, :], mask)
            loss2 = self.cal_one_channel(img1[:, 1:2, :, :], img2[:, 1:2, :, :], mask)
            loss3 = self.cal_one_channel(img1[:, 2:3, :, :], img2[:, 2:3, :, :], mask)
            return (loss1+loss2+loss3)/3.

class MaskLloss(nn.Module):
    def __init__(self, tag=1):
        super().__init__()
        self.tag = tag

    def forward(self, x, y, mask):
        # inside mask = 0
        # x [-1,1]
        # y [-1,1]
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        if self.tag == 1:
            return torch.sum(torch.abs((x-y)*mask)) / (torch.sum(mask)+1e-3)
        if self.tag == 2:
            return torch.sum(torch.abs((x**2-y**2)*mask)) / (torch.sum(mask)+1e-3)

class SpatialConsistencyLoss(nn.Module):
    """
    class SpatialConsistencyLoss(nn.Module)
    计算 空间一致性损失  即梯度相似度损失 ZeroDEC原文代码
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """
    def __init__(self, device):
        super(SpatialConsistencyLoss, self).__init__()
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).\
            to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).\
            to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).\
            to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).\
            to(device).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, org, enhance):
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        return E.mean()
