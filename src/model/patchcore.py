import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import numpy as np
from pathlib import Path
import pickle
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors

from src.conf.config import Config
from src.model.setup import model_setup
from src.dataset.util.plot import save_anomaly_map, inv_transform
from src.model.utils.sampling_methods.kcenter_greedy import kCenterGreedy


def embedding_concat(x, y):
    """
    yをxに合わせてアップサンプリング
    Args:
        x (_type_): torch.Size([2, 512, 32, 32])
        y (_type_): torch.Size([2, 1024, 16, 16])

    Returns:
        _type_: _description_
    """
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2) # 2
    
    # torch.Size([2, 512, 32, 32]) => torch.Size([2, 2048, 256])
    # (batch_size, c x kernel_size(2x2), *)
    # * = ((H1 - (kernel_size:2 - 1) - 1)/ stride(2) + 1)
    #     x ((W1 - (kernel_size:2 - 1) - 1)/ stride(2) + 1)
    # 16 x 16 = 256
    # kernel sizeと strideが同じ時、ViTのバッチ分割に相当
    # 1回の操作で 32/2 * 32 / 2　 = 256の値を取得
    # それを1ch 横に2回、縦に2回 行い、それが、全チャンネルに実行される
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s) # ([2, 2048, 256])
    x = x.view(B, C1, -1, H2, W2) # 2, 512, 4, 16, 16 # 各チャネルごとの値は、 4, 16, 16 # 縦横に1dilationで取得したパッチ画像(16, 16)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2) # 2, 1536, 4, 16, 16 # 
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2) # 2, 6144, 256
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s) # 2, 1536, 32, 32 # 小さい方の画像はupサンプリングしたものに相当
    return z

def embedding_concat(x, y):
    """
    yをxに合わせてアップサンプリング
    Args:
        x (_type_): torch.Size([2, 512, 32, 32])
        y (_type_): torch.Size([2, 1024, 16, 16])

    Returns:
        _type_: _description_
    """
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2) # 2
    y = F.interpolate(y, scale_factor=s, mode="bilinear")
    z = torch.cat([x, y], dim=1)
    return z
    

def reshape_embedding(embedding):
    """チャンネル数分の特徴量を32x32xbatchの位置に対して作成

    Args:
        embedding (_type_): _description_

    Returns:
        _type_: _description_
    """
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

def log_dir_prepare(cfg:Config):
    log_dir = Path(cfg.plot.save_dir).absolute()
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir / cfg.ml.dataset 
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir / Path(cfg.imdata.img_dir).stem 
    log_dir.mkdir(exist_ok=True)
    embedding_dir = log_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    plot_dir = log_dir / "plot"
    plot_dir.mkdir(exist_ok=True)
    return log_dir, embedding_dir, plot_dir

class PatchCoreModelModule(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super(PatchCoreModelModule, self).__init__()
        self.cfg = cfg
        
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.model = model_setup(cfg)
        
        # モデルの学習は必要ない(pretrainの情報を使用する)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # wide resnet50のlayer2とlayer3の出力をForward Hookを使用して取得
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.init_results_list()
        

        
    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []
    
    def forward(self, x_t):
        # 特徴量を出力
        # length:2 
        # y[0]:torch.Size([2, 512, 32, 32]) 
        # y[1]torch.Size([2, 1024, 16, 16])
        
        self.init_features()
        _ = self.model(x_t)
        return self.features
    
    def configure_optimizers(self):
        # 最適化は行わない
        return None

    def on_train_start(self):
        self.log_dir, self.embedding_dir, self.plot_dir = log_dir_prepare(self.cfg)
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_list = []
        
    def on_test_start(self):
        self.init_results_list()
        self.log_dir, self.embedding_dir, self.plot_dir = log_dir_prepare(self.cfg)

    
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature.cpu()))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))
    
    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list) # 2048 (1536,) 各位置の特徴がchannel分ある
        # Random projection
        # 高速化のため、使用する次元をランダムサンプリングで選び次元削減する
        # Johnson-Lindenstrauss lemmaに則って低次元に射影するランダムな行列を計算
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        # サンプル数を減らす高速化。
        print(total_embeddings.shape)
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.cfg.model.patchcore_coreset_sampling_rate))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        with open(self.embedding_dir / 'embedding.pickle', 'wb') as f:
            pickle.dump(self.embedding_coreset, f)

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        self.embedding_coreset = pickle.load(open(self.embedding_dir / 'embedding.pickle', 'rb'))
        x, gt, label, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_.cpu())))
        # NN
        nbrs = NearestNeighbors(n_neighbors=self.cfg.model.patchcore_n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        score_patches, _ = nbrs.kneighbors(embedding_test)
        anomaly_map = score_patches[:,0].reshape((self.cfg.plot.plot_img_size, self.cfg.plot.plot_img_size))
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (self.cfg.imdata.input_size, self.cfg.imdata.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        
        # save images
        x = inv_transform(x, self.cfg.imdata.inv_img_mean, self.cfg.imdata.inv_img_std)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        file_name = f"{batch_idx:05d}"
        save_anomaly_map(self.plot_dir, anomaly_map_resized_blur, input_x, gt_np*255, file_name, x_type[0])
    
    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        
if __name__ == "__main__":
    from src.dataset.loader import data_loader_setup
    cfg = Config()
    cfg.ml.phase = "test"
    cfg.ml.batch_size = 2
    
    dataloader = data_loader_setup(cfg)
    model = PatchCoreModelModule(cfg)
    
    for i, batch in enumerate(dataloader):
        bimg, bgt, label, imtype = batch
        y = model(bimg)
        print(len(y), y[0].shape, y[1].shape)
        print("-"*10)
        embeddings = []
        for feature in y:
            # torch.Size([2, 512, 32, 32])
            # torch.Size([2, 1024, 16, 16])
            m = torch.nn.AvgPool2d(3, 1, 1)
            feature = m(feature)
            print(feature.shape)
            embeddings.append(feature)
        embedding = embedding_concat(embeddings[0], embeddings[1])
        print(embedding.shape) # torch.Size([2, 1536, 32, 32])
        x = reshape_embedding(np.array(embedding)) # 2048 (1536,)
        print(len(x), x[0].shape)
        break