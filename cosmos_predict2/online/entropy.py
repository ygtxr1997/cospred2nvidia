import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEntropyMonitor(nn.Module):
    """
    原型软分配熵（prototype soft-assignment entropy）
    - 用 K 个原型把连续嵌入 z 映射成 p(k|z)，计算离散熵 H(z)
    - 支持:
        * init_prototypes(...): 从源域嵌入初始化原型（kmeans 或随机）
        * set_src_baseline(...): 记录源域熵均值作为对齐基线
        * entropy(...): 返回当前 batch 的 (H_all, H_mean)
        * loss(...): 把 H_mean 拉回到源域基线附近（折叶/hinge），或最小化/最大化
    - 形状适配:
        输入 z 可为 (B,T,1,W,D) / (B,T,W,D) / (N,D)，内部会自动展平为 (N,D)
    """
    def __init__(self, k: int = 32, tau: float = 1.0, learnable_tau: bool = False, iters_kmeans: int = 10):
        super().__init__()
        self.k = k
        self.iters_kmeans = iters_kmeans

        # 原型和源域基线作为 buffer（不会参与梯度）
        self.register_buffer("prototypes", torch.empty(0))   # (K, D)
        self.register_buffer("src_H_mean", torch.tensor(float("nan")))  # 源域熵均值
        self.register_buffer("eps", torch.tensor(1e-12))

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))

    # ---- 公共接口 -----------------------------------------------------------
    @property
    def has_prototypes(self) -> bool:
        return self.prototypes.numel() > 0

    def begin_reservoir(self, reservoir_size: int):
        """开启蓄水池抽样式的分批收集。最后用 finalize_from_reservoir() 一次性做 kmeans。"""
        self._reservoir_size = int(reservoir_size)
        self.register_buffer("_reservoir_buf", torch.empty(0))  # (M, D) 动态创建
        self.register_buffer("_reservoir_n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_seen", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def add_source_batch(self, z_batch):
        """把一批源域嵌入加入蓄水池（Reservoir Sampling）。"""
        z = self._flatten(z_batch)  # (N, D)
        if z.numel() == 0: return
        if self._reservoir_buf.numel() == 0:
            # 首次创建缓冲区，确定 D
            D = z.size(-1)
            M = int(self._reservoir_size)
            device, dtype = z.device, z.dtype
            self._reservoir_buf = torch.empty((M, D), device=device, dtype=dtype)
            self._reservoir_n.zero_()
            self._seen.zero_()

        for i in range(z.size(0)):
            t = int(self._seen.item())
            if t < self._reservoir_buf.size(0):
                self._reservoir_buf[t].copy_(z[i])
                self._reservoir_n += 1
            else:
                # 以  M/(t+1) 的概率替换
                j = torch.randint(0, t + 1, (1,), device=z.device).item()
                if j < self._reservoir_buf.size(0):
                    self._reservoir_buf[j].copy_(z[i])
            self._seen += 1

    @torch.no_grad()
    def finalize_from_reservoir(self, method: str = "kmeans", seed: int = 0):
        """用蓄水池中的样本一次性初始化原型；然后你可以遍历一遍源域批次去累计 src_H_mean。"""
        if self._reservoir_buf.numel() == 0 or int(self._reservoir_n.item()) == 0:
            raise RuntimeError("Reservoir is empty. Call begin_reservoir() + add_source_batch() first.")
        M = int(self._reservoir_n.item())
        z = self._reservoir_buf[:M]
        self.init_prototypes(z, method=method, seed=seed)

    @torch.no_grad()
    def init_prototypes(self, z, method: str = "kmeans", subsample: int = 4096, seed: int = 0):
        """
        用一批(源域)嵌入初始化原型。z 形状任意，内部会展平为 (N,D)。
        method: "kmeans" 或 "random"
        subsample: 为了加速，可对大批量先子采样
        """
        if self.has_prototypes:
            return

        z = self._flatten(z)
        if z.numel() == 0:
            raise ValueError("Empty embeddings for init_prototypes.")
        # 子采样
        if z.size(0) > subsample:
            g = torch.Generator(device=z.device)
            g.manual_seed(seed)
            idx = torch.randperm(z.size(0), generator=g, device=z.device)[:subsample]
            z = z[idx]

        K = min(self.k, z.size(0))
        if method == "random":
            self.prototypes = z[torch.randperm(z.size(0), device=z.device)[:K]].clone()
            return

        # 简易 KMeans（固定步数，足够快）
        # 初始化：kmeans++ 的近似：先随机一个，再按距离加权抽样
        protos = self._kmeans_plus_plus(z, K, seed=seed)
        for _ in range(self.iters_kmeans):
            # 分配
            d2 = torch.cdist(z, protos, p=2) ** 2  # (N,K)
            assign = d2.argmin(dim=1)              # (N,)
            # 更新
            new_protos = []
            for k in range(K):
                mask = (assign == k)
                if mask.any():
                    new_protos.append(z[mask].mean(dim=0))
                else:
                    # 空簇回退：随机重启一个点
                    new_protos.append(z[torch.randint(0, z.size(0), (1,), device=z.device)].squeeze(0))
            protos = torch.stack(new_protos, dim=0)
        self.prototypes = protos

    @torch.no_grad()
    def set_src_baseline(self, z=None, H_mean: float | None = None):
        """
        设置源域熵均值基线。传 H_mean 或传一批源域嵌入 z（用当前原型计算）。
        """
        if H_mean is not None:
            self.src_H_mean = torch.tensor(float(H_mean), device=self._dev())
            return
        if z is None:
            raise ValueError("set_src_baseline: need z or H_mean.")
        _, Hm = self.entropy(z, no_grad=True)
        self.src_H_mean = Hm

    def entropy(self, z, no_grad: bool = False):
        """
        计算当前 batch 的（逐样本熵 H_all, 熵均值 H_mean）
        返回: H_all (N,), H_mean (标量张量)
        """
        if not self.has_prototypes:
            raise RuntimeError("Prototypes not initialized. Call init_prototypes(...) first.")
        z = self._flatten(z)  # (N,D)
        fn = torch.no_grad if no_grad else _IdentityContext
        with fn():
            p = self._soft_assign(z)       # (N,K)
            H_all = -(p * (p.clamp_min(self.eps)).log()).sum(dim=-1)  # (N,)
            H_mean = H_all.mean()
        return H_all, H_mean

    def loss(self, z, mode: str = "match", band: float = 0.1):
        """
        给出一个简单的 TTA 对齐损失：
        - mode="match": 把 H_mean 拉回源域基线 src_H_mean 的 ±band 区间（折叶/hinge）
        - mode="min":   直接最小化 H_mean（谨慎使用，可能坍塌）
        - mode="max":   最大化 H_mean（某些需要增熵的场景）
        返回: loss(标量), stats(dict)
        """
        _, Hm = self.entropy(z, no_grad=False)
        stats = {"H_mean": Hm.detach()}
        if mode == "match":
            if torch.isnan(self.src_H_mean):
                raise RuntimeError("src_H_mean is NaN. Call set_src_baseline(...) first.")
            lo = self.src_H_mean - band
            hi = self.src_H_mean + band
            loss = F.relu(Hm - hi) + F.relu(lo - Hm)
            stats["H_target"] = self.src_H_mean.detach()
            stats["band"] = torch.tensor(band, device=Hm.device)
        elif mode == "min":
            loss = Hm
        elif mode == "max":
            loss = -Hm
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return loss, stats

    # ---- 内部工具 -----------------------------------------------------------
    def _soft_assign(self, z):
        # z: (N,D), protos: (K,D)
        tau = torch.clamp(self.tau, min=1e-6)
        d2 = torch.cdist(z, self.prototypes, p=2) ** 2
        logits = -d2 / (2.0 * tau * tau)
        return torch.softmax(logits, dim=-1)

    def _flatten(self, z):
        # 支持 (B,T,1,W,D) / (B,T,W,D) / (N,D)
        if z.dim() == 5:   # (B,T,1,W,D) or (B,T,W,D)
            z = z.squeeze(2) if z.size(2) == 1 else z
        if z.dim() == 4:   # (B,T,W,D)
            # z = z.reshape(-1, z.size(-1))
            z = z.reshape(z.size(0), -1)
        elif z.dim() == 2: # (N,D)
            pass
        else:
            raise ValueError(f"Unexpected z shape: {tuple(z.shape)}")
        return z

    def _dev(self):
        return self.tau.device

    @torch.no_grad()
    def _kmeans_plus_plus(self, z, K, seed: int = 0):
        g = torch.Generator(device=z.device)
        g.manual_seed(seed)
        N = z.size(0)
        # 选第一个中心
        idx0 = torch.randint(0, N, (1,), generator=g, device=z.device)
        centers = [z[idx0.item()]]
        # 迭代选其余中心
        for _ in range(1, K):
            d2 = torch.cdist(z, torch.stack(centers, 0), p=2).pow(2).min(dim=1).values
            probs = (d2 + 1e-12) / (d2.sum() + 1e-12)
            idx = torch.multinomial(probs, 1, generator=g)
            centers.append(z[idx.item()])
        return torch.stack(centers, dim=0)

class _IdentityContext:
    """让 with _IdentityContext(): 等价于不包 no_grad 的空上下文。"""
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


if __name__ == "__main__":
    # 1) 模块化放到 __init__ 里
    # self.action_entropy = ActionEntropyMonitor(k=32, tau=1.0, learnable_tau=False)

    # 2) 源域一次性初始化（训练/校准阶段做一次即可）
    # if not self.action_entropy.has_prototypes:
    #     self.action_entropy.init_prototypes(action_emb_B_T_1_W_D, method="kmeans")
    #     self.action_entropy.set_src_baseline(action_emb_B_T_1_W_D)  # 记录源域熵均值

    # 3) 推理/测试期（TTA）
    # ——统计 & 日志
    with torch.no_grad():
        _, Hm = self.action_entropy.entropy(action_emb_B_T_1_W_D, no_grad=True)
        self.affline_scale_log_info["action_entropy_proto"] = Hm

    # ——若需要把它作为 TTA 的一部分损失（去掉 no_grad，让梯度回传到你解冻的 LN/LoRA）
    # loss_align, stats = self.action_entropy.loss(action_emb_B_T_1_W_D, mode="match", band=0.1)
    # total_loss = total_loss + lambda_e * loss_align