"""Self-contained, deterministic Python coding exercises for source questions Q82–Q101.

Private reference implementations and executable checks stay server-side. Each call
returns independent dictionaries so editor/API clients cannot mutate the catalog.
"""
from copy import deepcopy
from textwrap import dedent


def _code(value):
    return dedent(value).strip() + "\n"


def _case(name, code):
    return {"name": name, "code": _code(code)}


def _exercise(number, title, description, starter, reference, tests, hints, advanced=False):
    return {
        "id": f"C{number}", "source_id": f"Q{number}", "number": number,
        "title": title, "chapter": "代码", "difficulty": "进阶" if advanced else "基础",
        "description": dedent(description).strip(), "starter": _code(starter),
        "hints": hints, "reference": _code(reference), "tests": tests,
        "test_names": [t["name"] for t in tests],
    }


_CATALOG = [
    _exercise(82, "AdaLN：将时间条件注入残差块", """
    实现 `timestep_embedding(t, dim)` 和 `adaln(x, condition, weight, bias, eps=1e-5)`。
    `t` 是 `[B]` 浮点时间，`dim` 为正偶数；前半维为 cos，后半维为 sin，
    频率 `exp(-log(10000)*arange(dim/2)/(dim/2))`。
    `x:[B,T,D]`，`condition:[B,C]`，`weight:[3D,C]`，`bias:[3D]`。
    先用 `condition @ weight.T + bias` 按 scale、shift、gate 顺序分三份，
    对 x 最后一维做无可学习仿射的 LayerNorm（总体方差），返回
    `x + gate[:,None,:] * (LN(x)*(1+scale[:,None,:])+shift[:,None,:])`。
    可将时间嵌入作为 condition；本题是 DiT 中的简化条件残差算子，不包含完整 DiT。
    例如 gate 全零时，输出严格等于 x。输入为 CPU 浮点张量，保持 dtype、形状与梯度。
    """, """
    import torch
    def timestep_embedding(t, dim):
        raise NotImplementedError
    def adaln(x, condition, weight, bias, eps=1e-5):
        raise NotImplementedError
    """, """
    import math
    import torch
    def timestep_embedding(t, dim):
        if dim <= 0 or dim % 2:
            raise ValueError('dim must be a positive even integer')
        frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2, dtype=t.dtype, device=t.device) / (dim // 2))
        phase = t[:, None] * frequency[None, :]
        return torch.cat((phase.cos(), phase.sin()), dim=-1)
    def adaln(x, condition, weight, bias, eps=1e-5):
        scale, shift, gate = (condition @ weight.T + bias).chunk(3, dim=-1)
        norm = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + eps)
        return x + gate[:, None, :] * (norm * (1 + scale[:, None, :]) + shift[:, None, :])
    """, [
        _case("时间嵌入的频率与布局", """
        t = torch.tensor([0.,1.],dtype=torch.float64)
        y = candidate.timestep_embedding(t,4)
        expected = torch.tensor([[1.,1.,0.,0.],[math.cos(1),math.cos(.01),math.sin(1),math.sin(.01)]],dtype=t.dtype)
        assert y.shape == (2,4) and y.dtype == t.dtype, '嵌入形状或dtype错误'
        assert torch.allclose(y,expected), 'cos/sin顺序或频率错误'
        """),
        _case("零门控与样本条件广播", """
        x = torch.tensor([[[1.,3.],[5.,7.]],[[4.,2.],[9.,1.]]])
        c = torch.ones(2,1); w = torch.zeros(6,1); b = torch.zeros(6)
        assert torch.equal(candidate.adaln(x,c,w,b),x), '零gate必须保持残差'
        b[2:4] = torch.tensor([2.,-1.]); b[4:] = 1
        y = candidate.adaln(x,c,w,b)
        expected = x + torch.nn.functional.layer_norm(x,(2,)) + torch.tensor([2.,-1.])
        assert y.shape == x.shape and torch.allclose(y,expected), 'scale/shift/gate切分错误'
        """),
        _case("条件投影数值与梯度", """
        x=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
        c=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
        w=torch.randn(12,5,dtype=torch.float64,requires_grad=True); b=torch.randn(12,dtype=torch.float64,requires_grad=True)
        s,h,g=(c@w.T+b).chunk(3,-1)
        expected=x+g[:,None]*(torch.nn.functional.layer_norm(x,(4,))*(1+s[:,None])+h[:,None])
        actual=candidate.adaln(x,c,w,b)
        assert torch.allclose(actual,expected,atol=1e-10), 'AdaLN数值不匹配'
        ga=torch.autograd.grad(actual.square().sum(),(x,c,w,b),retain_graph=True)
        ge=torch.autograd.grad(expected.square().sum(),(x,c,w,b))
        assert all(torch.allclose(a,e,atol=1e-8) for a,e in zip(ga,ge)), 'AdaLN梯度路径错误'
        """),
    ], ["先明确投影输出三段的顺序。", "LayerNorm沿D计算，条件需要增加T广播维。"], True),
    _exercise(83, "Flow Matching：插值、掩码损失与 Euler 采样", """
    实现三个函数：`flow_batch(data, noise, t)` 返回 `(x_t, velocity)`，
    `flow_loss(model, data, noise, t, mask=None)` 返回标量，
    `sample_euler(model, noise, steps)` 返回动作张量。
    data/noise 为相同 `[B,H,D]`，t 为 `[B]`；约定 `x_t=(1-t)*noise+t*data`，
    目标速度为 `data-noise`。model 接收 `(x_t,t)`，输出同形状速度。
    loss 是有效元素的平均平方误差；bool mask 为 `[B,H]`，全空掩码返回可反传的0。
    Euler 从 t=0 积分至1，每一步先调用 `model(x, full([B],i/steps))`，再加速度/steps。
    steps 必须为正整数，否则 ValueError。例如恒定速度2、初值0、任意正步数得到2。
    不要求拟合真实机器人数据或证明生成质量，禁止内置ODE求解器。
    """, """
    import torch
    def flow_batch(data, noise, t):
        raise NotImplementedError
    def flow_loss(model, data, noise, t, mask=None):
        raise NotImplementedError
    def sample_euler(model, noise, steps):
        raise NotImplementedError
    """, """
    import torch
    def flow_batch(data, noise, t):
        time = t.reshape(-1, 1, 1)
        return (1-time)*noise + time*data, data-noise
    def flow_loss(model, data, noise, t, mask=None):
        x, target = flow_batch(data, noise, t)
        error = (model(x,t)-target).square()
        if mask is None:
            return error.mean()
        valid = mask.unsqueeze(-1).expand_as(error)
        return error.masked_fill(~valid,0).sum() / valid.sum().clamp_min(1)
    def sample_euler(model, noise, steps):
        if not isinstance(steps,int) or isinstance(steps,bool) or steps <= 0:
            raise ValueError('steps must be a positive integer')
        x = noise.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i*dt, dtype=x.dtype, device=x.device)
            x = x + dt * model(x, t)
        return x
    """, [
        _case("插值端点及样本广播", """
        data=torch.arange(12.).reshape(2,3,2); noise=-torch.ones_like(data)
        x,v=candidate.flow_batch(data,noise,torch.tensor([0.,1.]))
        assert torch.equal(x[0],noise[0]) and torch.equal(x[1],data[1]), 't方向或广播错误'
        assert torch.equal(v,data-noise), '目标速度应指向数据'
        """),
        _case("掩码归一化、空掩码和梯度", """
        data=torch.ones(2,2,3); noise=torch.zeros_like(data); t=torch.tensor([.2,.8])
        p=torch.tensor(2.,requires_grad=True)
        model=lambda x,t: p*torch.ones_like(x)
        mask=torch.tensor([[True,False],[False,False]])
        loss=candidate.flow_loss(model,data,noise,t,mask)
        assert loss.ndim == 0 and torch.allclose(loss,torch.tensor(1.)), '按有效元素归一化'
        loss.backward(); assert torch.allclose(p.grad,torch.tensor(2.)), '模型参数梯度错误'
        p.grad=None
        empty=candidate.flow_loss(model,data,noise,t,torch.zeros_like(mask))
        assert empty.item() == 0 and torch.isfinite(empty), '空mask应返回0'
        empty.backward(); assert p.grad.item() == 0, '空mask仍需有效梯度路径'
        """),
        _case("Euler步长、时间和输入不变性", """
        start=torch.zeros(2,2,1,dtype=torch.float64); times=[]
        def velocity(x,t):
            times.append(t.clone())
            return 2*torch.ones_like(x)
        out=candidate.sample_euler(velocity,start,4)
        assert torch.allclose(out,torch.full_like(start,2.)) and torch.equal(start,torch.zeros_like(start)), '积分方向或输入修改错误'
        assert len(times)==4 and torch.equal(times[2],torch.full((2,),.5,dtype=start.dtype)), '时间网格错误'
        out=candidate.sample_euler(lambda x,t:t[:,None,None].expand_as(x),start,4)
        assert torch.allclose(out,torch.full_like(out,.375)), '应使用左端点Euler'
        try: candidate.sample_euler(velocity,start,0)
        except ValueError: pass
        else: assert False, 'steps=0应抛ValueError'
        """),
    ], ["必须先统一时间方向。", "mask的分母要乘动作维度D。", "Euler每步使用当前状态与当前时间。"], True),
    _exercise(84, "PPO：GAE 与裁剪策略目标", """
    将训练伪代码中可独立验证的核心实现为 `gae(rewards, values, dones, last_value, gamma=.99, lam=.95)`
    和 `ppo_loss(new_logp, old_logp, advantage, clip_eps=.2, mask=None)`。
    rewards/values/dones 是 `[T,B]`，last_value 是 `[B]`，done=1 表示该转移结束episode；
    delta=`r+gamma*(1-done)*next_value-value`，逆序累积 GAE，返回 `(advantage, advantage+values)`。
    策略loss为负的 `min(ratio*A, clamp(ratio,1-eps,1+eps)*A)` 的有效位置均值；
    ratio=`exp(new_logp-old_logp)`，布尔mask同形，全空返回可反传0。
    例：单步终止、r=2、V=.5时adv=1.5，不使用last_value。
    输入有限浮点CPU张量；T>=1；不要求标准化优势。
    本题只验证优势估计和策略目标，不实现环境rollout、Critic更新，不证明训练收敛。
    """, """
    import torch
    def gae(rewards, values, dones, last_value, gamma=.99, lam=.95):
        raise NotImplementedError
    def ppo_loss(new_logp, old_logp, advantage, clip_eps=.2, mask=None):
        raise NotImplementedError
    """, """
    import torch
    def gae(rewards, values, dones, last_value, gamma=.99, lam=.95):
        result = []
        carry = torch.zeros_like(last_value)
        for t in range(len(rewards)-1,-1,-1):
            next_value = last_value if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + gamma*(1-dones[t])*next_value-values[t]
            carry = delta + gamma * lam * (1 - dones[t]) * carry
            result.append(carry)
        advantage = torch.stack(result[::-1])
        return advantage, advantage+values
    def ppo_loss(new_logp, old_logp, advantage, clip_eps=.2, mask=None):
        ratio = (new_logp-old_logp).exp()
        loss = -torch.minimum(ratio*advantage,ratio.clamp(1-clip_eps,1+clip_eps)*advantage)
        if mask is None:
            return loss.mean()
        return loss.masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)
    """, [
        _case("终止状态阻断bootstrap与跨episode传播", """
        r=torch.tensor([[1.],[2.],[3.]]); v=torch.tensor([[.5],[.7],[.9]]); d=torch.tensor([[0.],[1.],[0.]])
        a,ret=candidate.gae(r,v,d,torch.tensor([5.]),gamma=.9,lam=.8)
        expected=torch.tensor([[2.066],[1.3],[6.6]])
        assert a.shape == r.shape and torch.allclose(a,expected,atol=1e-6), 'done应同时阻断bootstrap和GAE传播'
        assert torch.allclose(ret,a+v), 'return应为advantage+value'
        """),
        _case("批次隔离与lambda零退化", """
        r=torch.tensor([[2.,4.],[3.,5.]]); v=torch.tensor([[1.,2.],[2.,3.]]); d=torch.tensor([[1.,0.],[0.,1.]])
        a,_=candidate.gae(r,v,d,torch.tensor([4.,99.]),gamma=.5,lam=0.)
        assert torch.allclose(a,torch.tensor([[1.,3.5],[3.,2.]])), 'lambda=0应为各样本TD误差'
        """),
        _case("正负优势clip、mask及梯度", """
        new=torch.tensor([math.log(1.5),math.log(.5),0.],requires_grad=True)
        old=torch.zeros(3); adv=torch.tensor([2.,-2.,99.]); mask=torch.tensor([True,True,False])
        loss=candidate.ppo_loss(new,old,adv,.2,mask)
        assert torch.allclose(loss,torch.tensor(-.4),atol=1e-6), '负优势也要取minimum'
        loss.backward(); assert torch.allclose(new.grad,torch.zeros(3)), '被clip和mask位置梯度应为0'
        x=torch.tensor([.1],requires_grad=True)
        l=candidate.ppo_loss(x,torch.zeros(1),torch.ones(1)); l.backward()
        assert torch.allclose(x.grad,-x.detach().exp()), '未clip分支梯度错误'
        z=candidate.ppo_loss(x,torch.zeros(1),torch.ones(1),mask=torch.tensor([False]))
        assert z.item()==0 and z.requires_grad, '空mask必须返回可反传0'
        """),
    ], ["done决定bootstrap和递推是否跨越终止边界。", "PPO对负优势也取两个目标的最小值。"], True),
    _exercise(85, "GRPO：组内优势与 KL 约束目标", """
    实现 `group_advantages(rewards, eps=1e-8)` 和
    `grpo_loss(new_logp, old_logp, advantage, ref_logp, clip_eps=.2, beta=.04, mask=None)`。
    rewards 为 `[B,G]`，G>=1；每行使用总体标准差（unbiased=False）归一化：
    `(r-mean)/(std+eps)`，同奖励组应为0。
    其余输入同形，按token/动作位置给定，不在函数内广播组优势。
    ratio=`exp(new-old)`，KL估计=`exp(ref-new)-(ref-new)-1`，
    loss=`mean_valid(-min(ratio*A,clip(ratio)*A)+beta*KL)`。
    bool mask同形，空mask返回可反传0。参考logp和旧logp由调用者固定。
    例：奖励[1,3]归一化近似[-1,1]。仅测试组内优势和目标计算；
    不实现组采样、环境交互或训练循环，不证明训练收敛。
    """, """
    import torch
    def group_advantages(rewards, eps=1e-8):
        raise NotImplementedError
    def grpo_loss(new_logp, old_logp, advantage, ref_logp, clip_eps=.2, beta=.04, mask=None):
        raise NotImplementedError
    """, """
    import torch
    def group_advantages(rewards, eps=1e-8):
        return (rewards-rewards.mean(-1,keepdim=True))/(rewards.std(-1,keepdim=True,unbiased=False)+eps)
    def grpo_loss(new_logp, old_logp, advantage, ref_logp, clip_eps=.2, beta=.04, mask=None):
        ratio = (new_logp-old_logp).exp()
        objective = torch.minimum(ratio*advantage,ratio.clamp(1-clip_eps,1+clip_eps)*advantage)
        log_ratio = ref_logp-new_logp
        loss = -objective + beta*(log_ratio.exp()-log_ratio-1)
        if mask is None:
            return loss.mean()
        return loss.masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)
    """, [
        _case("组内总体标准差与常数组", """
        r=torch.tensor([[1.,3.],[7.,7.]],dtype=torch.float64)
        a=candidate.group_advantages(r)
        assert torch.allclose(a,torch.tensor([[-1.,1.],[0.,0.]],dtype=r.dtype),atol=1e-7), '须使用每组总体标准差'
        assert torch.isfinite(candidate.group_advantages(torch.ones(3,1))).all(), '单元素组不可产生NaN'
        """),
        _case("clip与KL数值", """
        n=torch.tensor([math.log(1.5),math.log(.5)],dtype=torch.float64)
        old=torch.zeros_like(n); a=torch.tensor([2.,-2.],dtype=n.dtype); ref=torch.tensor([.2,-.1],dtype=n.dtype)
        kl=torch.exp(ref-n)-(ref-n)-1
        expected=torch.tensor(-.4,dtype=n.dtype)+.3*kl.mean()
        out=candidate.grpo_loss(n,old,a,ref,beta=.3)
        assert out.ndim==0 and torch.allclose(out,expected), 'GRPO裁剪或KL方向错误'
        """),
        _case("有效token归一化与梯度", """
        n=torch.tensor([.1,.3],dtype=torch.float64,requires_grad=True); old=torch.zeros_like(n); a=torch.tensor([1.,9.],dtype=n.dtype); ref=torch.zeros_like(n)
        mask=torch.tensor([True,False])
        out=candidate.grpo_loss(n,old,a,ref,beta=.2,mask=mask)
        out.backward(); expected=-math.exp(.1)+.2*(1-math.exp(-.1))
        assert abs(n.grad[0].item()-expected)<1e-9 and n.grad[1].item()==0, 'KL梯度或mask错误'
        z=candidate.grpo_loss(n,old,a,ref,mask=torch.zeros(2,dtype=torch.bool))
        assert z.item()==0 and z.requires_grad, '空mask须返回可反传0'
        """),
    ], ["组内标准差不是样本无偏估计。", "KL估计中log_ratio为ref减new。"], True),
    _exercise(86, "单头 Self-Attention 与安全掩码", """
    实现 `self_attention(x, wq, wk, wv, mask=None)`，x为 `[B,T,D]`，
    投影矩阵均 `[D,D]`，约定 `Q=x@wq`（不转置权重），K/V类似。
    输出 `softmax(Q@K.transpose(-2,-1)/sqrt(D))@V`，形状 `[B,T,D]`。
    mask是可广播到 `[B,T,T]` 的bool张量，True允许关注；全被屏蔽的查询行输出0。
    例：Q/K全零时，对可见的V均匀平均。保留输入和投影参数梯度，输入有限，B,T,D>=1。
    禁止直接调用内置attention算子；允许使用matmul、softmax等基础张量算子。
    """, """
    import torch
    def self_attention(x, wq, wk, wv, mask=None):
        raise NotImplementedError
    """, """
    import math
    import torch
    def self_attention(x, wq, wk, wv, mask=None):
        q,k,v = x@wq,x@wk,x@wv
        scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        safe_scores = torch.where(torch.isneginf(scores).all(-1,keepdim=True),torch.zeros_like(scores),scores)
        weights=torch.softmax(safe_scores,-1)
        if mask is not None:
            weights=weights.masked_fill(~mask,0)
        return weights@v
    """, [
        _case("投影、缩放与输出形状", """
        x=torch.randn(2,3,4,dtype=torch.float64); w=[torch.randn(4,4,dtype=x.dtype) for _ in range(3)]
        q,k,v=[x@a for a in w]
        expected=(q@k.transpose(-2,-1)/2).softmax(-1)@v
        actual=candidate.self_attention(x,*w)
        assert actual.shape == x.shape and torch.allclose(actual,expected), '投影或sqrt(D)缩放错误'
        """),
        _case("因果与全屏蔽行", """
        x=torch.tensor([[[1.,2.],[3.,6.],[5.,10.]]]); eye=torch.eye(2); zero=torch.zeros(2,2)
        mask=torch.tensor([[True,False,False],[True,True,False],[False,False,False]])
        out=candidate.self_attention(x,zero,zero,eye,mask)
        assert torch.allclose(out,torch.tensor([[[1.,2.],[2.,4.],[0.,0.]]])), 'True含义、全mask或均匀权重错误'
        """),
        _case("反向梯度及mask行有限性", """
        x=torch.randn(1,3,2,dtype=torch.float64,requires_grad=True); w=[torch.randn(2,2,dtype=x.dtype,requires_grad=True) for _ in range(3)]
        mask=torch.tensor([[[True,False,False],[True,True,False],[False,False,False]]])
        out=candidate.self_attention(x,*w,mask)
        out.square().sum().backward()
        assert torch.isfinite(out).all() and torch.isfinite(x.grad).all(), '全mask行不应产生NaN梯度'
        assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in w), 'QKV参数梯度丢失'
        x2=x.detach().clone().requires_grad_(); ws=[a.detach().clone().requires_grad_() for a in w]
        q,k,v=[x2@a for a in ws]
        s=q@k.transpose(-2,-1)/math.sqrt(2); s=s.masked_fill(~mask,-torch.inf)
        safe=torch.where(mask.any(-1,keepdim=True),s,torch.zeros_like(s))
        expected=(safe.softmax(-1).masked_fill(~mask,0))@v
        expected.square().sum().backward()
        assert torch.allclose(x.grad,x2.grad,atol=1e-9), '输入梯度错误'
        assert all(torch.allclose(a.grad,b.grad,atol=1e-9) for a,b in zip(w,ws)), '投影梯度错误'
        """),
    ], ["mask应在softmax之前作用于分数。", "全mask行需要避免softmax(-inf)引起NaN。"], True),
    _exercise(87, "Multi-Head Attention 的拆分与合并", """
    实现 `multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None)`。
    x为 `[B,T,D]`，四个权重为 `[D,D]`，线性投影采用右乘权重；无bias/dropout。
    num_heads为正整数且整除D，否则ValueError。拆头后为 `[B,H,T,D/H]`，
    分数按 `sqrt(D/H)` 缩放，拼回 `[B,T,D]` 后右乘wo。
    mask可广播到 `[B,H,T,T]`，True允许；全mask查询行输出0。
    例如num_heads=1时应等价于单头注意力再输出投影。
    CPU浮点输入，保留梯度，禁止直接使用内置attention算子。
    """, """
    import torch
    def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
        raise NotImplementedError
    """, """
    import math
    import torch
    def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
        b,t,d=x.shape
        if num_heads <= 0 or d % num_heads:
            raise ValueError('heads must divide embedding width')
        h=num_heads; dh=d//h
        q,k,v=[(x@w).reshape(b,t,h,dh).transpose(1,2) for w in (wq,wk,wv)]
        scores=q@k.transpose(-2,-1)/math.sqrt(dh)
        if mask is not None:
            scores=scores.masked_fill(~mask,-torch.inf)
        safe=torch.where(torch.isneginf(scores).all(-1,keepdim=True),torch.zeros_like(scores),scores)
        weights=safe.softmax(-1)
        if mask is not None:
            weights=weights.masked_fill(~mask,0)
        return (weights@v).transpose(1,2).reshape(b,t,d)@wo
    """, [
        _case("多头布局与独立头对照", """
        x=torch.randn(2,3,6,dtype=torch.float64); ws=[torch.randn(6,6,dtype=x.dtype) for _ in range(4)]
        q,k,v=[x@w for w in ws[:3]]; parts=[]
        for head in range(3):
            sl=slice(2*head,2*head+2)
            parts.append((q[...,sl]@k[...,sl].transpose(-2,-1)/math.sqrt(2)).softmax(-1)@v[...,sl])
        expected=torch.cat(parts,-1)@ws[3]
        out=candidate.multi_head_attention(x,*ws,3)
        assert out.shape==x.shape and torch.allclose(out,expected,atol=1e-9), '头维布局、缩放或合并错误'
        """),
        _case("广播mask与非法头数", """
        x=torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.]]]); eye=torch.eye(4); zero=torch.zeros(4,4)
        mask=torch.tensor([[True,False],[False,False]])
        out=candidate.multi_head_attention(x,zero,zero,eye,eye,2,mask)
        assert torch.equal(out,torch.cat((x[:,:1],torch.zeros_like(x[:,:1])),1)), '全mask输出须为0'
        try: candidate.multi_head_attention(x,eye,eye,eye,eye,3)
        except ValueError: pass
        else: assert False, '不能整除D的头数应报ValueError'
        """),
        _case("对照基础算子的梯度", """
        x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True); ws=[torch.randn(4,4,dtype=x.dtype,requires_grad=True) for _ in range(4)]
        out=candidate.multi_head_attention(x,*ws,2)
        ga=torch.autograd.grad(out.square().sum(),(x,*ws),retain_graph=True)
        q,k,v=[(x@w).reshape(1,3,2,2).transpose(1,2) for w in ws[:3]]
        expect=((q@k.transpose(-2,-1)/math.sqrt(2)).softmax(-1)@v).transpose(1,2).reshape(1,3,4)@ws[3]
        ge=torch.autograd.grad(expect.square().sum(),(x,*ws))
        assert all(torch.allclose(a,b,atol=1e-8) for a,b in zip(ga,ge)), 'MHA反向梯度错误'
        """),
    ], ["拆头后交换T与H；拼头前交换回来。", "缩放因子使用每个head的维度。"], True),
    _exercise(88, "Pre-Norm Transformer Block Forward", """
    实现 `transformer_block(x, wq, wk, wv, wo, w1, b1, w2, b2, num_heads, mask=None, eps=1e-5)`。
    x为 `[B,T,D]`；QKV/输出权重 `[D,D]`；w1 `[D,F]`、b1 `[F]`、w2 `[F,D]`、b2 `[D]`。
    采用无可学习仿射的Pre-Norm结构：`y=x+MHA(LN(x))`；
    `out=y+GELU(LN(y)@w1+b1)@w2+b2`。GELU使用PyTorch默认精确形式。
    MHA右乘权重，无bias/dropout，头维缩放；mask可广播到 `[B,H,T,T]`，True允许，
    全mask行的attention贡献为0。num_heads正且整除D；有限浮点CPU输入。
    例：所有投影/FFN参数为0时返回原x。不得调用内置Transformer/MHA，允许layer_norm与gelu。
    """, """
    import torch
    def transformer_block(x, wq, wk, wv, wo, w1, b1, w2, b2, num_heads, mask=None, eps=1e-5):
        raise NotImplementedError
    """, """
    import math
    import torch
    from torch.nn.functional import layer_norm, gelu
    def transformer_block(x, wq, wk, wv, wo, w1, b1, w2, b2, num_heads, mask=None, eps=1e-5):
        b,t,d=x.shape
        if num_heads <= 0 or d % num_heads:
            raise ValueError('heads must divide width')
        z=layer_norm(x,(d,),eps=eps); dh=d//num_heads
        q,k,v=[(z@w).reshape(b,t,num_heads,dh).transpose(1,2) for w in (wq,wk,wv)]
        scores=q@k.transpose(-2,-1)/math.sqrt(dh)
        if mask is not None:
            scores=scores.masked_fill(~mask,-torch.inf)
        safe=torch.where(torch.isneginf(scores).all(-1,keepdim=True),torch.zeros_like(scores),scores)
        weights=safe.softmax(-1)
        if mask is not None:
            weights=weights.masked_fill(~mask,0)
        y=x+(weights@v).transpose(1,2).reshape(b,t,d)@wo
        return y+gelu(layer_norm(y,(d,),eps=eps)@w1+b1)@w2+b2
    """, [
        _case("残差恒等与FFN偏置", """
        x=torch.randn(2,3,4); zero=torch.zeros(4,4); w1=torch.zeros(4,5); b1=torch.zeros(5); w2=torch.zeros(5,4); b2=torch.zeros(4)
        args=(zero,zero,zero,zero,w1,b1,w2,b2,2)
        assert torch.equal(candidate.transformer_block(x,*args),x), '双残差应保持输入'
        b2.fill_(2)
        assert torch.allclose(candidate.transformer_block(x,*args),x+2), 'FFN输出偏置不可遗漏'
        """),
        _case("PreNorm顺序、mask与FFN对照", """
        x=torch.randn(2,3,4,dtype=torch.float64); ws=[torch.randn(4,4,dtype=x.dtype) for _ in range(4)]
        w1=torch.randn(4,7,dtype=x.dtype); b1=torch.randn(7,dtype=x.dtype); w2=torch.randn(7,4,dtype=x.dtype); b2=torch.randn(4,dtype=x.dtype)
        mask=torch.ones(3,3,dtype=torch.bool).tril(); mask[-1]=False
        z=torch.nn.functional.layer_norm(x,(4,)); parts=[]
        q,k,v=[z@w for w in ws[:3]]
        for h in range(2):
            sl=slice(2*h,2*h+2); s=q[...,sl]@k[...,sl].transpose(-2,-1)/math.sqrt(2)
            s=s.masked_fill(~mask,-torch.inf); s=torch.where(mask.any(-1,keepdim=True),s,torch.zeros_like(s))
            parts.append(s.softmax(-1).masked_fill(~mask,0)@v[...,sl])
        y=x+torch.cat(parts,-1)@ws[3]
        expected=y+torch.nn.functional.gelu(torch.nn.functional.layer_norm(y,(4,))@w1+b1)@w2+b2
        out=candidate.transformer_block(x,*ws,w1,b1,w2,b2,2,mask)
        assert out.shape==x.shape and torch.allclose(out,expected,atol=1e-9), 'PreNorm、FFN或mask数值错误'
        """),
        _case("整块反向传播", """
        x=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
        ws=[torch.randn(4,4,dtype=x.dtype,requires_grad=True) for _ in range(4)]
        w1=torch.randn(4,5,dtype=x.dtype,requires_grad=True); b1=torch.randn(5,dtype=x.dtype,requires_grad=True)
        w2=torch.randn(5,4,dtype=x.dtype,requires_grad=True); b2=torch.randn(4,dtype=x.dtype,requires_grad=True)
        out=candidate.transformer_block(x,*ws,w1,b1,w2,b2,2)
        out.square().sum().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (x,*ws,w1,b1,w2,b2)), '某条梯度路径被detach'
        assert torch.autograd.gradcheck(lambda z:candidate.transformer_block(z,*[w.detach() for w in ws],w1.detach(),b1.detach(),w2.detach(),b2.detach(),2),(x.detach().requires_grad_(),),eps=1e-6,atol=1e-4), '输入梯度未通过有限差分'
        """),
    ], ["两个LayerNorm位于各自子层前。", "第二个归一化的输入为加过attention残差的y。"], True),
    _exercise(89, "数值稳定的 Cross Entropy", """
    实现 `cross_entropy(logits, targets, ignore_index=-100)`。
    logits为 `[N,C]` 浮点张量，targets为 `[N]` long，N,C>=1；有效target在[0,C)。
    返回有效样本的平均负对数似然，ignore_index不参与分母；全忽略时返回可反传0。
    不得调用内置cross_entropy、nll_loss或log_softmax；允许logsumexp。
    例：logits=[0,0]、target=1时loss=log(2)。大幅值logits必须保持有限并可求梯度。
    """, """
    import torch
    def cross_entropy(logits, targets, ignore_index=-100):
        raise NotImplementedError
    """, """
    import torch
    def cross_entropy(logits, targets, ignore_index=-100):
        valid=targets != ignore_index
        safe_targets=targets.masked_fill(~valid,0)
        nll=torch.logsumexp(logits, dim=-1)-logits.gather(1,safe_targets[:,None]).squeeze(1)
        return nll.masked_fill(~valid,0).sum()/valid.sum().clamp_min(1)
    """, [
        _case("普通样本与类别索引", """
        x=torch.tensor([[0.,0.,0.],[1.,2.,3.]],dtype=torch.float64); y=torch.tensor([2,1])
        out=candidate.cross_entropy(x,y)
        expected=(math.log(3)+math.log(math.exp(1)+math.exp(2)+math.exp(3))-2)/2
        assert out.ndim==0 and abs(out.item()-expected)<1e-10, '类别选择或均值错误'
        """),
        _case("大logits稳定性与平移不变性", """
        x=torch.tensor([[10000.,9999.,9998.],[-10000.,-9998.,-9999.]],dtype=torch.float64); y=torch.tensor([0,2])
        a=candidate.cross_entropy(x,y); b=candidate.cross_entropy(x-10000,y)
        expected=torch.nn.functional.cross_entropy(x,y)
        assert torch.isfinite(a) and torch.allclose(a,expected,atol=1e-10) and torch.allclose(a,b,atol=1e-10), '应使用稳定logsumexp'
        """),
        _case("ignore分母与梯度", """
        x=torch.randn(3,4,dtype=torch.float64,requires_grad=True); y=torch.tensor([1,-100,3])
        a=candidate.cross_entropy(x,y); b=torch.nn.functional.cross_entropy(x,y)
        ga=torch.autograd.grad(a,x,retain_graph=True)[0]; gb=torch.autograd.grad(b,x)[0]
        assert torch.allclose(a,b) and torch.allclose(ga,gb,atol=1e-10), '忽略项分母或梯度错误'
        z=candidate.cross_entropy(x,torch.full((3,),-100))
        assert z.item()==0 and z.requires_grad, '全忽略应可反传0'
        assert torch.equal(torch.autograd.grad(z,x)[0],torch.zeros_like(x)), '全忽略梯度应为0'
        """),
    ], ["logsumexp减目标类别logit可避免先exp的溢出。", "ignore_index不能直接用于gather。"], False),
    _exercise(90, "带 KV Cache 的因果注意力", """
    实现 `cached_attention(q, k, v, cache=None)` 返回 `(output, new_cache)`。
    当前chunk的q/k/v同为 `[B,L,D]`，已完成投影，不拆head；L>=1。
    cache若非None是 `{'k': [B,P,D], 'v': [B,P,D]}`，追加当前K/V，
    当前第i个query只允许访问下标 `<=P+i` 的key，分数除sqrt(D)。
    new_cache使用相同两个key，长度P+L；不得原地修改旧cache或传入张量，保留dtype/device。
    例如先prefill两个token，再逐tokendecode，应与一次完整因果attention一致。
    本题只实现注意力级小型接口，不包含多层模型、位置编码或服务端缓存淘汰。
    """, """
    import torch
    def cached_attention(q, k, v, cache=None):
        raise NotImplementedError
    """, """
    import math
    import torch
    def cached_attention(q, k, v, cache=None):
        offset=0 if cache is None else cache['k'].shape[1]
        all_k=k if cache is None else torch.cat((cache['k'],k),dim=1)
        all_v=v if cache is None else torch.cat((cache['v'],v),dim=1)
        query_positions=offset + torch.arange(q.shape[1],device=q.device)
        key_positions=torch.arange(all_k.shape[1],device=q.device)
        allowed=key_positions[None,:] <= query_positions[:,None]
        scores=q@all_k.transpose(-2,-1)/math.sqrt(q.shape[-1])
        weights=scores.masked_fill(~allowed,-torch.inf).softmax(-1)
        return weights@all_v, {'k':all_k,'v':all_v}
    """, [
        _case("prefill因果性与零分数均值", """
        q=torch.zeros(1,3,2); k=torch.ones_like(q); v=torch.tensor([[[1.,2.],[3.,4.],[8.,9.]]])
        out,cache=candidate.cached_attention(q,k,v)
        expected=torch.tensor([[[1.,2.],[2.,3.],[4.,5.]]])
        assert torch.allclose(out,expected), 'prefill应使用因果mask'
        assert cache['k'].shape==(1,3,2) and torch.equal(cache['v'],v), 'prefill缓存错误'
        """),
        _case("prefill加逐tokendecode等于全量", """
        q=torch.randn(2,5,4,dtype=torch.float64); k=torch.randn_like(q); v=torch.randn_like(q)
        s=q@k.transpose(-2,-1)/2; mask=torch.ones(5,5,dtype=torch.bool).tril()
        expected=s.masked_fill(~mask,-torch.inf).softmax(-1)@v
        a,cache=candidate.cached_attention(q[:,:2],k[:,:2],v[:,:2]); parts=[a]
        for i in range(2,5):
            out,cache=candidate.cached_attention(q[:,i:i+1],k[:,i:i+1],v[:,i:i+1],cache); parts.append(out)
        assert torch.allclose(torch.cat(parts,1),expected,atol=1e-10), 'decode的query位置需要加缓存长度'
        assert cache['k'].shape==(2,5,4) and cache['k'].dtype==q.dtype, '缓存长度或dtype错误'
        """),
        _case("多token追加与旧缓存不变", """
        q=torch.zeros(1,2,1); k=torch.ones_like(q); v=torch.tensor([[[3.],[5.]]])
        cache={'k':torch.ones(1,1,1),'v':torch.tensor([[[1.]]])}; old=cache['v'].clone()
        out,new=candidate.cached_attention(q,k,v,cache)
        assert torch.allclose(out,torch.tensor([[[2.],[3.]]])), 'chunk内部因果offset错误'
        assert cache['v'].shape==(1,1,1) and torch.equal(cache['v'],old), '不得修改旧缓存'
        assert new is not cache and torch.equal(new['v'],torch.tensor([[[1.],[3.],[5.]]])), '应返回追加后的新字典'
        """),
    ], ["query的绝对位置是缓存长度加chunk内下标。", "不能对长度L的query误用从0开始的三角mask。"], True),
    _exercise(91, "RTC 条件前缀缓存：版本失效与训练梯度", """
    实现小型工程接口 `reusable_prefix(conditions, version, encode, cache=None, training=False)`，
    返回 `(encoded, new_cache, reused)`。conditions为 `[B,P,D]`，encode是可微函数。
    约定version是由调用者维护的不可变版本标识，必须覆盖图像、语言、状态等所有条件及encoder参数；
    相同version保证条件与编码器完全一致，本函数不做张量内容猜测。
    推理时：缓存为 `{'version':version, 'encoded':tensor}`；版本相同时复用，不调用encode；
    版本不同时计算encode(conditions)，detach后返回并写入新字典，不修改旧字典。
    训练时始终重新计算encode，不detach；返回 `(encoded, None, False)`，防止过期计算图被复用。
    例：版本从('camera',1)变为('camera',2)必须重新编码。
    这是RTC条件前缀的缓存一致性练习，不是论文复现，也不表示动作后缀可以无条件缓存。
    """, """
    def reusable_prefix(conditions, version, encode, cache=None, training=False):
        raise NotImplementedError
    """, """
    def reusable_prefix(conditions, version, encode, cache=None, training=False):
        if training:
            return encode(conditions), None, False
        if cache is not None and cache['version'] == version:
            return cache['encoded'], cache, True
        encoded=encode(conditions).detach()
        return encoded, {'version':version,'encoded':encoded}, False
    """, [
        _case("相同版本复用并避免重复编码", """
        x=torch.randn(1,3,2,requires_grad=True); calls=[]
        def encode(a): calls.append(1); return a*2
        a,cache,reused=candidate.reusable_prefix(x,('task',1),encode)
        b,new,reused2=candidate.reusable_prefix(x,('task',1),encode,cache)
        assert len(calls)==1 and not reused and reused2, '相同版本应只编码一次'
        assert torch.equal(a,x*2) and b is cache['encoded'] and not a.requires_grad, '推理缓存应detach并复用tensor'
        assert new['version']==('task',1), '缓存版本错误'
        """),
        _case("条件版本变化失效且旧缓存保持", """
        x=torch.ones(1,2,3); calls=[]
        def encode(a): calls.append(1); return a.square()
        a,old,_=candidate.reusable_prefix(x,1,encode)
        b,new,reused=candidate.reusable_prefix(3*x,2,encode,old)
        assert len(calls)==2 and not reused and torch.equal(b,9*x), '条件变化后错误复用缓存'
        assert old['version']==1 and torch.equal(old['encoded'],x), '不得原地修改旧缓存'
        assert new['version']==2, '缓存版本未更新'
        """),
        _case("训练重算、连续反传与参数梯度", """
        x=torch.randn(1,2,3,requires_grad=True); p=torch.tensor(2.,requires_grad=True); calls=[]
        def encode(a): calls.append(1); return a*p
        _,cache,_=candidate.reusable_prefix(x,1,encode)
        for i in range(2):
            x.grad=None; p.grad=None
            y,new,reused=candidate.reusable_prefix(x,1,encode,cache,training=True)
            assert new is None and not reused and y.requires_grad, '训练不可使用detach推理缓存'
            y.sum().backward()
            assert torch.allclose(x.grad,torch.full_like(x,2.)) and torch.allclose(p.grad,x.detach().sum()), '训练梯度路径错误'
        assert len(calls)==3, '每轮训练应重算，避免重用旧计算图'
        """),
    ], ["版本标识必须包含所有会改变前缀表示的信息。", "推理缓存和训练计算图的生命周期不同。"], True),
    _exercise(92, "矩阵乘法与分块遍历", """
    实现 `matmul_naive(a, b)` 和 `matmul_tiled(a, b, block_size=16)`，
    a、b为矩形Python二维数值列表，形状 `[m,n]` 与 `[n,k]`，非空输入维度均>=1。
    返回新的 `[m,k]` 二维列表；a为空时返回[]，不修改输入。
    不可使用numpy/torch或内置矩阵乘法；naive使用三重循环，tiled显式按block_size分块，
    推荐块内 i-p-j 顺序使输出行与B行连续访问。正整数block_size，否则ValueError。
    例：[[1,2]]乘[[3],[4]]得到[[11]]。需处理不能整除block_size的尾块。
    本题测试正确性、输入不变和分块参数接口，不设置CPU微基准速度阈值；
    Python列表分块不保证比BLAS或朴素实现更快。两者算术复杂度均O(mnk)。
    """, """
    def matmul_naive(a, b):
        raise NotImplementedError
    def matmul_tiled(a, b, block_size=16):
        raise NotImplementedError
    """, """
    def matmul_naive(a,b):
        if not a:
            return []
        m,n,k=len(a),len(b),len(b[0])
        out=[[0 for _ in range(k)] for _ in range(m)]
        for i in range(m):
            for j in range(k):
                for p in range(n):
                    out[i][j] += a[i][p]*b[p][j]
        return out
    def matmul_tiled(a,b,block_size=16):
        if not isinstance(block_size,int) or isinstance(block_size,bool) or block_size<=0:
            raise ValueError('block_size must be a positive integer')
        if not a:
            return []
        m,n,k=len(a),len(b),len(b[0])
        out=[[0 for _ in range(k)] for _ in range(m)]
        for ii in range(0,m,block_size):
            for pp in range(0,n,block_size):
                for jj in range(0,k,block_size):
                    for i in range(ii,min(ii+block_size,m)):
                        for p in range(pp,min(pp+block_size,n)):
                            av=a[i][p]
                            for j in range(jj,min(jj+block_size,k)):
                                out[i][j]+=av*b[p][j]
        return out
    """, [
        _case("非方阵与负数", """
        a=[[1,2,-1],[0,3,4]]; b=[[2,1],[1,-2],[3,0]]; expected=[[1,-3],[15,-6]]
        assert candidate.matmul_naive(a,b)==expected, '朴素乘法维度或索引错误'
        assert candidate.matmul_tiled(a,b,2)==expected, '分块乘法结果错误'
        """),
        _case("尾块、浮点与输入不变", """
        a=[[random.uniform(-2,2) for _ in range(5)] for _ in range(3)]
        b=[[random.uniform(-2,2) for _ in range(7)] for _ in range(5)]
        old_a=[row[:] for row in a]; old_b=[row[:] for row in b]
        expected=[[sum(a[i][p]*b[p][j] for p in range(5)) for j in range(7)] for i in range(3)]
        for block in (1,2,4,32):
            out=candidate.matmul_tiled(a,b,block)
            assert len(out)==3 and all(len(row)==7 for row in out), '输出维度错误'
            assert all(abs(out[i][j]-expected[i][j])<1e-10 for i in range(3) for j in range(7)), '尾块丢失或重复累加'
        assert a==old_a and b==old_b, '不得修改输入矩阵'
        """),
        _case("空输入与非法block_size", """
        assert candidate.matmul_naive([],[])==[] and candidate.matmul_tiled([],[],1)==[], '空矩阵结果应为[]'
        assert candidate.matmul_tiled([[2]],[[3]],64)==[[6]], '单元素或大块边界错误'
        for block in (0,-2,1.5,True):
            try: candidate.matmul_tiled([[1]],[[1]],block)
            except ValueError: pass
            else: assert False, 'block_size须为正整数'
        """),
    ], ["三个维度的尾块分别取min边界。", "i-p-j内层连续访问B和输出的行。"], True),
    _exercise(93, "大数组 TopK：有界小顶堆", """
    实现 `top_k(nums, k)`，返回nums中最大的k个元素，按降序排列，保留重复值。
    nums为整数列表，0<=k<=len(nums)；k越界或负数抛ValueError；不修改nums。
    例：[3,1,3,2]、k=3返回[3,3,2]，k=0返回[]。
    目标时间O(n log k + k log k)、额外空间O(k)，可使用heapq；不得对完整输入排序。
    数据可能含负数和重复值。功能测试不依赖不可靠的耗时门槛，复杂度需人工审阅。
    """, """
    def top_k(nums, k):
        raise NotImplementedError
    """, """
    import heapq
    def top_k(nums,k):
        if k<0 or k>len(nums):
            raise ValueError('k out of range')
        if k==0:
            return []
        heap=[]
        for item in nums:
            if len(heap)<k:
                heapq.heappush(heap,item)
            elif item>heap[0]:
                heapq.heapreplace(heap,item)
        return sorted(heap,reverse=True)
    """, [
        _case("重复值、负数与输出顺序", """
        a=[-5,3,1,3,-1,2]; old=a[:]
        assert candidate.top_k(a,4)==[3,3,2,1], '应保留重复值并降序返回'
        assert a==old, '不得修改输入'
        assert candidate.top_k([-9,-2,-4],2)==[-2,-4], '负数TopK错误'
        """),
        _case("零、全量及非法k", """
        assert candidate.top_k([],0)==[] and candidate.top_k([2,1],0)==[], 'k=0应返回[]'
        assert candidate.top_k([2,1,2],3)==[2,2,1], 'k=n错误'
        for k in (-1,3):
            try: candidate.top_k([1,2],k)
            except ValueError: pass
            else: assert False, 'k越界须ValueError'
        """),
        _case("固定种子随机对照", """
        for n in (1,10,101,1000):
            nums=[random.randint(-50,50) for _ in range(n)]
            for k in (1,n//2,n):
                assert candidate.top_k(nums,k)==sorted(nums,reverse=True)[:k], '随机TopK与排序基准不符'
        """),
    ], ["堆顶保留当前TopK中最小的元素。", "只有候选值大于堆顶才替换。"], False),
    _exercise(94, "雨水容量：双指针", """
    实现 `trap_water(heights)`。整数列表heights表示等宽柱子，柱宽为1且高度非负；
    计算各柱子上方可被左右较高边界围住的水量总和，返回非负整数。
    例：[2,0,2]返回2；[3,0,1,3]返回5。空列表或不足三根柱子返回0。
    不修改输入，目标O(n)时间、O(1)额外空间；不能模拟逐层灌水。
    """, """
    def trap_water(heights):
        raise NotImplementedError
    """, """
    def trap_water(heights):
        left,right=0,len(heights)-1
        left_max=right_max=water=0
        while left<=right:
            if left_max<=right_max:
                left_max=max(left_max,heights[left])
                water += left_max - heights[left]
                left+=1
            else:
                right_max=max(right_max,heights[right])
                water += right_max - heights[right]
                right-=1
        return water
    """, [
        _case("多盆地与不对称边界", """
        assert candidate.trap_water([0,1,0,2,1,0,1,3,2,1,2,1])==6, '多盆地累计错误'
        assert candidate.trap_water([4,2,0,3,2,5])==9, '左右边界处理错误'
        assert candidate.trap_water([3,0,1,3])==5, '内部矮柱的体积应扣除'
        """),
        _case("退化输入与输入不变", """
        for a in ([],[1],[2,0],[0,1,2,3],[3,2,1],[2,2,2]):
            old=a[:]; assert candidate.trap_water(a)==0, '无盆地应为0'
            assert a==old, '不得修改输入'
        assert candidate.trap_water([100000,0,100000])==100000, '大高度错误'
        """),
        _case("随机与逐位置上界对照", """
        for _ in range(60):
            a=[random.randrange(10) for _ in range(random.randrange(1,30))]
            expected=sum(max(0,min(max(a[:i+1]),max(a[i:]))-a[i]) for i in range(len(a)))
            assert candidate.trap_water(a)==expected, '随机双指针结果与定义不符'
        """),
    ], ["当前位置水量由较矮的左右最高边界决定。", "每次推进当前边界上限较小的一侧。"], False),
    _exercise(95, "网格陆地的连通分量", """
    实现 `count_islands(grid)`，grid为矩形二维字符串列表，每项为'0'或'1'。
    '1'为陆地，仅上下左右相邻算连通，返回陆地连通分量数；对角线不连通。
    例：[['1','0'],['0','1']]返回2；空grid或零列返回0。
    不修改输入，目标O(R*C)时间、O(R*C)辅助空间；大连通区域避免Python递归深度限制。
    """, """
    def count_islands(grid):
        raise NotImplementedError
    """, """
    def count_islands(grid):
        if not grid or not grid[0]:
            return 0
        rows,cols=len(grid),len(grid[0]); seen=set(); count=0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]!='1' or (r,c) in seen:
                    continue
                count+=1; seen.add((r,c)); stack=[(r,c)]
                while stack:
                    x,y=stack.pop()
                    for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                        if 0<=nx<rows and 0<=ny<cols and grid[nx][ny]=='1' and (nx,ny) not in seen:
                            seen.add((nx,ny)); stack.append((nx,ny))
        return count
    """, [
        _case("四连通与多个岛屿", """
        a=[list('11000'),list('11000'),list('00100'),list('00011')]
        assert candidate.count_islands(a)==3, '四连通区域计数错误'
        assert candidate.count_islands([list('10'),list('01')])==2, '对角不能算连通'
        """),
        _case("空网格与不可修改输入", """
        assert candidate.count_islands([])==0 and candidate.count_islands([[]])==0, '空网格边界错误'
        a=[list('111'),list('101'),list('111')]; old=[row[:] for row in a]
        assert candidate.count_islands(a)==1 and a==old, '带孔陆地应连通且输入不变'
        assert candidate.count_islands([list('000')])==0, '全水应为0'
        """),
        _case("长陆地与棋盘格", """
        assert candidate.count_islands([['1']*2500])==1, '长区域应避免递归栈溢出'
        grid=[['1' if (r+c)%2==0 else '0' for c in range(11)] for r in range(9)]
        assert candidate.count_islands(grid)==50, '棋盘格中的陆地彼此不相连'
        """),
    ], ["从未访问陆地启动一次DFS/BFS就发现一个新岛屿。", "使用显式栈或队列避免递归深度问题。"], False),
    _exercise(96, "方阵原地顺时针旋转", """
    实现 `rotate_clockwise(matrix)`，matrix为n×n的Python二维列表，n>=0。
    原地顺时针旋转90度，并返回None；必须保留外层列表与每行对象的身份，不能创建第二张矩阵。
    例：[[1,2],[3,4]]旋转后变为[[3,1],[4,2]]。空矩阵合法。
    目标O(n²)时间、O(1)额外空间。可转置后逐行反转，但不要整体替换各行。
    """, """
    def rotate_clockwise(matrix):
        raise NotImplementedError
    """, """
    def rotate_clockwise(matrix):
        n=len(matrix)
        for i in range(n):
            for j in range(i+1,n):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for row in matrix:
            row.reverse()
    """, [
        _case("奇数阶数值与对象身份", """
        a=[[1,2,3],[4,5,6],[7,8,9]]; outer=id(a); rows=[id(r) for r in a]
        result=candidate.rotate_clockwise(a)
        assert result is None and a==[[7,4,1],[8,5,2],[9,6,3]], '顺时针旋转或返回值错误'
        assert id(a)==outer and [id(r) for r in a]==rows, '必须保留原列表和每行对象'
        """),
        _case("空、单元素与偶数阶", """
        a=[]; assert candidate.rotate_clockwise(a) is None and a==[], '空矩阵应合法'
        a=[[42]]; candidate.rotate_clockwise(a); assert a==[[42]], '单元素错误'
        a=[[1,2],[3,4]]; candidate.rotate_clockwise(a); assert a==[[3,1],[4,2]], '偶数矩阵旋转方向错误'
        """),
        _case("四次旋转回到原矩阵", """
        for n in (4,5,10):
            a=[[r*n+c for c in range(n)] for r in range(n)]; old=[row[:] for row in a]
            for _ in range(4): candidate.rotate_clockwise(a)
            assert a==old, '四次90度旋转应还原'
        """),
    ], ["先沿主对角线转置，再将每一行原地反转。", "只交换对角线一侧，否则会交换两次。"], False),
    _exercise(97, "链表环入口：Floyd 双指针", """
    实现 `cycle_entry(head)`，输入节点具有 `.val` 与 `.next` 属性，末尾next可为None。
    返回环入口的原节点对象，无环返回None，不能根据节点值判断身份，不能修改链表。
    例：A→B→C→B返回B；同值节点并不代表同一个节点。
    目标O(n)时间、O(1)额外空间，禁止使用set记录已访问节点；测试可传入任意兼容节点类。
    模板中的ListNode仅用于自己构造示例。
    """, """
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val, self.next = val, next
    def cycle_entry(head):
        raise NotImplementedError
    """, """
    class ListNode:
        def __init__(self,val=0,next=None):
            self.val,self.next=val,next
    def cycle_entry(head):
        slow=fast=head
        while fast is not None and fast.next is not None:
            slow=slow.next; fast=fast.next.next
            if slow is fast:
                break
        else:
            return None
        slow=head
        while slow is not fast:
            slow=slow.next; fast=fast.next
        return slow
    """, [
        _case("无环、空链表与单节点", """
        class Node:
            def __init__(self,v): self.val=v; self.next=None
        assert candidate.cycle_entry(None) is None, '空链表应返回None'
        a=Node(1); assert candidate.cycle_entry(a) is None, '单节点无环'
        b=Node(2); a.next=b; assert candidate.cycle_entry(a) is None, '有尾节点的链表无环'
        """),
        _case("环入口身份与重复值", """
        class Node:
            def __init__(self,v): self.val=v; self.next=None
        nodes=[Node(7) for _ in range(6)]
        for a,b in zip(nodes,nodes[1:]): a.next=b
        nodes[-1].next=nodes[2]; old=[n.next for n in nodes]
        assert candidate.cycle_entry(nodes[0]) is nodes[2], '相同值不能当作相同节点'
        assert all(n.next is nxt for n,nxt in zip(nodes,old)), '不得修改链表'
        """),
        _case("头部入环、自环与长尾", """
        class Node:
            def __init__(self): self.val=1; self.next=None
        a=Node(); a.next=a; assert candidate.cycle_entry(a) is a, '自环入口错误'
        nodes=[Node() for _ in range(51)]
        for a,b in zip(nodes,nodes[1:]): a.next=b
        nodes[-1].next=nodes[0]; assert candidate.cycle_entry(nodes[0]) is nodes[0], '头部入环错误'
        nodes[-1].next=nodes[40]; assert candidate.cycle_entry(nodes[0]) is nodes[40], '长尾入口错误'
        """),
    ], ["快慢指针首次相遇后，将一根指针重置到head。", "第二阶段两根指针等速前进，使用is比较身份。"], False),
    _exercise(98, "整体有序二维矩阵搜索", """
    实现 `search_matrix(matrix, target)` 返回bool。
    matrix为矩形二维整数列表，每行严格递增，后一行首元素大于前一行末元素。
    空矩阵或零列返回False，不修改输入。例：[[1,3,5],[8,10,12]]中搜索8为True，搜索6为False。
    目标O(log(R*C))时间、O(1)空间；用一维二分下标映射到行列，不展开复制矩阵。
    """, """
    def search_matrix(matrix, target):
        raise NotImplementedError
    """, """
    def search_matrix(matrix,target):
        if not matrix or not matrix[0]:
            return False
        cols=len(matrix[0]); left,right=0,len(matrix)*cols-1
        while left<=right:
            mid=(left+right)//2; value=matrix[mid//cols][mid%cols]
            if value==target:
                return True
            if value<target:
                left=mid+1
            else:
                right=mid-1
        return False
    """, [
        _case("行边界与不存在目标", """
        a=[[1,3,5],[8,10,12]]
        for t in (1,5,8,12): assert candidate.search_matrix(a,t) is True, '行端点应找到'
        for t in (-1,6,13): assert candidate.search_matrix(a,t) is False, '区间外或间隙应返回False'
        """),
        _case("空矩阵、单行单列与输入不变", """
        assert candidate.search_matrix([],2) is False and candidate.search_matrix([[]],2) is False, '空矩阵错误'
        assert candidate.search_matrix([[4]],4) is True and candidate.search_matrix([[4]],3) is False, '单元素边界错误'
        a=[[-3],[0],[8]]; old=[row[:] for row in a]
        assert candidate.search_matrix(a,0) is True and a==old, '单列或输入修改错误'
        """),
        _case("矩形下标映射随机对照", """
        for rows,cols in ((2,7),(9,2),(1,31)):
            values=sorted(random.sample(range(-200,200),rows*cols))
            grid=[values[i*cols:(i+1)*cols] for i in range(rows)]
            for t in range(-205,205,7):
                assert candidate.search_matrix(grid,t)==(t in values), '行列数不同的下标映射错误'
        """),
    ], ["mid//列数是行，mid%列数是列。", "二分区间端点约定必须与更新方式一致。"], False),
    _exercise(99, "二分法求非负浮点平方根", """
    实现 `square_root(x, tol=1e-10)`，x为有限浮点数且0<=x<=1e12。
    用二分法求非负平方根，返回浮点数；误差应不超过 `2*tol*max(1,sqrt(x))`。
    x为负或tol<=0抛ValueError。tol的测试范围[1e-12,1e-6]。
    例：x=.25返回约.5，x=2返回约1.41421356，x=0返回0。
    不得使用sqrt、pow或**.5；二分上界至少为max(1,x)，注意x<1时不能只搜索[0,x]。
    需要可靠终止，可设置有限迭代上限处理浮点停滞。
    """, """
    def square_root(x, tol=1e-10):
        raise NotImplementedError
    """, """
    def square_root(x,tol=1e-10):
        if x<0 or tol<=0:
            raise ValueError('x must be nonnegative and tol positive')
        if x==0:
            return 0.0
        left,right=0.0,max(1.0,float(x))
        for _ in range(256):
            mid=(left+right)/2
            if right-left<=tol*max(1.0,mid) or mid==left or mid==right:
                return mid
            if mid*mid>x:
                right=mid
            else:
                left=mid
        return (left+right)/2
    """, [
        _case("零、整数平方与无理数", """
        for x in (0.,1.,4.,9.,2.,7.,10000.):
            result=candidate.square_root(x)
            assert isinstance(result,(float,int)) and abs(result-math.sqrt(x))<=2e-10*max(1,math.sqrt(x)), '平方根精度不符合接口'
        """),
        _case("小于1和大数的边界", """
        for x in (1e-12,1e-8,.01,.25,.99,1e8,1e12):
            result=candidate.square_root(x,1e-12)
            assert result>=0 and abs(result-math.sqrt(x))<=2e-12*max(1,math.sqrt(x)), '分数区间或大数收敛错误'
        """),
        _case("随机尺度与非法参数", """
        for _ in range(40):
            x=10**random.uniform(-10,12)
            y=candidate.square_root(x,1e-9)
            assert abs(y-math.sqrt(x))<=2e-9*max(1,math.sqrt(x)), '随机尺度精度错误'
        for x,tol in ((-1.,1e-10),(1.,0.),(1.,-1e-8)):
            try: candidate.square_root(x,tol)
            except ValueError: pass
            else: assert False, '负x或非正tol应ValueError'
        """),
    ], ["x在(0,1)时，平方根比x更大。", "以区间宽度相对当前量级作为终止标准。"], False),
    _exercise(100, "差值受限的最长连续窗口", """
    实现 `longest_bounded_subarray(nums, limit)`，nums为整数列表，limit为非负整数。
    返回最长连续子数组长度，要求其中任意两元素绝对差不超过limit，等价于max-min<=limit。
    例：[8,2,4,7]、limit=4返回2；[1,1,1]、limit=0返回3。空列表返回0。
    limit<0抛ValueError；不修改输入。目标O(n)时间、O(n)额外空间，
    用单调递增/递减索引队列维护最小/最大值，不能每次对窗口重新扫描求min/max。
    """, """
    def longest_bounded_subarray(nums, limit):
        raise NotImplementedError
    """, """
    from collections import deque
    def longest_bounded_subarray(nums,limit):
        if limit<0:
            raise ValueError('limit must be nonnegative')
        maximum=deque(); minimum=deque(); left=best=0
        for right,value in enumerate(nums):
            while maximum and nums[maximum[-1]]<value:
                maximum.pop()
            while minimum and nums[minimum[-1]]>value:
                minimum.pop()
            maximum.append(right); minimum.append(right)
            while nums[maximum[0]] - nums[minimum[0]] > limit:
                if maximum[0]==left:
                    maximum.popleft()
                if minimum[0]==left:
                    minimum.popleft()
                left+=1
            best=max(best,right-left+1)
        return best
    """, [
        _case("重复值、零限制与典型窗口", """
        assert candidate.longest_bounded_subarray([8,2,4,7],4)==2, '窗口收缩错误'
        assert candidate.longest_bounded_subarray([10,1,2,4,7,2],5)==4, '最长窗口可能位于中间'
        assert candidate.longest_bounded_subarray([4,2,2,2,4,4,2,2],0)==3, '差值等于limit时应合法'
        """),
        _case("空、负值与输入不变", """
        assert candidate.longest_bounded_subarray([],0)==0, '空数组长度0'
        a=[-5,-4,-3,0,-2]; old=a[:]
        assert candidate.longest_bounded_subarray(a,2)==3 and a==old, '负数或输入不变性错误'
        assert candidate.longest_bounded_subarray([99],0)==1, '单元素总合法'
        try: candidate.longest_bounded_subarray([1],-1)
        except ValueError: pass
        else: assert False, '负limit应ValueError'
        """),
        _case("随机与穷举窗口对照", """
        for _ in range(80):
            a=[random.randint(-10,10) for _ in range(random.randrange(1,25))]; limit=random.randrange(10)
            expected=max([0]+[j-i for i in range(len(a)) for j in range(i+1,len(a)+1) if max(a[i:j])-min(a[i:j])<=limit])
            assert candidate.longest_bounded_subarray(a,limit)==expected, '单调队列移除过期下标错误'
        """),
    ], ["合法条件只需检查当前窗口最大值与最小值。", "队列中存索引，便于删除越过左端点的元素。"], True),
    _exercise(101, "含负数数组中和为 K 的最长子数组", """
    实现 `longest_sum_k(nums, k)`，nums和k均为整数，nums可含正数、零和负数。
    返回和恰为k的最长连续子数组长度，不存在返回0，不修改输入。
    例：[1,-1,5,-2,3]、k=3返回4；[1,-1,1,-1]、k=0返回4。
    目标O(n)时间、O(n)额外空间。用前缀和到最早位置的哈希映射；
    不能使用仅对非负数成立的普通滑动窗口。空数组返回0，空子数组不计入答案。
    """, """
    def longest_sum_k(nums, k):
        raise NotImplementedError
    """, """
    def longest_sum_k(nums,k):
        first={0:-1}; prefix=best=0
        for i,value in enumerate(nums):
            prefix+=value
            if prefix-k in first:
                best=max(best,i-first[prefix-k])
            if prefix not in first:
                first[prefix]=i
        return best
    """, [
        _case("负数与最早前缀位置", """
        assert candidate.longest_sum_k([1,-1,5,-2,3],3)==4, '含负数不能用普通双指针'
        assert candidate.longest_sum_k([1,-1,1,-1],0)==4, '前缀和应保留最早下标'
        assert candidate.longest_sum_k([-2,-1,2,1],1)==2, '负数前缀处理错误'
        """),
        _case("空、全零与无解", """
        assert candidate.longest_sum_k([],0)==0, '空数组结果0'
        assert candidate.longest_sum_k([0,0,0,0],0)==4, '起点前缀0必须映射到-1'
        a=[2,4,6]; old=a[:]
        assert candidate.longest_sum_k(a,1)==0 and a==old, '无解或输入修改错误'
        assert candidate.longest_sum_k([-5],-5)==1, '单元素负数目标错误'
        """),
        _case("随机与枚举求和对照", """
        for _ in range(80):
            a=[random.randint(-4,4) for _ in range(random.randrange(1,25))]; k=random.randint(-8,8)
            expected=max([0]+[j-i for i in range(len(a)) for j in range(i+1,len(a)+1) if sum(a[i:j])==k])
            assert candidate.longest_sum_k(a,k)==expected, '前缀差或长度计算错误'
        """),
    ], ["若prefix[j]-prefix[i]=k，则(i,j]是候选。", "相同前缀和只存第一次出现的位置。"], True),
]


def get_exercises(include_private: bool = False) -> list[dict]:
    """Return the 20 exercises, omitting grading secrets unless explicitly requested."""
    exercises = deepcopy(_CATALOG)
    if not include_private:
        for exercise in exercises:
            exercise.pop("reference")
            exercise.pop("tests")
    return exercises
