# ST-Transformer 模型结构示意图

基于 `ST-Transformer.py` 的完整架构说明。

---

## 一、整体数据流

```mermaid
flowchart TB
    subgraph Input [输入层]
        X["x: (B, T, F)"]
        CityID["city_id: (B,)"]
    end

    subgraph Embed [嵌入与投影]
        InputProj["input_proj\nLinear(F → d_model)"]
        PosEnc["pos_enc\nSinusoidalPositionalEncoding"]
        CityEmbed["city_embed\nEmbedding(N_city → city_embed_dim)"]
        EmbedProj["embed_proj\nLinear(d_model+city_embed_dim → d_model)"]
    end

    subgraph ModeSwitch {attention_mode}
        TemporalOnly["temporal_only"]
        SpatialTemporal["spatial_temporal"]
        UnifiedST["unified_st"]
    end

    subgraph Output [输出层]
        FC["fc\nLinear(d_model → 1)"]
        Pred["PM2.5 预测 (B,)"]
    end

    X --> InputProj
    InputProj --> PosEnc
    PosEnc --> EmbedProj
    CityID --> CityEmbed
    CityEmbed --> EmbedProj
    EmbedProj --> ModeSwitch
    ModeSwitch --> FC
    FC --> Pred
```

---

## 二、三种注意力模式分支

```mermaid
flowchart TD
    subgraph Common [公共前处理]
        Tokens["temporal_tokens\n(B, T, d_model)"]
    end

    subgraph TemporalOnly [temporal_only 模式]
        T1["TemporalEncoder\nTransformerEncoder"]
        Mean1["mean(dim=1)"]
        Drop1["Dropout"]
        Out1["→ fc → 预测"]
    end

    subgraph SpatialTemporal [spatial_temporal 模式]
        T2["TemporalEncoder"]
        S2["SpatialSelfAttentionBlock\n城市间注意力"]
        Gate2["GateFusion\n门控融合"]
        Out2["→ fc → 预测"]
    end

    subgraph UnifiedST [unified_st 模式]
        S3["SpatialSelfAttentionBlock\n构建 city_tokens"]
        U3["UnifiedSTAttentionBlock\n统一时空注意力"]
        Gate3["GateFusion"]
        Out3["→ fc → 预测"]
    end

    Tokens --> T1
    T1 --> Mean1 --> Drop1 --> Out1

    Tokens --> T2
    Tokens --> S2
    T2 --> Gate2
    S2 --> Gate2
    Gate2 --> Out2

    Tokens --> S3
    S3 --> U3
    U3 --> Gate3
    Gate3 --> Out3
```

---

## 三、空间自注意力块 (SpatialSelfAttentionBlock)

```mermaid
flowchart LR
    subgraph Input [输入]
        CityTokens["city_tokens\n(B, N_city, city_embed_dim)"]
    end

    subgraph Block [SpatialSelfAttentionBlock]
        Proj["city_proj\nLinear → d_model"]
        Enc["TransformerEncoder\n多头自注意力 N_city×N_city"]
    end

    subgraph Output [输出]
        SpatialOut["(B, N_city, d_model)"]
    end

    CityTokens --> Proj --> Enc --> SpatialOut
```

**说明**：在同一 batch 内，对所有城市 token 做自注意力，捕捉城市间的空间依赖。

---

## 四、统一时空注意力块 (UnifiedSTAttentionBlock)

```mermaid
flowchart LR
    subgraph Input [输入]
        Temporal["temporal_tokens\n(B, T, d_model)"]
        Spatial["spatial_tokens\n(B, N_city, d_model)"]
    end

    subgraph Block [UnifiedSTAttentionBlock]
        Concat["concat(dim=1)"]
        UnifiedTokens["unified_tokens\n(B, T+N_city, d_model)"]
        Enc["TransformerEncoder\n(NT)×(NT) 全局注意力"]
    end

    subgraph Output [输出]
        UnifiedOut["(B, T+N_city, d_model)"]
    end

    Temporal --> Concat
    Spatial --> Concat
    Concat --> UnifiedTokens --> Enc --> UnifiedOut
```

**说明**：将时间 token 与空间 token 拼接为统一序列，做全局自注意力，建模任意时空点之间的依赖。

---

## 五、spatial_temporal 模式详细流程

```mermaid
flowchart TB
    subgraph Step1 [步骤 1: 时间编码]
        T1["temporal_tokens"]
        TE["TemporalEncoder"]
        TemporalRepr["temporal_repr = mean(temporal_out)"]
    end

    subgraph Step2 [步骤 2: 空间上下文]
        Summary["summary = mean(temporal_tokens)"]
        SampleToCity["sample_to_city\nLinear(d_model → city_embed_dim)"]
        CityTokens["city_tokens = city_embed.weight + summary_bias"]
        SAB["SpatialSelfAttentionBlock"]
        Gather["gather(city_id)\n取当前城市表示"]
        SpatialRepr["spatial_repr"]
    end

    subgraph Step3 [步骤 3: 门控融合]
        GateFC["gate_fc\nLinear(2*d_model → 1)"]
        Sigmoid["sigmoid"]
        Fused["fused = gate*temporal + (1-gate)*spatial"]
        FusionDropout["fusion_dropout"]
    end

    T1 --> TE --> TemporalRepr
    T1 --> Summary --> SampleToCity
    SampleToCity --> CityTokens
    CityTokens --> SAB --> Gather --> SpatialRepr
    TemporalRepr --> GateFC
    SpatialRepr --> GateFC
    GateFC --> Sigmoid --> Fused --> FusionDropout
```

---

## 六、unified_st 模式详细流程

```mermaid
flowchart TB
    subgraph Step1 [构建空间 token]
        Summary["summary = mean(temporal_tokens)"]
        SampleToCity["sample_to_city"]
        CityTokens["city_tokens"]
        SAB["SpatialSelfAttentionBlock"]
        SpatialTokens["spatial_tokens (B, N_city, d_model)"]
    end

    subgraph Step2 [统一注意力]
        Concat["concat(temporal, spatial)"]
        UnifiedBlock["UnifiedSTAttentionBlock"]
        UnifiedOut["unified_out (B, T+N_city, d_model)"]
    end

    subgraph Step3 [分解与融合]
        TemporalPart["unified_out[:, :T] → mean → temporal_repr"]
        SpatialPart["unified_out[:, T:] → gather(city_id) → spatial_repr"]
        GateFuse["GateFusion"]
    end

    Summary --> SampleToCity --> CityTokens --> SAB --> SpatialTokens
    SpatialTokens --> Concat
    Concat --> UnifiedBlock --> UnifiedOut
    UnifiedOut --> TemporalPart
    UnifiedOut --> SpatialPart
    TemporalPart --> GateFuse
    SpatialPart --> GateFuse
```

---

## 七、模块与张量形状对照表

| 模块 | 输入形状 | 输出形状 |
|------|----------|----------|
| input_proj | (B, T, F) | (B, T, d_model) |
| pos_enc | (B, T, d_model) | (B, T, d_model) |
| city_embed | city_id (B,) | (B, T, city_embed_dim) 广播 |
| embed_proj | (B, T, d_model+city_embed_dim) | (B, T, d_model) |
| TemporalEncoder | (B, T, d_model) | (B, T, d_model) |
| SpatialSelfAttentionBlock | (B, N_city, city_embed_dim) | (B, N_city, d_model) |
| UnifiedSTAttentionBlock | temporal(B,T,d) + spatial(B,N,d) | (B, T+N_city, d_model) |
| GateFusion | temporal_repr(B,d) + spatial_repr(B,d) | (B, d_model) |
| fc | (B, d_model) | (B, 1) → squeeze → (B,) |

---

## 八、CLI 参数与模式对应

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --attention-mode | temporal_only | temporal_only / spatial_temporal / unified_st |
| --spatial-heads | 4 | 空间注意力头数 |
| --spatial-layers | 1 | 空间 Transformer 层数 |
| --unified-heads | 4 | 统一时空注意力头数 |
| --unified-layers | 1 | 统一时空 Transformer 层数 |
| --fusion-dropout | 0.1 | 融合后 Dropout |
