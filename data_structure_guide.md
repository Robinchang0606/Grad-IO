# 上市公司网络指标构建 - 数据结构指南

## 概述

本文档详细说明了构建上市公司网络指标所需的数据结构和处理流程。主要包含两个核心指标：

1. **指标1**: 上市公司关联公司间的直接投入金额加总
2. **指标2**: 企业实际拥有的关联交易公司数量

## 数据结构要求

### 1. 公司基本信息表 (company_data)

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|----------|------|------|
| company_id | String | 公司唯一标识符 | "A001", "B002" |
| company_name | String | 公司名称 | "某某科技股份有限公司" |
| is_listed | Integer | 是否为上市公司 (1=是, 0=否) | 1 |
| industry | String | 所属行业 | "制造业", "金融业" |
| total_assets | Float | 总资产(万元) | 1000000.0 |
| registration_date | Date | 注册日期 | "2010-01-15" |
| stock_code | String | 股票代码(上市公司) | "000001" |

**CSV文件示例格式:**
```csv
company_id,company_name,is_listed,industry,total_assets,registration_date,stock_code
A001,科技公司A,1,科技业,1000000,2010-01-15,000001
A002,制造公司B,0,制造业,500000,2012-03-20,
A003,服务公司C,0,服务业,300000,2015-06-10,
```

### 2. 关联交易信息表 (transaction_data)

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|----------|------|------|
| transaction_id | Integer | 交易唯一标识符 | 1, 2, 3 |
| from_company | String | 交易发起方公司ID | "A001" |
| to_company | String | 交易接收方公司ID | "A002" |
| transaction_amount | Float | 交易金额(万元) | 1000000.0 |
| transaction_type | String | 交易类型 | "投资", "采购", "销售", "服务" |
| year | Integer | 交易年份 | 2023 |
| transaction_date | Date | 交易日期 | "2023-03-15" |
| ownership_ratio | Float | 持股比例(投资类交易) | 0.25 |

**CSV文件示例格式:**
```csv
transaction_id,from_company,to_company,transaction_amount,transaction_type,year,transaction_date,ownership_ratio
1,A001,A002,1000000,投资,2023,2023-03-15,0.25
2,A001,A003,800000,采购,2023,2023-04-20,
3,A002,A003,300000,销售,2023,2023-05-10,
```

## 指标构建方法

### 指标1: 直接投入金额加总

**定义**: 衡量上市公司在关联交易网络中的资金投入规模

**计算方法**:
1. **对外直接投资** = Σ(该公司作为投资方的所有投资金额)
2. **接受直接投资** = Σ(该公司作为被投资方接受的所有投资金额)  
3. **总直接投入** = 对外直接投资 + 接受直接投资

**公式表示**:
```
DirectInvestment_i = Σ(OutgoingInvestment_ij) + Σ(IncomingInvestment_ji)
```

其中:
- i 表示上市公司
- j 表示关联公司
- OutgoingInvestment_ij 表示公司i对公司j的投资金额
- IncomingInvestment_ji 表示公司j对公司i的投资金额

**衍生指标**:
- **投资强度** = 总直接投入 / 该指标最大值 (标准化指标)
- **投资交易笔数** = 参与的投资类交易数量
- **平均单笔投资** = 总直接投入 / 投资交易笔数

### 指标2: 关联交易公司数量

**定义**: 衡量上市公司在关联交易网络中的连接广度和网络位置

**计算方法**:
1. **直接关联公司数量** = 与该公司有直接交易关系的公司数量
2. **间接关联公司数量** = 通过直接关联公司能够到达的二度连接公司数量
3. **总网络规模** = 直接关联 + 间接关联公司数量

**网络中心性指标**:
- **度中心性** = 该公司的连接数 / 网络中最大可能连接数
- **介数中心性** = 经过该公司的最短路径数 / 所有最短路径总数  
- **接近中心性** = 1 / 该公司到其他所有公司的平均距离

**按交易类型分类统计**:
- **投资伙伴数量** = 有投资关系的公司数量
- **交易伙伴数量** = 有采购/销售关系的公司数量  
- **服务伙伴数量** = 有服务关系的公司数量

## 数据处理流程

### 1. 数据预处理
```python
# 数据清洗
- 去除重复记录
- 处理缺失值
- 统一数据格式
- 验证数据完整性

# 数据验证
- 检查公司ID的一致性
- 验证交易金额的合理性
- 确认日期格式正确性
```

### 2. 网络构建
```python
# 创建网络图
- 以公司为节点(Node)
- 以交易关系为边(Edge)
- 添加节点属性(公司信息)
- 添加边属性(交易信息)
```

### 3. 指标计算
```python
# 指标1计算流程
for each 上市公司:
    计算对外投资总额
    计算接受投资总额
    计算总直接投入
    计算衍生指标

# 指标2计算流程  
for each 上市公司:
    识别直接关联公司
    识别间接关联公司
    计算网络中心性指标
    按交易类型分类统计
```

### 4. 结果输出
```python
# 生成三个输出文件
- indicator_1_direct_investment.csv  # 指标1结果
- indicator_2_network_size.csv       # 指标2结果  
- combined_network_analysis.csv      # 综合分析结果
```

## 使用示例

### 1. 基本使用
```python
from network_indicators_construction import NetworkIndicatorBuilder

# 初始化
builder = NetworkIndicatorBuilder()

# 加载数据
builder.load_data('company_data.csv', 'transaction_data.csv')

# 生成报告
report = builder.generate_network_report()

# 导出结果
builder.export_results(report)
```

### 2. 使用示例数据
```python
# 使用内置示例数据
builder.create_sample_data()
builder.build_network_graph()
report = builder.generate_network_report()
```

## 输出结果说明

### 指标1输出字段
- `company_id`: 公司ID
- `company_name`: 公司名称
- `outgoing_investment`: 对外直接投资
- `incoming_investment`: 接受直接投资
- `total_direct_investment`: 总直接投入
- `investment_transactions_count`: 投资交易笔数
- `investment_intensity`: 投资强度(标准化)

### 指标2输出字段
- `company_id`: 公司ID
- `company_name`: 公司名称
- `direct_partners_count`: 直接关联公司数量
- `indirect_partners_count`: 间接关联公司数量
- `total_network_size`: 总网络规模
- `investment_partners`: 投资伙伴数量
- `trading_partners`: 交易伙伴数量
- `service_partners`: 服务伙伴数量
- `degree_centrality`: 度中心性
- `betweenness_centrality`: 介数中心性
- `closeness_centrality`: 接近中心性
- `network_density`: 网络密度

### 综合分析输出
- 包含上述所有字段
- `investment_score`: 投资评分(标准化)
- `network_score`: 网络评分(标准化)  
- `network_influence_score`: 综合网络影响力评分

## 注意事项

1. **数据质量**: 确保输入数据的准确性和完整性
2. **计算效率**: 对于大规模数据，考虑分批处理或并行计算
3. **结果解释**: 指标需要结合行业特点和公司规模进行解释
4. **动态分析**: 可以按年份分别计算，观察网络指标的时间趋势
5. **阈值设定**: 可以设定交易金额阈值，过滤小额交易的影响

## 扩展功能

1. **可视化**: 使用networkx和matplotlib绘制网络图
2. **行业分析**: 按行业分组计算网络指标
3. **时间序列**: 构建多年度的网络指标变化趋势
4. **风险评估**: 基于网络指标评估公司的关联交易风险
5. **对比分析**: 同行业公司间的网络指标对比

这个框架为上市公司网络指标的构建提供了完整的解决方案，可以根据具体需求进行调整和扩展。