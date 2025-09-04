"""
数字经济与企业网络协同的需求冲击抑制效应分析模型
==============================================

本模型专门分析数字经济使用与企业间网络协同对外部需求波动的抑制效应。
通过多种情景模拟，深入研究数字技术采用和企业网络如何帮助企业应对需求冲击。

主要功能：
- 多维度数字经济指标体系
- 企业间网络协同效应建模
- 需求冲击传播与抑制机制分析
- 情景对比分析与政策建议
- 可视化结果展示与报告生成

作者：数字经济研究团队
版本：1.0
日期：2025年1月
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')
np.random.seed(2025)

# 全局配置
OUTPUT_DIR = Path("数字经济分析结果")
OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass
class 数字经济指标:
    """数字经济使用指标体系"""
    数字技术采用率: float = 0.0  # 数字技术采用水平
    电子商务使用率: float = 0.0  # 电子商务平台使用
    数据分析能力: float = 0.0    # 数据分析与应用能力
    自动化程度: float = 0.0      # 生产自动化水平
    在线协作能力: float = 0.0    # 在线协作与远程办公
    数字营销水平: float = 0.0    # 数字营销与客户管理
    供应链数字化: float = 0.0    # 供应链数字化程度
    
    def 计算综合指数(self) -> float:
        """计算数字经济综合使用指数"""
        指标权重 = [0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1]
        指标值 = [self.数字技术采用率, self.电子商务使用率, self.数据分析能力,
                self.自动化程度, self.在线协作能力, self.数字营销水平, self.供应链数字化]
        return sum(w * v for w, v in zip(指标权重, 指标值))

@dataclass
class 企业网络指标:
    """企业间网络协同指标"""
    合作伙伴数量: int = 0         # 合作伙伴数量
    网络中心度: float = 0.0      # 在网络中的中心度
    信息共享程度: float = 0.0    # 信息共享水平
    资源共享程度: float = 0.0    # 资源共享水平
    协同创新能力: float = 0.0    # 协同创新水平
    风险分担能力: float = 0.0    # 风险分担机制
    
    def 计算网络协同指数(self) -> float:
        """计算网络协同综合指数"""
        标准化合作伙伴数 = min(self.合作伙伴数量 / 20, 1.0)  # 标准化到0-1
        return (标准化合作伙伴数 * 0.2 + self.网络中心度 * 0.2 + 
                self.信息共享程度 * 0.2 + self.资源共享程度 * 0.15 +
                self.协同创新能力 * 0.15 + self.风险分担能力 * 0.1)

@dataclass
class 冲击情景配置:
    """需求冲击情景配置"""
    情景名称: str
    冲击类型: str  # '全面冲击', '行业冲击', '区域冲击', '级联冲击'
    冲击强度: float = 0.3
    冲击持续期: int = 3
    恢复速度: float = 0.1
    受影响企业: List[int] = field(default_factory=list)
    冲击传播系数: float = 0.5

class 需求冲击生成器:
    """生成各类需求冲击场景"""
    
    def __init__(self, 企业数量: int):
        self.企业数量 = 企业数量
    
    def 生成全面冲击(self, 强度: float, 持续期: int) -> np.ndarray:
        """生成影响所有企业的全面需求冲击"""
        冲击矩阵 = np.ones((持续期, self.企业数量))
        for t in range(持续期):
            # 指数衰减的冲击强度
            衰减系数 = np.exp(-0.2 * t)
            冲击矩阵[t, :] = 1 - 强度 * 衰减系数
        return 冲击矩阵
    
    def 生成行业冲击(self, 强度: float, 持续期: int, 受影响行业: List[int]) -> np.ndarray:
        """生成特定行业的需求冲击"""
        冲击矩阵 = np.ones((持续期, self.企业数量))
        for t in range(持续期):
            衰减系数 = np.exp(-0.15 * t)
            for 企业id in 受影响行业:
                if 企业id < self.企业数量:
                    冲击矩阵[t, 企业id] = 1 - 强度 * 衰减系数
        return 冲击矩阵
    
    def 生成级联冲击(self, 强度: float, 持续期: int, 网络权重: np.ndarray, 
                   初始企业: List[int]) -> np.ndarray:
        """生成通过网络传播的级联冲击"""
        冲击矩阵 = np.ones((持续期, self.企业数量))
        
        # 初始冲击
        for 企业 in 初始企业:
            if 企业 < self.企业数量:
                冲击矩阵[0, 企业] = 1 - 强度
        
        # 级联传播
        for t in range(1, 持续期):
            for i in range(self.企业数量):
                if 冲击矩阵[t-1, i] < 1.0:  # 如果企业受到冲击
                    for j in range(self.企业数量):
                        if 网络权重[i, j] > 0.1:  # 强连接
                            传播强度 = 网络权重[i, j] * 0.4
                            冲击矩阵[t, j] = min(冲击矩阵[t, j], 
                                               1 - 强度 * 传播强度 * np.exp(-0.1 * t))
        return 冲击矩阵

class 数字经济抑制效应分析器:
    """分析数字经济对需求冲击的抑制效应"""
    
    def __init__(self):
        self.分析结果 = {}
    
    def 计算数字技术缓冲效应(self, 数字指标: 数字经济指标, 冲击强度: float) -> float:
        """计算数字技术对冲击的缓冲效应"""
        数字综合指数 = 数字指标.计算综合指数()
        
        # 不同数字技术的缓冲机制
        电商缓冲 = 数字指标.电子商务使用率 * 0.3  # 电商平台分散风险
        数据分析缓冲 = 数字指标.数据分析能力 * 0.25  # 数据驱动决策
        自动化缓冲 = 数字指标.自动化程度 * 0.2   # 降低人工依赖
        在线协作缓冲 = 数字指标.在线协作能力 * 0.15  # 远程协作能力
        数字营销缓冲 = 数字指标.数字营销水平 * 0.1   # 精准营销
        
        总缓冲效应 = (电商缓冲 + 数据分析缓冲 + 自动化缓冲 + 
                    在线协作缓冲 + 数字营销缓冲)
        
        # 缓冲效应随冲击强度递减
        实际缓冲 = 总缓冲效应 * (1 - 冲击强度 * 0.5)
        return min(实际缓冲, 0.8)  # 最大缓冲80%
    
    def 计算网络协同缓冲效应(self, 网络指标: 企业网络指标, 冲击强度: float) -> float:
        """计算企业网络协同对冲击的缓冲效应"""
        网络协同指数 = 网络指标.计算网络协同指数()
        
        # 不同网络机制的缓冲效应
        信息共享缓冲 = 网络指标.信息共享程度 * 0.3   # 信息共享降低不确定性
        资源共享缓冲 = 网络指标.资源共享程度 * 0.25  # 资源互补
        风险分担缓冲 = 网络指标.风险分担能力 * 0.25  # 风险分散
        协同创新缓冲 = 网络指标.协同创新能力 * 0.2   # 创新应对
        
        总缓冲效应 = (信息共享缓冲 + 资源共享缓冲 + 风险分担缓冲 + 协同创新缓冲)
        
        # 网络效应在强冲击下可能减弱
        实际缓冲 = 总缓冲效应 * (1 - 冲击强度 * 0.3)
        return min(实际缓冲, 0.7)  # 最大缓冲70%
    
    def 计算协同增强效应(self, 数字指标: 数字经济指标, 网络指标: 企业网络指标) -> float:
        """计算数字经济与网络协同的增强效应"""
        数字指数 = 数字指标.计算综合指数()
        网络指数 = 网络指标.计算网络协同指数()
        
        # 协同效应：数字技术增强网络效率
        数字网络协同 = 数字指标.在线协作能力 * 网络指标.信息共享程度 * 0.3
        数据驱动网络 = 数字指标.数据分析能力 * 网络指标.协同创新能力 * 0.25
        平台化协作 = 数字指标.电子商务使用率 * 网络指标.资源共享程度 * 0.2
        
        协同增强 = 数字网络协同 + 数据驱动网络 + 平台化协作
        return min(协同增强, 0.5)  # 最大增强50%

class 数字经济冲击分析模型:
    """数字经济需求冲击分析主模型"""
    
    def __init__(self, 企业数量: int = 20, 时间周期: int = 10):
        self.企业数量 = 企业数量
        self.时间周期 = 时间周期
        self.冲击生成器 = 需求冲击生成器(企业数量)
        self.抑制效应分析器 = 数字经济抑制效应分析器()
        
        # 初始化企业数据
        self._初始化企业数据()
        
        # 存储分析结果
        self.模拟结果 = defaultdict(list)
    
    def _初始化企业数据(self):
        """初始化企业的数字经济和网络指标"""
        self.企业数字指标 = []
        self.企业网络指标 = []
        
        for i in range(self.企业数量):
            # 生成企业数字经济指标（正态分布）
            数字指标 = 数字经济指标(
                数字技术采用率=max(0, min(1, np.random.normal(0.5, 0.2))),
                电子商务使用率=max(0, min(1, np.random.normal(0.4, 0.25))),
                数据分析能力=max(0, min(1, np.random.normal(0.3, 0.2))),
                自动化程度=max(0, min(1, np.random.normal(0.35, 0.2))),
                在线协作能力=max(0, min(1, np.random.normal(0.6, 0.2))),
                数字营销水平=max(0, min(1, np.random.normal(0.4, 0.2))),
                供应链数字化=max(0, min(1, np.random.normal(0.3, 0.2)))
            )
            self.企业数字指标.append(数字指标)
            
            # 生成企业网络指标
            网络指标 = 企业网络指标(
                合作伙伴数量=max(1, int(np.random.poisson(8))),
                网络中心度=max(0, min(1, np.random.beta(2, 5))),
                信息共享程度=max(0, min(1, np.random.normal(0.4, 0.2))),
                资源共享程度=max(0, min(1, np.random.normal(0.3, 0.2))),
                协同创新能力=max(0, min(1, np.random.normal(0.25, 0.15))),
                风险分担能力=max(0, min(1, np.random.normal(0.35, 0.2)))
            )
            self.企业网络指标.append(网络指标)
        
        # 生成企业间网络连接矩阵
        self.网络连接矩阵 = self._生成网络连接矩阵()
    
    def _生成网络连接矩阵(self) -> np.ndarray:
        """生成企业间网络连接权重矩阵"""
        # 基于小世界网络模型
        G = nx.watts_strogatz_graph(self.企业数量, 6, 0.3)
        网络矩阵 = nx.adjacency_matrix(G).toarray().astype(float)
        
        # 添加权重（基于企业特征相似性）
        for i in range(self.企业数量):
            for j in range(i+1, self.企业数量):
                if 网络矩阵[i, j] > 0:
                    # 计算企业相似性
                    数字相似性 = 1 - abs(self.企业数字指标[i].计算综合指数() - 
                                      self.企业数字指标[j].计算综合指数())
                    网络相似性 = 1 - abs(self.企业网络指标[i].计算网络协同指数() - 
                                      self.企业网络指标[j].计算网络协同指数())
                    
                    权重 = (数字相似性 + 网络相似性) / 2 * np.random.uniform(0.3, 1.0)
                    网络矩阵[i, j] = 网络矩阵[j, i] = 权重
        
        return 网络矩阵
    
    def 模拟情景分析(self, 冲击配置: 冲击情景配置) -> Dict:
        """执行单个情景的模拟分析"""
        print(f"🔍 正在分析情景：{冲击配置.情景名称}")
        print(f"   冲击类型：{冲击配置.冲击类型}")
        print(f"   冲击强度：{冲击配置.冲击强度:.1%}")
        
        # 生成冲击模式
        if 冲击配置.冲击类型 == '全面冲击':
            冲击矩阵 = self.冲击生成器.生成全面冲击(
                冲击配置.冲击强度, 冲击配置.冲击持续期)
        elif 冲击配置.冲击类型 == '行业冲击':
            冲击矩阵 = self.冲击生成器.生成行业冲击(
                冲击配置.冲击强度, 冲击配置.冲击持续期, 冲击配置.受影响企业)
        elif 冲击配置.冲击类型 == '级联冲击':
            冲击矩阵 = self.冲击生成器.生成级联冲击(
                冲击配置.冲击强度, 冲击配置.冲击持续期, 
                self.网络连接矩阵, 冲击配置.受影响企业)
        
        # 模拟时间序列
        时间序列结果 = {
            '原始冲击': [],
            '数字缓冲后': [],
            '网络缓冲后': [],
            '协同缓冲后': [],
            '企业表现': []
        }
        
        for t in range(self.时间周期):
            if t < len(冲击矩阵):
                原始冲击 = 冲击矩阵[t, :]
            else:
                # 恢复期
                恢复因子 = np.exp(-冲击配置.恢复速度 * (t - len(冲击矩阵)))
                原始冲击 = 1.0 - (1.0 - 冲击矩阵[-1, :]) * 恢复因子
            
            # 计算各种缓冲效应
            数字缓冲冲击 = np.zeros(self.企业数量)
            网络缓冲冲击 = np.zeros(self.企业数量)
            协同缓冲冲击 = np.zeros(self.企业数量)
            
            for i in range(self.企业数量):
                冲击强度 = 1 - 原始冲击[i]
                
                # 数字技术缓冲
                数字缓冲率 = self.抑制效应分析器.计算数字技术缓冲效应(
                    self.企业数字指标[i], 冲击强度)
                数字缓冲冲击[i] = 原始冲击[i] + (1 - 原始冲击[i]) * 数字缓冲率
                
                # 网络协同缓冲
                网络缓冲率 = self.抑制效应分析器.计算网络协同缓冲效应(
                    self.企业网络指标[i], 冲击强度)
                网络缓冲冲击[i] = 原始冲击[i] + (1 - 原始冲击[i]) * 网络缓冲率
                
                # 协同增强缓冲
                协同增强率 = self.抑制效应分析器.计算协同增强效应(
                    self.企业数字指标[i], self.企业网络指标[i])
                总缓冲率 = min(数字缓冲率 + 网络缓冲率 + 协同增强率, 0.9)
                协同缓冲冲击[i] = 原始冲击[i] + (1 - 原始冲击[i]) * 总缓冲率
            
            # 存储结果
            时间序列结果['原始冲击'].append(原始冲击.copy())
            时间序列结果['数字缓冲后'].append(数字缓冲冲击.copy())
            时间序列结果['网络缓冲后'].append(网络缓冲冲击.copy())
            时间序列结果['协同缓冲后'].append(协同缓冲冲击.copy())
            
            # 计算企业表现指标
            企业表现 = self._计算企业表现(协同缓冲冲击, t)
            时间序列结果['企业表现'].append(企业表现)
            
            print(f"   第{t}期：平均需求水平 {np.mean(协同缓冲冲击):.3f}, "
                  f"平均企业表现 {np.mean(企业表现):.3f}")
        
        return {
            '配置': 冲击配置,
            '时间序列': 时间序列结果,
            '抑制效应统计': self._计算抑制效应统计(时间序列结果)
        }
    
    def _计算企业表现(self, 需求水平: np.ndarray, 时期: int) -> np.ndarray:
        """计算企业综合表现指标"""
        企业表现 = np.zeros(self.企业数量)
        
        for i in range(self.企业数量):
            # 基础表现受需求影响
            基础表现 = 需求水平[i] * 0.7
            
            # 数字经济提升表现
            数字提升 = self.企业数字指标[i].计算综合指数() * 0.2
            
            # 网络协同提升表现
            网络提升 = self.企业网络指标[i].计算网络协同指数() * 0.1
            
            企业表现[i] = 基础表现 + 数字提升 + 网络提升 + np.random.normal(0, 0.05)
            企业表现[i] = max(0, min(1, 企业表现[i]))
        
        return 企业表现
    
    def _计算抑制效应统计(self, 时间序列结果: Dict) -> Dict:
        """计算抑制效应统计指标"""
        原始冲击数据 = np.array(时间序列结果['原始冲击'])
        数字缓冲数据 = np.array(时间序列结果['数字缓冲后'])
        网络缓冲数据 = np.array(时间序列结果['网络缓冲后'])
        协同缓冲数据 = np.array(时间序列结果['协同缓冲后'])
        
        统计结果 = {
            '数字技术抑制率': np.mean((数字缓冲数据 - 原始冲击数据) / (1 - 原始冲击数据 + 1e-8)),
            '网络协同抑制率': np.mean((网络缓冲数据 - 原始冲击数据) / (1 - 原始冲击数据 + 1e-8)),
            '协同增强抑制率': np.mean((协同缓冲数据 - 原始冲击数据) / (1 - 原始冲击数据 + 1e-8)),
            '波动性降低': {
                '原始波动': np.std(原始冲击数据),
                '协同后波动': np.std(协同缓冲数据),
                '波动降低率': 1 - np.std(协同缓冲数据) / np.std(原始冲击数据)
            },
            '恢复速度提升': self._计算恢复速度(原始冲击数据, 协同缓冲数据)
        }
        
        return 统计结果
    
    def _计算恢复速度(self, 原始数据: np.ndarray, 缓冲数据: np.ndarray) -> Dict:
        """计算冲击恢复速度"""
        # 找到冲击最严重的时期
        最低点_原始 = np.argmin(np.mean(原始数据, axis=1))
        最低点_缓冲 = np.argmin(np.mean(缓冲数据, axis=1))
        
        # 计算恢复到90%正常水平的时间
        def 找恢复时间(数据, 起始点):
            目标水平 = 0.9
            for t in range(起始点, len(数据)):
                if np.mean(数据[t]) >= 目标水平:
                    return t - 起始点
            return len(数据) - 起始点
        
        原始恢复时间 = 找恢复时间(原始数据, 最低点_原始)
        缓冲恢复时间 = 找恢复时间(缓冲数据, 最低点_缓冲)
        
        return {
            '原始恢复时间': 原始恢复时间,
            '缓冲恢复时间': 缓冲恢复时间,
            '恢复速度提升': max(0, (原始恢复时间 - 缓冲恢复时间) / 原始恢复时间) if 原始恢复时间 > 0 else 0
        }

def 创建可视化分析图表(模拟结果: Dict, 保存路径: Path = None):
    """创建综合分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'数字经济需求冲击抑制效应分析：{模拟结果["配置"].情景名称}', 
                 fontsize=16, fontweight='bold')
    
    时间序列 = 模拟结果['时间序列']
    时间轴 = range(len(时间序列['原始冲击']))
    
    # 1. 冲击传播与缓冲效应对比
    ax = axes[0, 0]
    原始平均 = [np.mean(x) for x in 时间序列['原始冲击']]
    数字平均 = [np.mean(x) for x in 时间序列['数字缓冲后']]
    网络平均 = [np.mean(x) for x in 时间序列['网络缓冲后']]
    协同平均 = [np.mean(x) for x in 时间序列['协同缓冲后']]
    
    ax.plot(时间轴, 原始平均, 'r-', linewidth=2, label='原始冲击', marker='o')
    ax.plot(时间轴, 数字平均, 'b--', linewidth=2, label='数字技术缓冲', marker='s')
    ax.plot(时间轴, 网络平均, 'g--', linewidth=2, label='网络协同缓冲', marker='^')
    ax.plot(时间轴, 协同平均, 'purple', linewidth=3, label='协同增强缓冲', marker='*')
    ax.set_title('需求冲击缓冲效应对比')
    ax.set_xlabel('时间周期')
    ax.set_ylabel('平均需求水平')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 企业表现演进
    ax = axes[0, 1]
    企业表现数据 = np.array(时间序列['企业表现'])
    表现平均 = np.mean(企业表现数据, axis=1)
    表现标准差 = np.std(企业表现数据, axis=1)
    
    ax.plot(时间轴, 表现平均, 'navy', linewidth=2, label='平均表现')
    ax.fill_between(时间轴, 表现平均 - 表现标准差, 表现平均 + 表现标准差, 
                    alpha=0.3, label='表现区间')
    ax.set_title('企业表现时间演进')
    ax.set_xlabel('时间周期')
    ax.set_ylabel('企业表现指数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 抑制效应量化
    ax = axes[0, 2]
    抑制统计 = 模拟结果['抑制效应统计']
    抑制类型 = ['数字技术\n抑制', '网络协同\n抑制', '协同增强\n抑制']
    抑制率 = [抑制统计['数字技术抑制率'], 抑制统计['网络协同抑制率'], 
             抑制统计['协同增强抑制率']]
    
    bars = ax.bar(抑制类型, [x*100 for x in 抑制率], 
                  color=['skyblue', 'lightgreen', 'gold'], alpha=0.8)
    ax.set_title('冲击抑制效应量化')
    ax.set_ylabel('抑制率 (%)')
    
    # 添加数值标签
    for bar, rate in zip(bars, 抑制率):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 4. 企业数字化水平分布
    ax = axes[1, 0]
    数字化水平 = [指标.计算综合指数() for 指标 in 模拟结果['模型'].企业数字指标]
    ax.hist(数字化水平, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_title('企业数字化水平分布')
    ax.set_xlabel('数字化综合指数')
    ax.set_ylabel('企业数量')
    
    # 5. 网络协同能力分布
    ax = axes[1, 1]
    网络协同水平 = [指标.计算网络协同指数() for 指标 in 模拟结果['模型'].企业网络指标]
    ax.hist(网络协同水平, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_title('企业网络协同能力分布')
    ax.set_xlabel('网络协同指数')
    ax.set_ylabel('企业数量')
    
    # 6. 数字化与网络协同的关系
    ax = axes[1, 2]
    ax.scatter(数字化水平, 网络协同水平, alpha=0.6, s=60)
    
    # 添加趋势线
    z = np.polyfit(数字化水平, 网络协同水平, 1)
    p = np.poly1d(z)
    ax.plot(数字化水平, p(数字化水平), "r--", alpha=0.8)
    
    ax.set_title('数字化与网络协同关系')
    ax.set_xlabel('数字化水平')
    ax.set_ylabel('网络协同水平')
    
    # 计算相关系数
    相关系数 = np.corrcoef(数字化水平, 网络协同水平)[0, 1]
    ax.text(0.05, 0.95, f'相关系数: {相关系数:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    
    if 保存路径:
        plt.savefig(保存路径, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存至：{保存路径}")
    
    plt.show()

def 生成分析报告(所有结果: Dict, 保存目录: Path = OUTPUT_DIR):
    """生成综合分析报告"""
    报告内容 = []
    报告内容.append("# 数字经济与企业网络协同的需求冲击抑制效应分析报告")
    报告内容.append(f"\n**生成时间：** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    报告内容.append("\n---\n")
    
    报告内容.append("## 1. 执行摘要")
    报告内容.append("本研究通过多情景模拟分析，深入探讨了数字经济使用与企业间网络协同")
    报告内容.append("对外部需求波动的抑制效应。研究发现：")
    
    # 计算总体统计
    总体抑制率 = []
    for 情景名, 结果 in 所有结果.items():
        总体抑制率.append(结果['抑制效应统计']['协同增强抑制率'])
    
    平均抑制率 = np.mean(总体抑制率)
    报告内容.append(f"- 数字经济与网络协同的综合抑制效应平均达到 **{平均抑制率:.1%}**")
    报告内容.append(f"- 不同情景下的抑制效应存在显著差异，范围在 {min(总体抑制率):.1%} 到 {max(总体抑制率):.1%} 之间")
    
    报告内容.append("\n## 2. 分情景分析结果")
    
    for 情景名, 结果 in 所有结果.items():
        报告内容.append(f"\n### 2.{list(所有结果.keys()).index(情景名)+1} {情景名}")
        
        配置 = 结果['配置']
        统计 = 结果['抑制效应统计']
        
        报告内容.append(f"**情景设置：** {配置.冲击类型}，强度{配置.冲击强度:.1%}，持续{配置.冲击持续期}期")
        报告内容.append(f"**主要发现：**")
        报告内容.append(f"- 数字技术抑制率：{统计['数字技术抑制率']:.1%}")
        报告内容.append(f"- 网络协同抑制率：{统计['网络协同抑制率']:.1%}")
        报告内容.append(f"- 协同增强抑制率：{统计['协同增强抑制率']:.1%}")
        报告内容.append(f"- 波动性降低：{统计['波动性降低']['波动降低率']:.1%}")
        报告内容.append(f"- 恢复速度提升：{统计['恢复速度提升']['恢复速度提升']:.1%}")
    
    报告内容.append("\n## 3. 政策建议")
    报告内容.append("基于分析结果，我们提出以下政策建议：")
    报告内容.append("1. **加强数字基础设施建设**：提升企业数字技术采用的基础条件")
    报告内容.append("2. **促进企业间协作网络**：建立行业协作平台，增强信息与资源共享")
    报告内容.append("3. **推动数字化转型**：重点支持中小企业的数字化升级")
    报告内容.append("4. **建立风险预警机制**：利用数字技术提升需求波动的预测能力")
    
    # 保存报告
    报告文件 = 保存目录 / f"需求冲击抑制效应分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(报告文件, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))
    
    print(f"📊 分析报告已生成：{报告文件}")
    return 报告文件

def 执行综合分析():
    """执行完整的数字经济冲击抑制效应分析"""
    print("=" * 80)
    print("🚀 数字经济与企业网络协同的需求冲击抑制效应分析")
    print("=" * 80)
    
    # 创建分析模型
    模型 = 数字经济冲击分析模型(企业数量=25, 时间周期=12)
    
    # 定义分析情景
    分析情景 = [
        冲击情景配置(
            情景名称="全面需求萎缩情景",
            冲击类型="全面冲击",
            冲击强度=0.4,
            冲击持续期=4,
            恢复速度=0.15
        ),
        冲击情景配置(
            情景名称="重点行业冲击情景", 
            冲击类型="行业冲击",
            冲击强度=0.5,
            冲击持续期=3,
            恢复速度=0.2,
            受影响企业=list(range(0, 8))  # 前8家企业
        ),
        冲击情景配置(
            情景名称="供应链级联冲击情景",
            冲击类型="级联冲击", 
            冲击强度=0.35,
            冲击持续期=5,
            恢复速度=0.1,
            受影响企业=[0, 5, 12],  # 关键节点企业
            冲击传播系数=0.6
        ),
        冲击情景配置(
            情景名称="温和持续冲击情景",
            冲击类型="全面冲击",
            冲击强度=0.2,
            冲击持续期=6,
            恢复速度=0.05
        )
    ]
    
    # 执行各情景分析
    所有分析结果 = {}
    
    for 情景配置 in 分析情景:
        print(f"\n{'='*60}")
        print(f"开始分析：{情景配置.情景名称}")
        print(f"{'='*60}")
        
        # 执行模拟
        结果 = 模型.模拟情景分析(情景配置)
        结果['模型'] = 模型  # 保存模型引用用于可视化
        
        # 创建可视化图表
        图表路径 = OUTPUT_DIR / f"{情景配置.情景名称}_分析图表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        创建可视化分析图表(结果, 图表路径)
        
        所有分析结果[情景配置.情景名称] = 结果
        print(f"✅ {情景配置.情景名称} 分析完成")
    
    # 生成对比分析
    print(f"\n{'='*60}")
    print("生成综合对比分析")
    print(f"{'='*60}")
    
    创建情景对比图表(所有分析结果)
    
    # 生成分析报告
    生成分析报告(所有分析结果)
    
    print("\n" + "=" * 80)
    print("✅ 数字经济需求冲击抑制效应分析完成！")
    print("📊 所有分析结果已保存至：", OUTPUT_DIR)
    print("🔍 请查看生成的图表和报告文件")
    print("=" * 80)
    
    return 所有分析结果

def 创建情景对比图表(所有结果: Dict):
    """创建多情景对比分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('数字经济需求冲击抑制效应：多情景对比分析', fontsize=16, fontweight='bold')
    
    情景名称列表 = list(所有结果.keys())
    颜色列表 = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # 1. 不同情景下的抑制效应对比
    ax = axes[0, 0]
    抑制类型 = ['数字技术', '网络协同', '协同增强']
    x = np.arange(len(抑制类型))
    width = 0.8 / len(情景名称列表)
    
    for i, (情景名, 结果) in enumerate(所有结果.items()):
        统计 = 结果['抑制效应统计']
        抑制率 = [统计['数字技术抑制率'], 统计['网络协同抑制率'], 统计['协同增强抑制率']]
        ax.bar(x + i * width, [r*100 for r in 抑制率], width, 
               label=情景名, alpha=0.8, color=颜色列表[i])
    
    ax.set_title('不同情景下的抑制效应对比')
    ax.set_ylabel('抑制率 (%)')
    ax.set_xticks(x + width * (len(情景名称列表) - 1) / 2)
    ax.set_xticklabels(抑制类型)
    ax.legend()
    
    # 2. 波动性降低效果对比
    ax = axes[0, 1]
    波动降低率 = []
    for 情景名, 结果 in 所有结果.items():
        波动降低率.append(结果['抑制效应统计']['波动性降低']['波动降低率'] * 100)
    
    bars = ax.bar(range(len(情景名称列表)), 波动降低率, 
                  color=颜色列表[:len(情景名称列表)], alpha=0.7)
    ax.set_title('波动性降低效果对比')
    ax.set_ylabel('波动降低率 (%)')
    ax.set_xticks(range(len(情景名称列表)))
    ax.set_xticklabels([name[:6]+'...' if len(name)>6 else name for name in 情景名称列表], rotation=45)
    
    # 添加数值标签
    for bar, rate in zip(bars, 波动降低率):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. 恢复速度提升对比
    ax = axes[1, 0]
    恢复提升率 = []
    for 情景名, 结果 in 所有结果.items():
        恢复提升率.append(结果['抑制效应统计']['恢复速度提升']['恢复速度提升'] * 100)
    
    bars = ax.bar(range(len(情景名称列表)), 恢复提升率,
                  color=颜色列表[:len(情景名称列表)], alpha=0.7)
    ax.set_title('恢复速度提升对比')
    ax.set_ylabel('恢复速度提升率 (%)')
    ax.set_xticks(range(len(情景名称列表)))
    ax.set_xticklabels([name[:6]+'...' if len(name)>6 else name for name in 情景名称列表], rotation=45)
    
    # 添加数值标签
    for bar, rate in zip(bars, 恢复提升率):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. 时间序列对比（选择协同增强缓冲效果）
    ax = axes[1, 1]
    for i, (情景名, 结果) in enumerate(所有结果.items()):
        时间序列 = 结果['时间序列']
        协同平均 = [np.mean(x) for x in 时间序列['协同缓冲后']]
        时间轴 = range(len(协同平均))
        ax.plot(时间轴, 协同平均, linewidth=2, marker='o', 
                label=情景名[:8]+'...' if len(情景名)>8 else 情景名, 
                color=颜色列表[i])
    
    ax.set_title('不同情景下需求恢复路径')
    ax.set_xlabel('时间周期')
    ax.set_ylabel('平均需求水平')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图表
    对比图表路径 = OUTPUT_DIR / f"多情景对比分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(对比图表路径, dpi=300, bbox_inches='tight')
    print(f"✅ 对比分析图表已保存：{对比图表路径}")
    
    plt.show()

if __name__ == "__main__":
    # 执行完整分析
    分析结果 = 执行综合分析()